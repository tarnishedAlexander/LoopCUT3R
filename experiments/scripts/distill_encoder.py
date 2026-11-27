#!/usr/bin/env python3
"""Distil the recurrent encoder onto CUT3R's encoder output.

The encoder diagnostic (experiments/README.md) found the recurrent encoder's
output orthogonal to CUT3R's (token cosine ~0), so CUT3R's decoder cannot read
it.  This trains only the encoder - patch embedding, 3 shared blocks x 8 passes,
final norm - to reproduce CUT3R's normalised encoder output token by token:

  * teacher: CUT3R's 24 blocks, rebuilt in this tree as RecurrentBlocks with
    RoPE, LayerScale fixed at 1 and LayerNorm eps 1e-6 (same function; checked
    against encoder_gap.py dumps with --check_teacher);
  * student: the recurrent encoder with RoPE (enc_rope=True) and gradient
    through all 8 passes (no TBPTL), gradient checkpointing per pass;
  * loss: mean token (1 - cosine) + relative MSE on the final output,
    optionally plus pass k against teacher block 3k (--inter_w);
  * data: the images of a training config's train_dataset (its views are used
    as independent images; no 3D labels are read);
  * validation: cosine to the teacher on the encoder_gap.py image set
    (ScanNet, TUM, Sintel), per dataset.

    python experiments/scripts/distill_encoder.py \
        --config experiments/configs/local/attr_hybrid_full.yaml \
        --student checkpoints/attr_hybrid_full/checkpoints/checkpoint-final.pth \
        --cut3r checkpoints/cut3r_512_dpt_4_64.pth --init student \
        --val_images OUT/images.pt --out_dir checkpoints/distill_enc

--init student   blocks, patch embedding and norm from --student
--init cut3r:9,10,11   the 3 shared blocks from those CUT3R blocks, patch
                       embedding and norm from CUT3R
The output checkpoint holds only encoder-side tensors plus the model config
(enc_rope=True); build a full model from it with init_from_cut3r.py --loop.
"""
import argparse, argparse as _ap, math, os, re, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, "src"), ROOT]
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from omegaconf import OmegaConf

torch.backends.cuda.matmul.allow_tf32 = True
torch.multiprocessing.set_sharing_strategy("file_system")  # as train.py: avoids fd exhaustion with many workers
from dust3r.model import ARCroco3DStereo, ARCroco3DStereoConfig, inf  # noqa: F401
from dust3r.datasets import get_data_loader
from croco.models.blocks import RecurrentBlock

PAIRS = [("attn.in_proj_weight", "attn.qkv.weight"), ("attn.in_proj_bias", "attn.qkv.bias"),
         ("attn.out_proj.weight", "attn.proj.weight"), ("attn.out_proj.bias", "attn.proj.bias")]
CUT2REC = [("norm1.", "norm1."), ("attn.qkv.", "attn.qkv."), ("attn.proj.", "attn.proj."),
           ("norm2.", "norm2."), ("mlp.fc1.", "mlp.0."), ("mlp.fc2.", "mlp.2.")]


def set_field(cfg, key, value):
    pat = re.compile(rf"\b{key}=([^,()]+|\([^()]*\))")
    return pat.sub(f"{key}={value}", cfg, count=1) if pat.search(cfg) else cfg[:-2] + f", {key}={value}))"


def load_sd(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck, {k.removeprefix("module."): v for k, v in ck.get("model", ck).items()}


def cut3r_block_into(blk, csd, i):
    """Copy CUT3R encoder block i into a rope RecurrentBlock (LayerScale = 1)."""
    sd = {}
    for c, r in CUT2REC:
        for k, v in csd.items():
            if k.startswith(f"enc_blocks.{i}.{c}"):
                sd[r + k[len(f"enc_blocks.{i}.{c}"):]] = v
    sd["ls1.gamma"] = torch.ones_like(blk.ls1.gamma); sd["ls2.gamma"] = torch.ones_like(blk.ls2.gamma)
    blk.load_state_dict(sd, strict=True)
    blk.norm1.eps = blk.norm2.eps = 1e-6


class Encoder(torch.nn.Module):
    def __init__(self, patch_embed, blocks, norm, passes, ckpt=False):
        super().__init__()
        self.patch_embed, self.blocks, self.norm, self.passes, self.ckpt = patch_embed, blocks, norm, passes, ckpt

    def forward(self, img, true_shape, every=None, passes=None):
        x, pos = self.patch_embed(img, true_shape=true_shape)
        inter = []
        for p in range(passes or self.passes):
            def run(x, pos=pos):
                for b in self.blocks:
                    x = b(x, pos)
                return x
            x = checkpoint(run, x, use_reentrant=False) if (self.ckpt and self.training) else run(x)
            if every:
                inter.append(x)
        return self.norm(x), inter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="training YAML: model string and train_dataset")
    ap.add_argument("--student", required=True)
    ap.add_argument("--cut3r", required=True)
    ap.add_argument("--init", default="student")
    ap.add_argument("--val_images", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=4, help="samples per step; each has num_views images")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--inter_w", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val_every", type=int, default=2000)
    ap.add_argument("--check_teacher", default=None, help="encoder_gap.py dump of CUT3R on --val_images")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--passes_min", type=int, default=None,
                    help="with --passes_max: each step unrolls a random number of passes in [min, max]")
    ap.add_argument("--passes_max", type=int, default=None)
    ap.add_argument("--val_passes", default=None, help="comma list of pass counts to validate, e.g. 4,6,8,12")
    ap.add_argument("--num_steps", type=int, default=None, help="override the model's passes (e.g. 1)")
    ap.add_argument("--arch", choices=["recurrent", "narrow"], default="recurrent",
                    help="narrow: baseline of --narrow_depth independent blocks of width --narrow_dim, "
                         "random init, projected to 1024")
    ap.add_argument("--narrow_dim", type=int, default=384)
    ap.add_argument("--narrow_depth", type=int, default=24)
    a = ap.parse_args()
    import random
    os.makedirs(a.out_dir, exist_ok=True)
    dev = "cuda"

    cfg_y = OmegaConf.load(a.config)
    model_str = set_field(cfg_y.model, "enc_rope", True)
    if a.num_steps:
        model_str = set_field(set_field(model_str, "num_steps", a.num_steps), "use_tbptl", False)
    model = eval(model_str)
    passes, nblk = int(model.num_steps), len(model.enc_blocks)
    rope = model.rope

    _, ssd = load_sd(a.student)
    _, csd = load_sd(a.cut3r)

    # ---- teacher: CUT3R's encoder in this tree
    t_blocks = torch.nn.ModuleList()
    for i in range(24):
        b = RecurrentBlock(dim=1024, num_heads=16, mlp_ratio=4.0, init_values=1e-5, rope=rope)
        cut3r_block_into(b, csd, i); t_blocks.append(b)
    import copy
    t_embed = copy.deepcopy(model.patch_embed)
    t_embed.load_state_dict({k[len("patch_embed."):]: v for k, v in csd.items() if k.startswith("patch_embed.")})
    t_norm = copy.deepcopy(model.enc_norm)
    t_norm.load_state_dict({k[len("enc_norm."):]: v for k, v in csd.items() if k.startswith("enc_norm.")})
    teacher = Encoder(t_embed, t_blocks, t_norm, passes=1).to(dev).eval().requires_grad_(False)

    # ---- student
    s_embed, s_blocks, s_norm = model.patch_embed, model.enc_blocks, model.enc_norm
    if a.init == "student":
        sd = {}
        for k, v in ssd.items():
            if k.startswith(("patch_embed.", "enc_blocks.", "enc_norm.")):
                for old, new in PAIRS:
                    if k.startswith("enc_blocks.") and k.endswith("." + old):
                        k = k[: -len(old)] + new
                sd[k] = v
        missing = [k for k in sd if not k.startswith(("patch_embed.", "enc_blocks.", "enc_norm."))]
        torch.nn.Module.load_state_dict(model, sd, strict=False)
        got = {k for k in model.state_dict() if k.startswith(("patch_embed.", "enc_blocks.", "enc_norm."))}
        assert got == set(sd), f"student encoder keys do not match: {sorted(got ^ set(sd))[:6]} {missing[:3]}"
    elif a.init.startswith("cut3r:"):
        idx = [int(x) for x in a.init.split(":", 1)[1].split(",")]
        assert len(idx) == nblk, f"need {nblk} CUT3R block indices"
        for b, i in zip(s_blocks, idx):
            cut3r_block_into(b, csd, i)
        s_embed.load_state_dict(t_embed.state_dict()); s_norm.load_state_dict(t_norm.state_dict())
    elif a.arch != "narrow":
        sys.exit(f"unknown --init {a.init}")
    if a.arch == "narrow":
        from dust3r.patch_embed import get_patch_embed
        d = a.narrow_dim
        s_embed = get_patch_embed(type(model.patch_embed).__name__, model.patch_embed.img_size, 16, d)
        s_blocks = torch.nn.ModuleList([RecurrentBlock(dim=d, num_heads=d // 64, mlp_ratio=4.0, init_values=None,
                                                       rope=rope) for _ in range(a.narrow_depth)])
        s_norm = torch.nn.Sequential(torch.nn.LayerNorm(d), torch.nn.Linear(d, 1024))
        passes, nblk = 1, a.narrow_depth
    student = Encoder(s_embed, s_blocks, s_norm, passes=passes, ckpt=True).to(dev)
    # Free host memory before the loader forks its workers: the full state
    # dicts and the model's decoder (~5-7 GB) are not needed any more.
    keep = ("patch_embed_ray_map.", "enc_blocks_ray_map.", "enc_norm_ray_map.", "mask_token",
            "masked_img_token", "masked_ray_map_token")
    ssd = {k: v for k, v in ssd.items() if k.startswith(keep)}
    del csd
    enc_names = ("patch_embed", "enc_blocks", "enc_norm", "rope")
    for name, _ in list(model.named_children()):
        if name not in enc_names:
            setattr(model, name, torch.nn.Identity())
    import gc; gc.collect()
    n_par = sum(p.numel() for p in student.parameters())
    print(f"student {n_par / 1e6:.2f} M params ({nblk} blocks x {passes} passes), init={a.init}, arch={a.arch}")
    val_passes = [int(x) for x in a.val_passes.split(",")] if a.val_passes else [passes]

    val = torch.load(a.val_images, weights_only=False)

    @torch.no_grad()
    def validate():
        student.eval(); res = {}
        for it in val:
            img, ts = it["img"].to(dev), it["true_shape"].reshape(1, 2).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                t, _ = teacher(img, ts)
                for n in val_passes:
                    s, _ = student(img, ts, passes=n)
                    key = it["set"] if len(val_passes) == 1 else f"p{n}/{it['set']}"
                    res.setdefault(key, []).append(F.cosine_similarity(s.float(), t.float(), dim=-1).mean().item())
        student.train()
        return {k: round(sum(v) / len(v), 4) for k, v in res.items()}

    if a.check_teacher:
        ref = torch.load(a.check_teacher, weights_only=False)["acts"]
        cs = []
        with torch.no_grad():
            for it, r in zip(val, ref):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    t, _ = teacher(it["img"].to(dev), it["true_shape"].reshape(1, 2).to(dev))
                cs.append(F.cosine_similarity(t[0].float().cpu(), r[-1].float(), dim=-1).mean().item())
        print(f"teacher vs CUT3R tree: token cosine min {min(cs):.4f} mean {sum(cs) / len(cs):.4f}")
        assert min(cs) > 0.98, "teacher does not reproduce CUT3R's encoder"

    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=a.lr,
                            betas=(0.9, 0.95), weight_decay=0.05)
    step0 = 0
    last = os.path.join(a.out_dir, "last.pth")
    if a.resume and os.path.exists(last):
        ck = torch.load(last, map_location="cpu", weights_only=False)
        if a.arch == "narrow":
            student.load_state_dict(ck["model"])
        else:
            torch.nn.Module.load_state_dict(model, ck["model"], strict=False)
        opt.load_state_dict(ck["opt"]); step0 = ck["step"]
        print("resumed at step", step0)

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        q = (s - a.warmup) / max(1, a.steps - a.warmup)
        return a.min_lr + 0.5 * (a.lr - a.min_lr) * (1 + math.cos(math.pi * q))

    def save(path, step, extra=None):
        if a.arch == "narrow":
            torch.save({"student": student.state_dict(), "step": step, "distill": vars(a), **(extra or {})}, path)
            return
        enc = {k: v for k, v in model.state_dict().items()
               if k.startswith(("patch_embed.", "enc_blocks.", "enc_norm."))}
        for k, v in ssd.items():  # ray-map encoder and mask tokens unchanged from --student
            if k.startswith(("patch_embed_ray_map.", "enc_blocks_ray_map.", "enc_norm_ray_map.",
                             "mask_token", "masked_img_token", "masked_ray_map_token")):
                enc[k] = v
        torch.save({"model": enc, "args": _ap.Namespace(model=model_str), "step": step,
                    "distill": vars(a), **(extra or {})}, path)

    log = open(os.path.join(a.out_dir, "log.txt"), "a")
    v = validate()
    print(f"step {step0} val cos {v}", flush=True); log.write(f"{step0}\tval\t{v}\n"); log.flush()

    cy = OmegaConf.to_container(cfg_y, resolve=True)
    loader = get_data_loader(cy["train_dataset"], batch_size=a.batch, num_workers=a.workers,
                             pin_mem=False, shuffle=True, drop_last=True, accelerator=None)
    step, epoch, t0, run = step0, 0, time.time(), []
    student.train()
    while step < a.steps:
        bs = getattr(loader, "batch_sampler", None)
        for obj in (loader.dataset, bs, getattr(bs, "batch_sampler", None), getattr(bs, "sampler", None),
                    getattr(loader, "sampler", None)):
            if obj is not None and hasattr(obj, "set_epoch"):
                obj.set_epoch(epoch)
        for views in loader:
            if step >= a.steps:
                break
            img = torch.cat([vw["img"] for vw in views]).to(dev, non_blocking=True)
            ts = torch.cat([vw["true_shape"] for vw in views]).to(dev)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    t_out, _ = teacher(img, ts)
                    t_inter = None
                n_pass = random.randint(a.passes_min, a.passes_max) if a.passes_min else None
                s_out, s_inter = student(img, ts, every=bool(a.inter_w), passes=n_pass)
                s, t = s_out.float(), t_out.float()
                loss = (1 - F.cosine_similarity(s, t, dim=-1)).mean() + ((s - t) ** 2).mean() / (t ** 2).mean()
                if a.inter_w:
                    with torch.no_grad():
                        x, pos = teacher.patch_embed(img, true_shape=ts); t_inter = []
                        for i, b in enumerate(teacher.blocks):
                            x = b(x, pos)
                            if (i + 1) % nblk == 0:
                                t_inter.append(x)
                    for si, ti in zip(s_inter[:-1], t_inter[:-1]):
                        loss = loss + a.inter_w * (1 - F.cosine_similarity(si.float(), ti.float(), dim=-1)).mean() / (passes - 1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step(); step += 1
            run.append((loss.item(), F.cosine_similarity(s.detach(), t, dim=-1).mean().item()))
            if step % 100 == 0:
                l, c = torch.tensor(run).mean(0).tolist(); run = []
                ips = 100 * img.shape[0] / (time.time() - t0); t0 = time.time()
                msg = f"step {step} loss {l:.4f} train_cos {c:.4f} lr {lr_at(step):.2e} gnorm {gn:.2f} {ips:.1f} img/s"
                print(msg, flush=True); log.write(msg + "\n"); log.flush()
            if step % a.val_every == 0 or step == a.steps:
                v = validate()
                print(f"step {step} val cos {v}", flush=True); log.write(f"{step}\tval\t{v}\n"); log.flush()
                save(os.path.join(a.out_dir, "encoder.pth"), step, {"val_cos": v})
                state = (student.state_dict() if a.arch == "narrow" else
                         {k: x for k, x in model.state_dict().items()
                          if k.startswith(("patch_embed.", "enc_blocks.", "enc_norm."))})
                torch.save({"model": state, "opt": opt.state_dict(), "step": step}, last)
        epoch += 1
    print("done", step)


if __name__ == "__main__":
    main()
