#!/usr/bin/env python3
"""
Regression test for the per-step decoder conditioning (`dec_t_scale`).

The whole point of zero-initialising the last layer of `t_proj` is that turning
the flag on changes nothing until the new parameters are trained: a checkpoint
trained without conditioning must resume and produce bitwise identical outputs.
If that ever stops holding, every "before/after" comparison built on top of it
is worthless, so it is worth a test.

    python tools/ab/test_dec_t_scale.py                  # block-level only
    python tools/ab/test_dec_t_scale.py --ckpt <path>     # plus the full model

Checks:
  1. DecoderBlock(t_scale=True) fed a step index == DecoderBlock(t_scale=False),
     exactly, when the shared weights are the same.
  2. `t` reaches the block at all: perturb t_proj away from zero and the output
     must move.  Guards against the modulation being silently dropped (`t=None`
     is exactly what the encoder's own t_scale path suffers from today --
     `_run_recurrent_encoder_blocks` calls blk(x, pos) and never passes t).
  3. Every new tensor is a t_proj tensor, and there are 4 per conditioned block.
  4. With --ckpt: the full model, same weights, dec_t_scale off vs on, same
     input -> same pts3d / conf / pose.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

import dust3r.utils.path_to_croco  # noqa: F401
from models.blocks import DecoderBlock  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def block_tests(dim=64, heads=4, b=2, n=7, m=5, seed=0):
    print("\n[1] DecoderBlock, t_scale off vs on")
    torch.manual_seed(seed)
    plain = DecoderBlock(dim, heads, qkv_bias=True).eval()
    torch.manual_seed(seed)
    cond = DecoderBlock(dim, heads, qkv_bias=True, t_scale=True).eval()

    # the shared weights are built in the same order under the same seed, but
    # copy them anyway so the test does not depend on that
    missing, unexpected = cond.load_state_dict(plain.state_dict(), strict=False)
    check("no unexpected keys loading plain -> conditioned", not unexpected, str(unexpected))
    check(
        "new tensors are exactly the 4 t_proj tensors",
        sorted(missing) == ["t_proj.0.bias", "t_proj.0.weight", "t_proj.2.bias", "t_proj.2.weight"],
        str(sorted(missing)),
    )

    torch.manual_seed(1)
    x = torch.randn(b, n, dim)
    y = torch.randn(b, m, dim)

    with torch.no_grad():
        ref, _ = plain(x, y, None, None)
        for t in (0.0, 1.0 / 3, 1.0, torch.full((b,), 0.5)):
            out, _ = cond(x, y, None, None, t)
            d = (out - ref).abs().max().item()
            label = "tensor" if torch.is_tensor(t) else f"{t:.3f}"
            check(f"identity at init, t={label}", d == 0.0, f"max abs diff {d:.3e}")
        out_none, _ = cond(x, y, None, None, None)
        check("t=None is the plain path", (out_none - ref).abs().max().item() == 0.0)

    print("\n[1b] gradients reach the new parameters")
    torch.manual_seed(seed)
    grad_blk = DecoderBlock(dim, heads, qkv_bias=True, t_scale=True).train()
    out, _ = grad_blk(x, y, None, None, 0.25)
    out.square().mean().backward()
    gw = grad_blk.t_proj[-1].weight.grad
    gb = grad_blk.t_proj[-1].bias.grad
    check("the output layer receives a gradient at step 1",
          gw is not None and float(gw.abs().max()) > 0,
          f"max |grad| {float(gw.abs().max()):.3e}" if gw is not None else "grad is None")
    check("its gradient is not uniform across channels (no symmetry trap)",
          float(gw.std()) > 0, f"std {float(gw.std()):.3e}")
    gh = grad_blk.t_proj[0].weight.grad
    print(f"       hidden layer grad max |.| {float(gh.abs().max()):.3e}"
          f"  (zero at step 1 by construction; nonzero once the output layer moves)")

    # t must actually do something once trained
    with torch.no_grad():
        torch.nn.init.normal_(cond.t_proj[-1].weight, std=0.5)
        o0, _ = cond(x, y, None, None, 0.0)
        o1, _ = cond(x, y, None, None, 1.0)
    check("a trained t_proj makes step 0 differ from step 1", (o1 - o0).abs().max().item() > 1e-3,
          f"max abs diff {(o1 - o0).abs().max().item():.3e}")
    check("a trained t_proj moves the output away from the plain block",
          (o0 - ref).abs().max().item() > 1e-3)

    print("\n[2] parameter cost at dim=768")
    big = DecoderBlock(768, 12, qkv_bias=True, t_scale=True)
    extra = sum(p.numel() for n_, p in big.named_parameters() if n_.startswith("t_proj"))
    total = sum(p.numel() for p in big.parameters())
    print(f"       t_proj {extra:,} of {total:,} per block ({100 * extra / total:.2f}%)")
    print(f"       6 blocks (3 img + 3 state): {6 * extra:,}")
    check("t_proj is 4 * dim wide", big.t_proj[-1].out_features == 4 * 768)


def model_test(ckpt, model_cfg, device, seq, nframes, size):
    """Same weights, flag off vs on, the real inference path, same frames."""
    repo = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    sys.path.insert(0, repo)
    os.chdir(repo)
    from add_ckpt_path import add_path_to_dust3r

    add_path_to_dust3r(ckpt)
    import demo as D
    from src.dust3r.model import ARCroco3DStereo, ARCroco3DStereoConfig
    from src.dust3r.inference import inference_recurrent
    from src.dust3r.utils.safe_eval import safe_eval_expr

    print(f"\n[3] full model, {ckpt}")
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg_str = model_cfg or getattr(sd.get("args", None), "model", None)
    if not cfg_str:
        raise SystemExit("no model config embedded in the checkpoint; pass --model-cfg")
    weights = sd.get("model", sd)

    def build(flag):
        if "dec_t_scale" in cfg_str:
            s = cfg_str.replace("dec_t_scale=True", f"dec_t_scale={flag}")
            s = s.replace("dec_t_scale=False", f"dec_t_scale={flag}")
        else:
            s = cfg_str.replace("ARCroco3DStereoConfig(", f"ARCroco3DStereoConfig(dec_t_scale={flag}, ")
        net = safe_eval_expr(
            s,
            {
                "ARCroco3DStereo": ARCroco3DStereo,
                "ARCroco3DStereoConfig": ARCroco3DStereoConfig,
                "inf": float("inf"),
            },
        )
        rep = net.load_state_dict(weights, strict=False)
        return net.to(device).eval(), rep

    off, r_off = build(False)
    on, r_on = build(True)
    print(f"       off: {len(r_off.missing_keys)} missing / {len(r_off.unexpected_keys)} unexpected")
    print(f"       on : {len(r_on.missing_keys)} missing / {len(r_on.unexpected_keys)} unexpected")
    check("flag off loads cleanly", not r_off.missing_keys and not r_off.unexpected_keys)
    check("flag on reports no unexpected keys", not r_on.unexpected_keys)
    extra = [k for k in r_on.missing_keys if k not in r_off.missing_keys]
    check("flag on adds only t_proj keys", extra and all(".t_proj." in k for k in extra), str(extra[:2]))
    want = 4 * (len(on.dec_blocks) + len(on.dec_blocks_state))
    check(
        "4 new tensors per conditioned block",
        len(extra) == want,
        f"{len(extra)} new, {len(on.dec_blocks)} img + {len(on.dec_blocks_state)} state blocks",
    )
    # The check that caught the real bug: CroCoNet.initialize_weights() runs
    # self.apply(self._init_weights) and xavier-inits every nn.Linear, t_proj
    # included, so the zero init inside DecoderBlock.__init__ does not survive
    # construction.  Without reset_t_proj() the final layer comes out at
    # +/- sqrt(6 / (16 + 4 * dim)) = 4.4e-2 and the flag is not the identity.
    blocks = list(on.dec_blocks) + list(on.dec_blocks_state)
    worst_w = max(float(b.t_proj[-1].weight.detach().abs().max()) for b in blocks)
    worst_b = max(float(b.t_proj[-1].bias.detach().abs().max()) for b in blocks)
    check("t_proj output layer survived the global init pass as zeros",
          worst_w == 0.0 and worst_b == 0.0,
          f"max |w| {worst_w:.3e}, max |b| {worst_b:.3e}")
    hid = max(float(b.t_proj[0].weight.detach().abs().max()) for b in blocks)
    check("t_proj hidden layer is NOT zero (it must break symmetry)", hid > 0.0,
          f"max |w| {hid:.3e}")

    n_off = sum(p.numel() for p in off.parameters())
    n_on = sum(p.numel() for p in on.parameters())
    print(f"       params {n_off:,} -> {n_on:,}  (+{n_on - n_off:,}, {100 * (n_on - n_off) / n_off:.3f}%)")
    check("the decoder is weight-shared, not 12 distinct blocks",
          len(on.dec_blocks) < on.dec_num_steps_img or len(on.dec_blocks) <= 3,
          f"{len(on.dec_blocks)} blocks over {on.dec_num_steps_img} steps")

    frames = sorted(
        os.path.join(seq, f) for f in os.listdir(seq)
        if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png")
    )[:nframes]
    check(f"found frames in {seq}", len(frames) >= 2, f"{len(frames)} frames")
    if len(frames) < 2:
        return
    print(f"       {len(frames)} frames at {size}px")

    outs = {}
    for name, net in (("off", off), ("on", on)):
        views = D.prepare_input(frames, [True] * len(frames), size)
        with torch.no_grad():
            res, _ = inference_recurrent(views, net, device, verbose=False)
        outs[name] = [
            {k: v.detach().float().cpu().numpy() for k, v in p.items() if torch.is_tensor(v)}
            for p in res["pred"]
        ]

    check("same number of predictions", len(outs["off"]) == len(outs["on"]))
    keys = sorted(set(outs["off"][0]) & set(outs["on"][0]))
    check("the forward produced comparable tensors", bool(keys), ", ".join(keys))
    worst = {}
    for i in range(min(len(outs["off"]), len(outs["on"]))):
        for k in keys:
            d = float(np.abs(outs["off"][i][k] - outs["on"][i][k]).max())
            worst[k] = max(worst.get(k, 0.0), d)
    for k in keys:
        check(f"bitwise identical across all frames: {k}", worst[k] == 0.0,
              f"max abs diff {worst[k]:.3e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None, help="a training checkpoint, to also test the full model")
    ap.add_argument("--model-cfg", default=None, help="model config string, for a stripped checkpoint")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seq", default="examples/002", help="a directory of frames")
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    block_tests()
    if args.ckpt:
        model_test(args.ckpt, args.model_cfg, args.device, args.seq, args.frames, args.size)

    print()
    if FAILED:
        print(f"FAILED ({len(FAILED)}): " + ", ".join(FAILED))
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
