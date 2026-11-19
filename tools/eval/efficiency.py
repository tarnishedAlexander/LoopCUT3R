#!/usr/bin/env python3
"""
Parameters, block applications per frame, peak VRAM and FPS of a checkpoint.

Each model runs from ITS OWN source tree (--repo).  Loading a checkpoint from a
tree that does not match it builds a different architecture and drops weights
without raising (`load_state_dict(strict=False)` only prints).  Here the load is
checked tensor by tensor and the script aborts on any mismatch.

Two modes, on the same frames:
  batch   `inference(...)`: all frames in one call.
  stream  `inference_recurrent(...)`: one frame at a time (online use).

    python tools/eval/efficiency.py --repo . --label loopcut3r \
        --model_path checkpoints/loopcut3r_512_dpt.pth \
        --img_dir examples/001 --frames 13 90 300 --out efficiency_loop.json
"""
import argparse, collections, glob, json, os, sys, time
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--model_path", required=True)
ap.add_argument("--model_cfg", default=None)
ap.add_argument("--img_dir", required=True)
ap.add_argument("--label", required=True)
ap.add_argument("--size", type=int, default=512)
ap.add_argument("--frames", type=int, nargs="+", default=[13])
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--timed", type=int, default=5)
ap.add_argument("--out", default=None)
args = ap.parse_args()

sys.path.insert(0, args.repo)
os.chdir(args.repo)
from add_ckpt_path import add_path_to_dust3r
add_path_to_dust3r(args.model_path)
import demo as D
from src.dust3r.model import ARCroco3DStereo
from src.dust3r.inference import inference, inference_recurrent

dev = "cuda"
kw = {"model_cfg_str": args.model_cfg} if args.model_cfg else {}
model = ARCroco3DStereo.from_pretrained(args.model_path, **kw).to(dev).eval()

# ---- verify the load, do not assume it ---------------------------------------
ck = torch.load(args.model_path, map_location="cpu", weights_only=False)
sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}  # DataParallel
msd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
missing = [k for k in msd if k not in sd]
unexpected = [k for k in sd if k not in msd]
different = [k for k in msd if k in sd and
             (msd[k].shape != sd[k].shape or not torch.equal(msd[k].float(), sd[k].float()))]
print(f"[load] missing={len(missing)} unexpected={len(unexpected)} different={len(different)}", flush=True)
if missing or unexpected or different:
    sys.exit(f"ABORT: {args.model_path} does not match the model built from {args.repo}")
del ck, sd, msd

# ---- parameters --------------------------------------------------------------
groups = collections.OrderedDict()
for n, p in model.named_parameters():
    g = n.split(".")[0]
    groups[g] = groups.get(g, 0) + p.numel()
total = sum(groups.values())
print(f"[params] total {total/1e6:.2f} M")
for g, v in sorted(groups.items(), key=lambda kv: -kv[1]):
    print(f"   {g:<28}{v/1e6:>10.2f} M")
nblk = {k: len(getattr(model, k)) for k in ("enc_blocks", "enc_blocks_ray_map", "dec_blocks", "dec_blocks_state")
        if hasattr(model, k)}
print(f"[distinct blocks] {nblk}")

# ---- frames -------------------------------------------------------------------
paths = sorted(p for p in glob.glob(os.path.join(args.img_dir, "*"))
               if os.path.splitext(p)[1].lower() in (".jpg", ".jpeg", ".png"))
assert paths, args.img_dir


def make_views(n):
    sel = [paths[i % len(paths)] for i in range(n)]
    return D.prepare_input(sel, [True] * n, args.size)


# ---- block applications per frame ---------------------------------------------
calls = collections.Counter()
hooks = []
for k in nblk:
    for blk in getattr(model, k):
        hooks.append(blk.register_forward_hook(lambda m, i, o, k=k: calls.update([k])))
v = make_views(4)
with torch.no_grad():
    inference_recurrent(v, model, dev, verbose=False)
for h in hooks:
    h.remove()
applications = {k: calls[k] / 4 for k in nblk}
print(f"[block applications per frame, 4 frames streamed] {applications}")
hw = tuple(int(x) for x in v[0]["true_shape"][0].tolist())
print(f"[resolution] HxW={hw}")

# ---- memory and speed --------------------------------------------------------
torch.cuda.empty_cache()
weights_gb = torch.cuda.memory_allocated() / 2**30
print(f"[memory] weights only on GPU: {weights_gb:.2f} GB")
res = []
for mode, fn in (("batch", inference), ("stream", inference_recurrent)):
    for n in args.frames:
        v = make_views(n)
        timed = args.timed if n <= 100 else 2
        try:
            with torch.no_grad():
                for _ in range(args.warmup if n <= 100 else 1):
                    out = fn(v, model, dev, verbose=False); del out
                torch.cuda.synchronize(); torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                ts = []
                for _ in range(timed):
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    out = fn(v, model, dev, verbose=False)
                    torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
                    del out
            peak = torch.cuda.max_memory_allocated() / 2**30
            fps = n / (sum(ts) / len(ts))
            print(f"[{mode:<6}] frames={n:<5} peak={peak:6.2f} GB  (above weights +{peak-weights_gb:5.2f})  "
                  f"fps={fps:7.2f}  ms/frame={1000/fps:6.1f}", flush=True)
            res.append(dict(mode=mode, frames=n, peak_gb=round(peak, 3), fps=round(fps, 2)))
        except torch.cuda.OutOfMemoryError:
            print(f"[{mode:<6}] frames={n:<5} OOM", flush=True)
            res.append(dict(mode=mode, frames=n, peak_gb=None, fps=None))
        del v
        torch.cuda.empty_cache()

if args.out:
    json.dump(dict(label=args.label, repo=args.repo, model_path=args.model_path, hw=hw,
                   params_total=total, params=groups, distinct_blocks=nblk, block_applications_per_frame=applications,
                   weights_gb=round(weights_gb, 3), runs=res), open(args.out, "w"), indent=1)
