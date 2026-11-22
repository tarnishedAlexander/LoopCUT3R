#!/usr/bin/env python3
"""Load a checkpoint strictly and run it on two frames of examples/001.

    python experiments/scripts/check_init.py checkpoints/init_hybrid_full.pth [...]

Prints parameter count, whether every tensor matched, peak VRAM and the median
confidence.  A cold-started mix is expected to have low confidence: this only
checks that the checkpoint is well formed before spending GPU-days on it.
"""
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]
import torch
import demo
from src.dust3r.model import load_model
from src.dust3r.inference import inference_recurrent

frames = sorted(os.path.join(ROOT, "examples", "001", f) for f in os.listdir(os.path.join(ROOT, "examples", "001")))[:2]
for path in sys.argv[1:]:
    model = load_model(path, "cpu", verbose=False)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = {k.removeprefix("module."): v for k, v in ck.get("model", ck).items()}
    msd = model.state_dict()
    bad = [k for k in msd if k not in sd or msd[k].shape != sd[k].shape] + [k for k in sd if k not in msd]
    model.load_state_dict(sd, strict=True)
    model = model.to("cuda").eval()
    views = demo.prepare_input(frames, [True] * 2, 512)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out = inference_recurrent(views, model, "cuda", verbose=False)
    preds = out[0] if isinstance(out, tuple) else out
    preds = preds["pred"] if isinstance(preds, dict) and "pred" in preds else preds
    confs = [p["conf"].float().median().item() for p in preds if isinstance(p, dict) and "conf" in p]
    print(f"{os.path.basename(path)}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M params, "
          f"mismatched tensors {len(bad)}, peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB, "
          f"median conf {[round(c, 3) for c in confs]}")
    del model; torch.cuda.empty_cache()
