#!/usr/bin/env python3
"""Check that this machine can run LoopCUT3R at full speed.

    python docker/selftest.py [--model_path checkpoints/loopcut3r_512_dpt.pth]

1. The GPU is visible and PyTorch ships kernels for its architecture.
2. The compiled RoPE kernel loads, runs on this GPU and matches the PyTorch
   implementation (CroCo silently falls back to the slow one otherwise).
3. With --model_path, the checkpoint loads with every tensor matched and runs
   on two frames of examples/001.
Exits non-zero on the first failure.
"""
import argparse, importlib.util, os, sys, types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src"), os.path.join(ROOT, "src", "croco")]

import torch


def fail(msg):
    sys.exit(f"FAIL: {msg}")


ap = argparse.ArgumentParser()
ap.add_argument("--model_path", default=None)
args = ap.parse_args()

# 1. GPU and architecture -------------------------------------------------------
if not torch.cuda.is_available():
    fail("no CUDA GPU visible (driver or nvidia-container-toolkit missing?)")
major, minor = torch.cuda.get_device_capability()
archs = torch.cuda.get_arch_list()
mem = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"GPU       {torch.cuda.get_device_name()}, sm_{major}{minor}, {mem:.1f} GB")
print(f"PyTorch   {torch.__version__}, CUDA {torch.version.cuda}, kernels: {' '.join(archs)}")
# A cubin for sm_XY runs on sm_XZ with Z >= Y.
if not any(a.startswith(f"sm_{major}") and int(a[len(f'sm_{major}'):] or 0) <= minor for a in archs):
    fail(f"this PyTorch build has no kernels for sm_{major}{minor}")
torch.zeros(1, device="cuda").add_(1)

# 2. RoPE kernel ------------------------------------------------------------------
from models import pos_embed
if "curope" not in pos_embed.RoPE2D.__module__:
    fail("compiled RoPE kernel not found: CroCo would use its slow PyTorch fallback")
import curope

# Reference: the same pos_embed.py loaded with the kernel hidden, so that it
# defines its PyTorch RoPE2D.
saved = sys.modules.get("models.curope")
sys.modules["models.curope"] = types.ModuleType("models.curope")
spec = importlib.util.spec_from_file_location("_pos_embed_ref", pos_embed.__file__)
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
sys.modules["models.curope"] = saved

# Same memory layout as in the attention blocks: (B, N, heads, D) viewed as (B, heads, N, D).
tok = torch.randn(2, 196, 16, 64, device="cuda").transpose(1, 2)
grid = torch.stack(torch.meshgrid(torch.arange(14), torch.arange(14), indexing="ij"), -1)
pos = grid.reshape(1, -1, 2).expand(2, -1, -1).contiguous().cuda()
fast = pos_embed.RoPE2D(freq=100.0)(tok.clone(), pos)
slow = ref.RoPE2D(freq=100.0)(tok.clone(), pos)
diff = (fast - slow).abs().max().item()
print(f"RoPE      compiled kernel ({os.path.basename(curope.__file__)}), max diff vs PyTorch {diff:.1e}")
if diff > 1e-3:
    fail("compiled RoPE kernel disagrees with the PyTorch implementation")

# 3. Checkpoint -------------------------------------------------------------------
if args.model_path:
    import demo
    from src.dust3r.model import ARCroco3DStereo
    from src.dust3r.inference import inference_recurrent
    model = ARCroco3DStereo.from_pretrained(args.model_path).to("cuda").eval()
    ck = torch.load(args.model_path, map_location="cpu", weights_only=False)
    sd = {k.removeprefix("module."): v for k, v in ck.get("model", ck).items()}
    msd = model.state_dict()
    bad = [k for k in msd if k not in sd or msd[k].shape != sd[k].shape] + [k for k in sd if k not in msd]
    if bad:
        fail(f"{len(bad)} tensors differ between checkpoint and model, e.g. {bad[:3]}")
    print(f"Model     {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters, all tensors matched")
    frames = sorted(os.path.join(ROOT, "examples", "001", f) for f in os.listdir(os.path.join(ROOT, "examples", "001")))[:2]
    views = demo.prepare_input(frames, [True] * 2, 512)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        inference_recurrent(views, model, "cuda", verbose=False)
    print(f"Inference 2 frames at 512, peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")

print("OK")
