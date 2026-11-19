#!/usr/bin/env python3
"""Check that a checkpoint loads into the model built from a given source tree.

    python tools/eval/check_weights.py <repo> <checkpoint>

`load_model()` loads with `strict=False` and only prints the report, so a
checkpoint built by a different tree (e.g. CUT3R's 24 independent encoder
blocks into LoopCUT3R's 3 shared ones) loads "successfully" with random
weights.  Want: MISSING 0 and UNEXPECTED 0.
"""
import sys, os, torch, re
repo = sys.argv[1]; ckpt = sys.argv[2]
sys.path.insert(0, repo); os.chdir(repo)
from add_ckpt_path import add_path_to_dust3r
add_path_to_dust3r(ckpt)
from src.dust3r.model import ARCroco3DStereo

ck = torch.load(ckpt, map_location="cpu", weights_only=False)
sd = ck["model"] if "model" in ck else ck
sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
args = ck.get("args", None)
print("embedded model string:", (str(getattr(args, "model", None))[:200]) if args else "NONE")

# build exactly the way from_pretrained does
m = ARCroco3DStereo.from_pretrained(ckpt)
msd = m.state_dict()
missing = [k for k in msd if k not in sd]
unexpected = [k for k in sd if k not in msd]
def enc_n(keys):
    n = set()
    for k in keys:
        g = re.match(r"enc_blocks\.(\d+)\.", k)
        if g: n.add(int(g.group(1)))
    return len(n)
def dec_n(keys, pre):
    n = set()
    for k in keys:
        g = re.match(re.escape(pre) + r"\.(\d+)\.", k)
        if g: n.add(int(g.group(1)))
    return len(n)
print(f"checkpoint: enc_blocks={enc_n(sd)}  dec_blocks={dec_n(sd,'dec_blocks')}  dec_blocks_state={dec_n(sd,'dec_blocks_state')}")
print(f"built model: enc_blocks={enc_n(msd)}  dec_blocks={dec_n(msd,'dec_blocks')}  dec_blocks_state={dec_n(msd,'dec_blocks_state')}")
print(f"MISSING in ckpt (model params left at init): {len(missing)}")
for k in missing[:12]: print("   -", k)
print(f"UNEXPECTED in ckpt (weights thrown away): {len(unexpected)}")
for k in unexpected[:12]: print("   +", k)
print("runtime: enc num_steps =", getattr(m, "num_steps", "?"), " enc num_blocks =", len(m.enc_blocks))
print("runtime: dec_num_steps_img =", getattr(m, "dec_num_steps_img", "?"),
      " dec_num_blocks_img =", getattr(m, "dec_num_blocks_img", "?"),
      " dec_depth =", getattr(m, "dec_depth", "?"))
