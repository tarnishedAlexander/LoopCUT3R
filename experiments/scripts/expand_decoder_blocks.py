#!/usr/bin/env python3
"""
Expand a LoopCUT3R checkpoint's weight-shared decoder into N distinct blocks.

The decoder applies ``dec_blocks[step_idx % len(dec_blocks)]``, so 3 blocks over
12 steps run in the cycle 0,1,2,0,1,2,... Replicating block ``i % 3`` into slot
``i`` of an N-block model therefore reproduces the source model's function
*exactly* at initialisation, while giving every step its own parameters to
specialise. That makes the capacity of the decoder the only variable, with no
interface mismatch against the heads and no weights borrowed from CUT3R.

    python tools/expand_decoder_blocks.py IN.pth OUT.pth --blocks 12

With --from-cut3r the slots are filled from CUT3R's released decoder instead,
which is shape-identical (24 tensors per block, same names). That variant is a
different experiment: it tests CUT3R's trained decoder rather than capacity
alone, and it starts from a function that does not match the source heads.
"""
import argparse
import re
from collections import OrderedDict

import torch

GROUPS = ("dec_blocks", "dec_blocks_state")


def load_state(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    return OrderedDict((k.replace("module.", ""), v) for k, v in sd.items())


def block_count(sd, group):
    idx = {
        int(m.group(1))
        for k in sd
        if (m := re.match(rf"{group}\.(\d+)\.", k))
    }
    return len(idx)


def expand(sd, group, n_out, src=None):
    """Write n_out slots of `group`, cycling over the source's blocks."""
    donor = src if src is not None else sd
    n_in = block_count(donor, group)
    if n_in == 0:
        raise SystemExit(f"no {group}.* weights in the donor checkpoint")
    tail = {}  # block-relative key -> tensor, per source block
    for k, v in donor.items():
        m = re.match(rf"{group}\.(\d+)\.(.+)", k)
        if m:
            tail.setdefault(int(m.group(1)), {})[m.group(2)] = v
    out = OrderedDict((k, v) for k, v in sd.items() if not re.match(rf"{group}\.\d+\.", k))
    for i in range(n_out):
        for rel, v in tail[i % n_in].items():
            out[f"{group}.{i}.{rel}"] = v.clone()
    return out, n_in


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--blocks", type=int, required=True, help="distinct blocks per path")
    ap.add_argument("--from-cut3r", metavar="CUT3R_CKPT", default=None,
                    help="fill slots from CUT3R's decoder instead of cycling the source")
    args = ap.parse_args()

    sd = load_state(args.inp)
    src = load_state(args.from_cut3r) if args.from_cut3r else None

    for g in GROUPS:
        before = block_count(sd, g)
        sd, n_in = expand(sd, g, args.blocks, src)
        after = block_count(sd, g)
        origin = "CUT3R" if src is not None else "source (cyclic)"
        print(f"  {g}: {before} -> {after} blocks, filled from {origin} ({n_in} donors)")

    n = sum(v.numel() for v in sd.values())
    torch.save({"model": sd}, args.out)
    print(f"wrote {args.out}  ({n/1e6:.1f} M params)")


if __name__ == "__main__":
    main()
