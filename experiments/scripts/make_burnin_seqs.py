#!/usr/bin/env python3
"""Build burn-in probe sequences: the first frame duplicated at the front, so
the recurrent state is warmed to its own fixed point before the frame we score.
The duplicate's prediction is dropped afterwards by bi_slice.py."""
import pathlib, shutil, sys

src = pathlib.Path(sys.argv[1])            # examples/002
out = pathlib.Path(sys.argv[2])            # $AB/bi_seqs
frames = sorted(p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True)
name = src.name
made = []

# burn-in pairs: [f_i, f_i, f_{i+1}]  -> scored pair is (f_i, f_{i+1}) with warm state
for i in range(len(frames) - 1):
    d = out / f"pair_{name}_{i+1:02d}_{i+2:02d}"
    d.mkdir()
    for k, f in enumerate([frames[i], frames[i], frames[i + 1]]):
        shutil.copy2(f, d / f"{k:02d}{f.suffix}")
    made.append(d.name)

# burn-in full sequence: [f_1, f_1, f_2, ... f_N]
d = out / name
d.mkdir()
seq = [frames[0]] + frames
for k, f in enumerate(seq):
    shutil.copy2(f, d / f"{k:02d}{f.suffix}")
made.append(f"{d.name} ({len(seq)} inputs -> {len(frames)} scored)")

print(f"{len(frames)} source frames; wrote {len(made)} burn-in sequences to {out}")
for m in made:
    print("   ", m)
