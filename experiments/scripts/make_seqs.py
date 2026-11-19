#!/usr/bin/env python3
"""
Build the probe sequences for experiments A and B out of an examples/ directory.

  pair_<seq>_<i>_<j>   frames i and j as an independent 2-frame sequence, so the
                       model sees them with a freshly initialised state.  Run
                       against the same pair inside the full sequence, this
                       separates "the state accumulates corruption" from "the
                       decoder cannot register two frames even when the state is
                       clean".
  rep_<seq>_<i>_xN     frame i repeated N times.  The input never changes, so
                       every predicted point must stay put; whatever moves is
                       the model disagreeing with itself, with no confound from
                       scene difficulty or camera motion.

Frames are copied, not symlinked: the probe directories usually live on another
filesystem (or get mounted into a container at a different path), and a symlink
to a host path dangles there.

    python tools/ab/make_seqs.py examples/002 /path/to/ab_seqs
"""
import argparse
import pathlib
import shutil

IMG_EXT = {".jpg", ".jpeg", ".png"}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("seq_dir", help="an examples/<seq> directory of frames")
    ap.add_argument("out_dir", help="where to write the probe sequences")
    ap.add_argument("--name", default=None, help="sequence label (default: dir name)")
    ap.add_argument("--pairs", action="store_true", default=True)
    ap.add_argument("--no-pairs", dest="pairs", action="store_false")
    ap.add_argument(
        "--repeat",
        nargs="*",
        type=int,
        default=[1],
        metavar="FRAME",
        help="1-based frame indices to repeat (default: 1); empty disables",
    )
    ap.add_argument("--repeat-n", type=int, default=8, help="repetitions per frame")
    ap.add_argument("--clean", action="store_true", help="wipe out_dir first")
    args = ap.parse_args()

    src = pathlib.Path(args.seq_dir)
    frames = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT)
    if not frames:
        raise SystemExit(f"no images in {src}")
    name = args.name or src.name
    out = pathlib.Path(args.out_dir)
    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    made = []
    if args.pairs:
        for i in range(len(frames) - 1):
            d = out / f"pair_{name}_{i+1:02d}_{i+2:02d}"
            d.mkdir(exist_ok=True)
            shutil.copy2(frames[i], d / f"f0{frames[i].suffix}")
            shutil.copy2(frames[i + 1], d / f"f1{frames[i+1].suffix}")
            made.append(d.name)

    for f in args.repeat or []:
        if not 1 <= f <= len(frames):
            raise SystemExit(f"--repeat {f} out of range (1..{len(frames)})")
        d = out / f"rep_{name}_{f:02d}_x{args.repeat_n}"
        d.mkdir(exist_ok=True)
        for k in range(args.repeat_n):
            shutil.copy2(frames[f - 1], d / f"r{k}{frames[f-1].suffix}")
        made.append(d.name)

    print(f"{len(frames)} source frames in {src}")
    print(f"wrote {len(made)} probe sequences to {out}")
    for n in made:
        print("   ", n)


if __name__ == "__main__":
    main()
