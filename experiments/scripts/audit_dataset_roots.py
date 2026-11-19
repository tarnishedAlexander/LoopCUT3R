"""Audit every dataset in a config: does the built object point at the ROOT the
config asked for, and does it hand back a loadable sample?

Run this before any long training run.  A mixture of 30 datasets fails in ways
that do not look like failures: a dataset can be built from the wrong tree, or
can raise on every single sample while the retry loop swallows it, and the only
symptom is a number at the end that nobody can attribute.

It exists because of two real incidents:

  * A stale scan cache made `WildRGBD_Multi` serve 2511 CO3D sequences, labelled
    WildRGBD with `is_metric=True`, out of `disk14tb/processed_co3d` -- the cache
    restored `self.ROOT` over the configured one.  The `ROOT MISMATCH` column is
    exactly that check, and `tools/ab/test_scan_cache_root.py` pins the fix.
  * `ThreeDKenBurns` raised "assignment destination is read-only" on every
    sample, because the native EXR reader returned a `np.frombuffer` view and
    the loader writes into it in place.  The per-dataset sample load is that
    check.

Usage:
    python tools/ab/audit_dataset_roots.py config/recurrent_stage3c_tscale.yaml

Exit code 0 means every dataset in the config points where the config says and
yields one loadable window.  `metric=` and the depth mean are printed so a unit
mix-up (metres against arbitrary scale) is visible at a glance.
"""
import re
import sys
import traceback

sys.path.insert(0, "src")

from omegaconf import OmegaConf

import dust3r.datasets  # noqa: F401  (registers the dataset classes for eval)
from dust3r.datasets import *  # noqa: F401,F403

cfg_path = sys.argv[1]
cfg = OmegaConf.load(cfg_path)
OmegaConf.resolve(cfg)

names = sorted(
    (k for k in cfg.keys() if re.fullmatch(r"dataset\d+", str(k))),
    key=lambda k: int(str(k)[7:]),
)

print(f"{len(names)} datasets in {cfg_path}\n")
bad = []
warn = []   # real defects in the data that are not migration/root problems
for name in names:
    spec = str(cfg[name]).strip()
    cls = spec.split("(", 1)[0]
    m = re.search(r"ROOT='([^']*)'", spec)
    want = m.group(1) if m else None
    try:
        ds = eval(spec)
    except Exception as e:
        print(f"  {name:<10} {cls:<26} BUILD FAILED  {type(e).__name__}: {e}")
        bad.append((name, cls, "build failed"))
        continue

    got = getattr(ds, "ROOT", None)
    label = getattr(ds, "dataset_label", None)
    metric = getattr(ds, "is_metric", None)
    n = len(getattr(ds, "scene_list", []) or [])

    root_ok = (want is None) or (str(got).rstrip("/") == want.rstrip("/"))
    status = "ok " if root_ok else "ROOT MISMATCH"

    try:
        views = ds[(0, 0, int(cfg.num_views))]
        d = views[0]["depthmap"]
        valid = d[d > 0]
        # An all-zero depthmap is not a load failure, so it reaches none of the
        # checks above: `d[d > 0].mean()` on an empty array is a RuntimeWarning
        # and a nan, which prints and passes.  MapFree sat in the mixture at
        # 11.8% of every epoch with 0% valid depth and this audit called it ok.
        # Report it loudly, but as a warning: the exit code is about whether the
        # configured tree is the one being read, which is a different question.
        if valid.size == 0:
            sample = f"{len(views)} views, NO DEPTH (0 of {d.size} px > 0)"
            warn.append((name, cls, "depthmap is entirely zero"))
        else:
            sample = (f"{len(views)} views, depth mean {float(valid.mean()):.3f}"
                      f", valid {100.0 * valid.size / d.size:.1f}%")
    except Exception as e:
        sample = f"SAMPLE FAILED {type(e).__name__}: {str(e)[:90]}"
        bad.append((name, cls, sample))

    print(f"  {name:<10} {cls:<26} {status:<13} n={n:<7} metric={str(metric):<5} "
          f"label={str(label):<14} {sample}")
    if not root_ok:
        print(f"             wanted {want}")
        print(f"             got    {got}")
        bad.append((name, cls, "root mismatch"))

print()
if warn:
    print(f"WARNINGS ({len(warn)}) -- these load fine but the data is defective:")
    for name, cls, why in warn:
        print(f"  {name} {cls}: {why}")
    print()
if bad:
    print(f"PROBLEMS ({len(bad)}):")
    for name, cls, why in bad:
        print(f"  {name} {cls}: {why}")
    sys.exit(1)
print("all datasets: ROOT matches the config and one sample loads")
