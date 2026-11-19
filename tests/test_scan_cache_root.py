#!/usr/bin/env python
"""A scan cache must never move a dataset to a different tree.

`BaseMultiViewDataset.__init_subclass__` wraps every subclass `__init__` with a
pickle cache of the directory scan.  Two properties of that wrapper used to make
it possible for one dataset to be served out of another dataset's tree, silently:

  1. The cache file name carries only the class and the split, and `cache_dir`
     falls back to the PARENT of the root whenever the root is not a real
     directory -- which is every H5VFS-backed dataset, because the VFS hooks are
     installed inside `BaseMultiViewDataset.__init__`, i.e. after the cache path
     has already been computed.  So several datasets on one disk share a cache.
  2. `ROOT` was restored from the pickle, so a cache written from another tree
     also redirected every subsequent path lookup into that tree.

Together those served 2511 CO3D sequences as `WildRGBD`, with `is_metric=True`,
out of `disk14tb/processed_co3d` while the config asked for
`disk14tb/preprocess_wildrgbd`.  The run only failed at all because that CO3D
directory had since been packed into an .h5; had it still existed, non-metric
CO3D depth would have trained as metric ground truth with no error anywhere.

Needs no data and no GPU: it builds throwaway subclasses over empty temp dirs.

    python tests/test_scan_cache_root.py
"""
import os
import os.path as osp
import shutil
import sys
import tempfile

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), "..", "src"))

from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset

FAILURES = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


# A dataset whose "scan" is just a note of the root it scanned, so a restored
# cache is immediately distinguishable from a fresh scan.
class Probe(BaseMultiViewDataset):
    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.scanned_from = ROOT
        self.scenes = {"only": [0, 1, 2]}

    def _get_views(self, idx, resolution, rng, num_views):
        raise NotImplementedError


class SubProbe(Probe):
    """A subclass with its own __init__, i.e. its own wrapper, like
    WildRGBD_Multi over Co3d_Multi."""

    def __init__(self, *args, ROOT, **kwargs):
        super().__init__(*args, ROOT=ROOT, **kwargs)
        self.sub = True


def build(cls, root, **kw):
    return cls(ROOT=root, split="train", num_views=4, resolution=224, **kw)


tmp = tempfile.mkdtemp(prefix="scan_cache_test_")
try:
    # Roots that do not exist as directories, so cache_dir falls back to `tmp`
    # for both and they collide on one file -- the H5VFS-backed case.
    a = osp.join(tmp, "processed_alpha")
    b = osp.join(tmp, "preprocess_beta")
    cache = osp.join(tmp, ".scan_Probe_train.pkl")

    print("[1] a cache from another tree is rejected, not silently applied")
    first = build(Probe, a)
    check("first build scans and writes a cache", osp.exists(cache),
          osp.basename(cache))
    check("first build kept its own root", first.ROOT == a)

    second = build(Probe, b)
    check("ROOT is the one the caller asked for, not the cached one",
          second.ROOT == b, f"got {second.ROOT}")
    check("the mismatched cache was rejected and a fresh scan ran",
          getattr(second, "scanned_from", None) == b,
          f"scanned_from={getattr(second, 'scanned_from', None)}")

    print("[2] a cache from the same dataset on a different disk is still used")
    # Same leaf name, different parent: the same dataset copied to a faster
    # disk.  Rescanning those is only slow, never wrong, so the cache stands.
    other_disk = osp.join(tmp, "faster")
    os.makedirs(other_disk, exist_ok=True)
    moved = osp.join(other_disk, "preprocess_beta")
    shutil.copyfile(cache, osp.join(other_disk, ".scan_Probe_train.pkl"))
    third = build(Probe, moved)
    check("restored from the relocated cache",
          getattr(third, "scanned_from", None) == b,
          f"scanned_from={getattr(third, 'scanned_from', None)}")
    check("but ROOT still points at the disk the config named",
          third.ROOT == moved, f"got {third.ROOT}")

    print("[3] super().__init__() does not write a cache under the parent's name")
    # Every subclass gets its own wrapper, so a subclass that calls
    # super().__init__() re-enters the parent's wrapper, which used to write a
    # second cache named after the PARENT -- overwriting the parent's real cache
    # with the subclass's scenes and forcing a full rescan later.
    for stale in (cache, osp.join(tmp, ".scan_SubProbe_train.pkl")):
        if osp.exists(stale):
            os.remove(stale)
    sub = build(SubProbe, b)
    check("the subclass wrote its own cache",
          osp.exists(osp.join(tmp, ".scan_SubProbe_train.pkl")))
    check("the parent class's cache was NOT written", not osp.exists(cache),
          "no .scan_Probe_train.pkl")
    check("the subclass still initialised fully", getattr(sub, "sub", False))
    check("the re-entrancy flag does not leak onto the instance",
          "_scan_cache_active" not in sub.__dict__)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print()
if FAILURES:
    print(f"{len(FAILURES)} check(s) failed:")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
print("all checks passed")
