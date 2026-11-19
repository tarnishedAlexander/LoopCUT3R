"""Run a repo's eval/relpose/launch.py under a modern evo + matplotlib.

Two API drifts, neither of which touches a metric value:

  evo        main_ape.ape() / main_rpe.rpe() no longer take est_name/ref_name.
             Those were labels for plot legends and result names; the code only
             reads result.stats["rmse"].  Unsupported keyword arguments are
             dropped instead of pinning an old evo, so both models are scored
             by the same current implementation.
  matplotlib cm.get_cmap was removed in 3.9.

Runs the script in-process, so the patches apply.  num_processes was 1 anyway,
so skipping `accelerate launch` changes nothing: launch.py builds its own
Accelerator.
"""
import inspect, runpy, sys
import matplotlib, matplotlib.cm as cm
if not hasattr(cm, "get_cmap"):
    cm.get_cmap = lambda name=None, lut=None: matplotlib.colormaps[name]

import evo.main_ape as main_ape, evo.main_rpe as main_rpe


def drop_unsupported(mod, fname):
    fn = getattr(mod, fname)
    allowed = set(inspect.signature(fn).parameters)

    def wrapper(*a, **kw):
        dropped = [k for k in kw if k not in allowed]
        for k in dropped:
            kw.pop(k)
        if dropped:
            print(f"[shim] {fname}(): dropped {dropped}", file=sys.stderr)
        return fn(*a, **kw)

    setattr(mod, fname, wrapper)


drop_unsupported(main_ape, "ape")
drop_unsupported(main_rpe, "rpe")

target = sys.argv.pop(1)
sys.argv[0] = target
runpy.run_path(target, run_name="__main__")
