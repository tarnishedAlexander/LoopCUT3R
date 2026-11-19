"""Run CUT3R's eval/video_depth/launch.py from inside the CUT3R tree.

Only reason this wrapper exists: CUT3R's eval/video_depth/utils.py calls
matplotlib.cm.get_cmap, removed in matplotlib 3.9.  LoopCUT3R's copy of that
file has the one-line fix; every other file in eval/video_depth is byte
identical between the two repos, so shimming the call keeps the metric
identical instead of forking the file.
"""
import runpy, sys
import matplotlib, matplotlib.cm as cm, matplotlib.pyplot as plt
if not hasattr(cm, "get_cmap"):
    cm.get_cmap = lambda name=None, lut=None: matplotlib.colormaps[name]
target = sys.argv.pop(1)
sys.argv[0] = target
runpy.run_path(target, run_name="__main__")
