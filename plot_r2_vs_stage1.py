"""Compare r2 epoch 0 (in-progress) vs Stage 1 epochs 1-16 training metrics."""
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

STAGE1_LOG = "checkpoints/recurrent_stage1/train.log"
R2_LOG     = "checkpoints/recurrent_stage1_r2/train.log"
OUT        = "plots/r2_vs_stage1_comparison.png"

# ── helpers ──────────────────────────────────────────────────────────────────

def parse_running(line, key):
    m = re.search(rf"{re.escape(key)}: \S+ \(([0-9.]+)\)", line)
    return float(m.group(1)) if m else None

def parse_field(line, key):
    m = re.search(rf"{re.escape(key)}: ([0-9.]+)", line)
    return float(m.group(1)) if m else None

# ── Stage 1: last print line per epoch ───────────────────────────────────────

s1_epochs, s1_loss, s1_pts4, s1_self4, s1_pose = [], [], [], [], []

with open(STAGE1_LOG) as f:
    lines = f.readlines()

epoch_last = {}
for line in lines:
    m = re.search(r"Epoch: \[(\d+)\]", line)
    if m and "Total time" not in line and "MemStart" not in line and "MemClean" not in line:
        epoch_last[int(m.group(1))] = line

for ep in sorted(epoch_last):
    line = epoch_last[ep]
    loss = parse_running(line, "loss")
    pts  = parse_running(line, "Regr3DPose_pts3d/4")
    slf  = parse_running(line, "Regr3DPose_self_pts3d/4")
    pose = parse_running(line, "pose_loss")
    if all(v is not None for v in [loss, pts, slf, pose]):
        s1_epochs.append(ep + 1)   # 1-indexed label
        s1_loss.append(loss)
        s1_pts4.append(pts)
        s1_self4.append(slf)
        s1_pose.append(pose)

# ── r2: step-level running averages from LAST run only ───────────────────────

r2_steps, r2_loss, r2_pts4, r2_self4, r2_pose, r2_lr = [], [], [], [], [], []

with open(R2_LOG) as f:
    lines = f.readlines()

# Find last "Start training" line
last_start = 0
for i, l in enumerate(lines):
    if "Start training" in l:
        last_start = i

total_steps = 95725
for line in lines[last_start:]:
    m = re.search(r"Epoch: \[0\]  \[ *(\d+)/\d+\]", line)
    if not m:
        continue
    if "Total time" in line or "MemStart" in line or "MemClean" in line:
        continue
    step = int(m.group(1))
    loss = parse_running(line, "loss")
    pts  = parse_running(line, "Regr3DPose_pts3d/4")
    slf  = parse_running(line, "Regr3DPose_self_pts3d/4")
    pose = parse_running(line, "pose_loss")
    lr   = parse_field(line, "lr")
    if all(v is not None for v in [loss, pts, slf, pose]):
        r2_steps.append(step / total_steps)   # fractional epoch progress
        r2_loss.append(loss)
        r2_pts4.append(pts)
        r2_self4.append(slf)
        r2_pose.append(pose)
        r2_lr.append(lr)

r2_steps = np.array(r2_steps)
r2_loss  = np.array(r2_loss)
r2_pts4  = np.array(r2_pts4)
r2_self4 = np.array(r2_self4)
r2_pose  = np.array(r2_pose)
r2_lr    = np.array(r2_lr)

# ── Plot ─────────────────────────────────────────────────────────────────────

STAGE1_C = "#5B8DB8"   # steel blue for stage 1 bars
R2_C     = "#E07040"   # warm orange for r2 line
WARMUP_C = "#DDDDDD"   # light grey for warmup region

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor("#F7F7F7")

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32,
                       left=0.07, right=0.97, top=0.90, bottom=0.08)

metrics = [
    ("Total Loss",        s1_loss,  r2_loss,  "loss"),
    ("pts3d/4 (cross-view)", s1_pts4, r2_pts4, "pts3d/4"),
    ("self_pts3d/4",      s1_self4, r2_self4, "self_pts3d/4"),
    ("Pose Loss",         s1_pose,  r2_pose,  "pose_loss"),
]

# warmup ends at 0.5 of r2 epoch
WARMUP_END = 0.5

for idx, (title, s1_vals, r2_vals, _) in enumerate(metrics):
    ax = fig.add_subplot(gs[idx // 2, idx % 2])
    ax.set_facecolor("white")

    # Warmup shading
    ax.axvspan(0, WARMUP_END, color=WARMUP_C, alpha=0.45, zorder=0)
    ax.axvline(WARMUP_END, color="#AAAAAA", lw=0.8, ls="--", zorder=1)

    # Stage 1 epoch bars (x = epoch number, width = 1)
    bar_x = np.array(s1_epochs) - 0.5          # centre at integer epoch
    bars = ax.bar(bar_x, s1_vals, width=0.85,
                  color=STAGE1_C, alpha=0.55, label="Stage 1 (epoch avg)",
                  zorder=2, edgecolor="none")

    # r2 running average line (x-axis scaled to epoch units: 0→1)
    # but we want it on the same x as stage 1 (1→16), so offset by +0.5 since
    # epoch 1 of stage1 spans x=[0.5,1.5]. r2 epoch 0 goes from 0→1 on the LEFT.
    # Cleanest: put stage1 on x=1..16, r2 on x=0..1 (separate region).
    # Actually let's use a twin x-axis and place r2 at x<1.

    ax2 = ax.twinx()
    ax2.set_yticks([])
    ax2.set_yticklabels([])

    # Put r2 in x-range [0, 1) and stage1 in [1, 17]
    ax.bar(bar_x + 1, s1_vals, width=0.0, alpha=0)   # invisible, just for range

    # r2 line on primary axis
    # Rescale r2 progress to x=[0,1)
    r2_x = r2_steps   # 0..~0.1 so far
    ax.plot(r2_x, r2_vals, color=R2_C, lw=2.0, zorder=5, label="r2 (running avg)")
    ax.fill_between(r2_x, r2_vals, alpha=0.12, color=R2_C, zorder=4)

    # Current value dot
    if len(r2_vals):
        ax.scatter([r2_x[-1]], [r2_vals[-1]], color=R2_C, s=50, zorder=6)

    # Axis labels
    ax.set_xlabel("← r2 epoch 0 progress →          ← Stage 1 epoch →", fontsize=8, color="#555")
    ax.set_ylabel(title, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)

    # x ticks: 0.0=start, 0.5=warmup-end, then 1-16 for stage1
    stage1_ticks  = list(range(1, len(s1_epochs) + 1))
    stage1_labels = [str(e) for e in s1_epochs]
    r2_ticks  = [0.0, 0.5]
    r2_labels = ["0%", "50%\n(peak LR)"]
    ax.set_xticks(r2_ticks + [t - 0.5 for t in stage1_ticks],
                  r2_labels + stage1_labels, fontsize=7)

    ax.set_xlim(-0.05, len(s1_epochs) + 0.6)
    ax.grid(axis="y", color="#E0E0E0", lw=0.6, zorder=0)
    ax.spines[["top","right"]].set_visible(False)
    ax2.spines[["top","right"]].set_visible(False)

    # Divider between r2 and stage1 regions
    ax.axvline(1.0, color="#888", lw=1.0, ls=":", zorder=3)

    if idx == 0:
        ax.legend(fontsize=8, loc="upper right",
                  handles=[
                      plt.Rectangle((0,0),1,1, color=STAGE1_C, alpha=0.55),
                      plt.Line2D([0],[0], color=R2_C, lw=2),
                  ],
                  labels=["Stage 1 epochs 1-16", f"r2 epoch 0 ({r2_steps[-1]*100:.1f}% done)"],
                  framealpha=0.9)

# ── Title and annotations ─────────────────────────────────────────────────────

current_pct = r2_steps[-1] * 100 if len(r2_steps) else 0
fig.suptitle(
    f"r2 Epoch 0 vs Stage 1  —  r2 currently {current_pct:.1f}% complete, LR still in warmup",
    fontsize=13, fontweight="bold", y=0.96
)

# Warmup annotation on first subplot
ax0 = fig.axes[0]
ax0.text(0.25, 0.97, "warmup\n(lr≈0)", transform=ax0.transAxes,
         fontsize=7, color="#999", ha="center", va="top")
ax0.text(0.52, 0.97, "cosine\ndescent", transform=ax0.transAxes,
         fontsize=7, color="#999", ha="left", va="top")

import os
os.makedirs("plots", exist_ok=True)
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {OUT}")
