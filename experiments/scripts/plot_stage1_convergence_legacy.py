"""
LoopCUT3R Stage 1 — Thesis-quality convergence plots.

Parses train.log (text format) and produces 6 individual figures:
  1. cross_view_steps.png        — pts3d/1-4 over epochs (recurrence progression)
  2. cross_view_gap.png          — pts3d/4 - pts3d/1 (convergence criterion)
  3. pose_loss.png               — camera pose loss over epochs
  4. self_reconstruction.png     — self_pts3d/1-4 per recurrent step
  5. learning_rate.png           — LR schedule over training epochs
  6. rgb_loss.png                — RGB appearance loss per recurrent step

Run:
  python plot_stage1_convergence.py
"""

import re
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

LOG_PATH = "checkpoints/recurrent_stage1/train.log"
OUT_DIR  = "checkpoints/recurrent_stage1/plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Matplotlib global style ────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['DejaVu Sans'],
    'font.size':         13,
    'axes.labelsize':    14,
    'axes.titlesize':    15,
    'axes.titleweight':  'bold',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.linestyle':    ':',
    'grid.alpha':        0.45,
    'grid.color':        '#aaaaaa',
    'xtick.labelsize':   12,
    'ytick.labelsize':   12,
    'legend.fontsize':   11,
    'legend.frameon':    True,
    'legend.framealpha': 0.95,
    'legend.edgecolor':  '#cccccc',
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
})

# ── Colour palettes ────────────────────────────────────────────────────────────
STEP_COLS  = ['#93c4de', '#4a9ec4', '#1f6fa0', '#0d3e63']  # blues steps 1-4
SELF_COLS  = ['#a8d5a2', '#5db85a', '#2e8b2e', '#0f5c0f']  # greens steps 1-4
RGB_COLS   = ['#d4a8e0', '#a855c8', '#7c3aed', '#4c1d95']  # purples steps 1-4
POSE_COL   = '#c0392b'
GAP_COL    = '#e67e22'
LR_COL     = '#7f8c8d'
TARGET_COL = '#2ecc71'

# ── Annotation events ─────────────────────────────────────────────────────────
EVENTS = {
    7:  ('SGDR\nrestart', '#888888'),
    10: ('TBPTL update\n(high-LR oscillation)', '#2980b9'),
}

# ── Log parsing ───────────────────────────────────────────────────────────────

def parse_metrics(line):
    m = {}
    for hit in re.finditer(r'([\w/.()_]+):\s+[\d.]+\s+\(([\d.]+)\)', line):
        m[hit.group(1)] = float(hit.group(2))
    lr = re.search(r'\blr:\s+([\d.e+\-]+)', line)
    if lr:
        m['lr'] = float(lr.group(1))
    return m

test_rows  = {}
train_rows = {}

with open(LOG_PATH, 'r') as fh:
    for line in fh:
        if 'MemClean' in line:
            continue
        if 'Test Epoch:' in line and '[62/63]' in line:
            em = re.search(r'Test Epoch:\s+\[(\d+)\]', line)
            if em:
                test_rows[int(em.group(1))] = parse_metrics(line)
        elif 'Epoch:' in line and 'Test Epoch:' not in line:
            em = re.search(r'Epoch:\s+\[(\d+)\].*?\[(\d+)/\d+\]', line)
            if em:
                idx, step = int(em.group(1)), int(em.group(2))
                row = parse_metrics(line)
                row['_step'] = step
                if idx not in train_rows or step >= train_rows[idx]['_step']:
                    train_rows[idx] = row

# ── Array builders ────────────────────────────────────────────────────────────

def get_test(key):
    epochs, vals = [], []
    for ep in sorted(test_rows):
        v = test_rows[ep].get(key)
        if v is not None:
            epochs.append(ep); vals.append(v)
    return np.array(epochs), np.array(vals)

def get_train(key):
    epochs, vals = [], []
    for ep in sorted(train_rows):
        v = train_rows[ep].get(key)
        if v is not None:
            epochs.append(ep + 1); vals.append(v)
    return np.array(epochs), np.array(vals)

K_PTS  = [f'Regr3DPose_pts3d/{i}'      for i in range(1, 5)]
K_SELF = [f'Regr3DPose_self_pts3d/{i}' for i in range(1, 5)]
K_RGB  = [f'RGBLoss_rgb/{i}'           for i in range(1, 5)]
K_POSE = 'pose_loss'

# ── Shared helpers ────────────────────────────────────────────────────────────

def annotate_events(ax):
    ylim = ax.get_ylim()
    span = ylim[1] - ylim[0]
    for ep, (label, col) in EVENTS.items():
        ax.axvline(ep, color=col, linewidth=1.3, linestyle='--', alpha=0.7, zorder=1)
        ax.text(ep + 0.15, ylim[0] + span * 0.05, label,
                fontsize=9, color=col, va='bottom', ha='left', linespacing=1.3)

def set_xticks(ax, epochs):
    ax.set_xticks(epochs[::2] if len(epochs) > 10 else epochs)
    ax.set_xlabel('Epoch', labelpad=6)

def xlim(ax, epochs):
    ax.set_xlim(epochs[0] - 0.4, epochs[-1] + 0.4)

def top_legend(fig, ax, ncol=None):
    handles, labels = ax.get_legend_handles_labels()
    nc = ncol or len(handles)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.99),
               ncol=nc, frameon=True, framealpha=0.95, edgecolor='#cccccc',
               fontsize=10.5)
    plt.tight_layout(rect=[0, 0, 1, 0.89])

def save(name):
    plt.savefig(os.path.join(OUT_DIR, name))
    plt.close()
    print(f'Saved {name}')

# ═════════════════════════════════════════════════════════════════════════════
# 1 — Cross-view steps
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5.2))
step_data = [get_test(k) for k in K_PTS]
ep1, v1 = step_data[0];  ep4, v4 = step_data[3]
all_epochs = ep1

for i, ((ep, v), col, lbl) in enumerate(
        zip(step_data, STEP_COLS, ['Step 1','Step 2','Step 3','Step 4'])):
    ax.plot(ep, v, color=col, linewidth=2.2 if i in (0,3) else 1.7,
            marker='o', markersize=4, label=lbl, zorder=3)
ax.fill_between(ep1, v1, v4, alpha=0.12, color=GAP_COL, label='Gap (step 4 − step 1)')

ax.set_ylabel('Reconstruction Loss (L21)', labelpad=6)
ax.set_title('Cross-View Reconstruction Quality per Recurrent Step')
set_xticks(ax, all_epochs); xlim(ax, all_epochs)
annotate_events(ax)
top_legend(fig, ax, ncol=5)
save('1_cross_view_steps.png')

# ═════════════════════════════════════════════════════════════════════════════
# 2 — Cross-view gap
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5.2))
ep1, v1 = get_test(K_PTS[0]); ep4, v4 = get_test(K_PTS[3])
gap = v4 - v1

ax.plot(ep1, gap, color=GAP_COL, linewidth=2.5, marker='o', markersize=5,
        label='Gap (pts3d/4 − pts3d/1)', zorder=3)
ax.fill_between(ep1, gap, 0, where=(gap > 0), alpha=0.13, color=GAP_COL)
ax.axhline(0, color=TARGET_COL, linewidth=1.8, linestyle='--',
           label='Convergence target (gap = 0)', zorder=2)

ax.set_ylabel('Recurrence Gap (L21)', labelpad=6)
ax.set_title('Cross-View Recurrence Gap — Stage 1 Convergence Criterion')
set_xticks(ax, ep1); xlim(ax, ep1)
annotate_events(ax)
top_legend(fig, ax, ncol=2)
save('2_cross_view_gap.png')

# ═════════════════════════════════════════════════════════════════════════════
# 3 — Pose loss
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5.2))
ep, v = get_test(K_POSE)

ax.plot(ep, v, color=POSE_COL, linewidth=2.5, marker='o', markersize=5,
        label='Pose loss', zorder=3)
ax.axhline(0.38, color=TARGET_COL, linewidth=1.8, linestyle='--',
           label='Target threshold (0.38)', zorder=2)

ax.set_ylabel('Pose Loss (L21)', labelpad=6)
ax.set_title('Camera Pose Estimation Loss')
set_xticks(ax, ep); xlim(ax, ep)
annotate_events(ax)
top_legend(fig, ax, ncol=2)
save('3_pose_loss.png')

# ═════════════════════════════════════════════════════════════════════════════
# 4 — Self-reconstruction
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5.2))
self_data = [get_test(k) for k in K_SELF]

for i, ((ep, v), col, lbl) in enumerate(
        zip(self_data, SELF_COLS, ['Step 1','Step 2','Step 3','Step 4'])):
    ax.plot(ep, v, color=col, linewidth=2.2 if i in (0,3) else 1.7,
            marker='s', markersize=4, label=lbl, zorder=3)

ax.set_ylabel('Self-Reconstruction Loss (L21)', labelpad=6)
ax.set_title('Self-Reconstruction Quality per Recurrent Step\n'
             '(monotone step 1 > step 2 > step 3 > step 4 = recurrence learning)')
set_xticks(ax, self_data[0][0]); xlim(ax, self_data[0][0])
annotate_events(ax)
top_legend(fig, ax, ncol=4)
save('4_self_reconstruction.png')

# ═════════════════════════════════════════════════════════════════════════════
# 5 — Learning rate
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5.2))
ep, lr = get_train('lr')

ax.plot(ep, lr, color=LR_COL, linewidth=2.5, marker='o', markersize=4,
        label='Learning rate', zorder=3)
ax.set_yscale('log')
ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
ax.set_ylabel('Learning Rate (log scale)', labelpad=6)
ax.set_title('Learning Rate Schedule — Stage 1 Cosine Decay')
set_xticks(ax, ep); xlim(ax, ep)

ylim = ax.get_ylim()
for evt_ep, (label, col) in EVENTS.items():
    ax.axvline(evt_ep, color=col, linewidth=1.3, linestyle='--', alpha=0.7, zorder=1)
    ax.text(evt_ep + 0.15,
            10 ** (np.log10(ylim[0]) + 0.06*(np.log10(ylim[1])-np.log10(ylim[0]))),
            label, fontsize=9, color=col, va='bottom', ha='left', linespacing=1.3)

top_legend(fig, ax, ncol=1)
save('5_learning_rate.png')

# ═════════════════════════════════════════════════════════════════════════════
# 6 — RGB appearance loss per recurrent step
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5.2))
rgb_data = [get_test(k) for k in K_RGB]

for i, ((ep, v), col, lbl) in enumerate(
        zip(rgb_data, RGB_COLS, ['Step 1','Step 2','Step 3','Step 4'])):
    ax.plot(ep, v, color=col, linewidth=2.2 if i in (0,3) else 1.7,
            marker='D', markersize=4, label=lbl, zorder=3)

ax.set_ylabel('RGB Loss (L21)', labelpad=6)
ax.set_title('RGB Appearance Reconstruction Loss per Recurrent Step')
set_xticks(ax, rgb_data[0][0]); xlim(ax, rgb_data[0][0])
annotate_events(ax)
top_legend(fig, ax, ncol=4)
save('6_rgb_loss.png')

# ═════════════════════════════════════════════════════════════════════════════
# 7 — Training total loss
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5.2))
ep, v = get_train('loss')

ax.plot(ep, v, color='#2c3e50', linewidth=2.5, marker='o', markersize=5,
        label='Total training loss', zorder=3)
ax.fill_between(ep, v, v.min(), alpha=0.08, color='#2c3e50')

ax.set_ylabel('Total Loss', labelpad=6)
ax.set_title('Total Training Loss — Stage 1')
set_xticks(ax, ep); xlim(ax, ep)
annotate_events(ax)
top_legend(fig, ax, ncol=1)
save('7_total_training_loss.png')

# ═════════════════════════════════════════════════════════════════════════════
# 8 — Cross-view pts3d: final step only (step 4)
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5.2))
ep, v = get_test('Regr3DPose_pts3d/4')

ax.plot(ep, v, color=STEP_COLS[3], linewidth=2.5, marker='o', markersize=5,
        label='Cross-view pts3d (step 4)', zorder=3)
ax.fill_between(ep, v, v.min(), alpha=0.09, color=STEP_COLS[3])

ax.set_ylabel('Reconstruction Loss (L21)', labelpad=6)
ax.set_title('Cross-View Geometry — Final Recurrent Step\n'
             '(step 4 = decoder output after full recurrence)')
set_xticks(ax, ep); xlim(ax, ep)
annotate_events(ax)
top_legend(fig, ax, ncol=1)
save('8_cross_view_final_step.png')

# ═════════════════════════════════════════════════════════════════════════════
# 9 — Self-reconstruction: final step only (step 4)
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5.2))
ep, v = get_test('Regr3DPose_self_pts3d/4')

ax.plot(ep, v, color=SELF_COLS[3], linewidth=2.5, marker='s', markersize=5,
        label='Self pts3d (step 4)', zorder=3)
ax.fill_between(ep, v, v.min(), alpha=0.09, color=SELF_COLS[3])

ax.set_ylabel('Self-Reconstruction Loss (L21)', labelpad=6)
ax.set_title('Self-Reconstruction — Final Recurrent Step\n'
             '(same-frame consistency after full recurrence)')
set_xticks(ax, ep); xlim(ax, ep)
annotate_events(ax)
top_legend(fig, ax, ncol=1)
save('9_self_recon_final_step.png')

# ═════════════════════════════════════════════════════════════════════════════
# 10 — RGB appearance: final step only (step 4)
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5.2))
ep, v = get_test('RGBLoss_rgb/4')

ax.plot(ep, v, color=RGB_COLS[3], linewidth=2.5, marker='D', markersize=5,
        label='RGB loss (step 4)', zorder=3)
ax.fill_between(ep, v, v.min(), alpha=0.09, color=RGB_COLS[3])

ax.set_ylabel('RGB Loss (L21)', labelpad=6)
ax.set_title('RGB Appearance Reconstruction — Final Recurrent Step')
set_xticks(ax, ep); xlim(ax, ep)
annotate_events(ax)
top_legend(fig, ax, ncol=1)
save('10_rgb_final_step.png')

# ═════════════════════════════════════════════════════════════════════════════
# 11 — Normalized multi-loss convergence overview (all components, step 4)
#      Each curve normalized to its first epoch value = 1.0
#      Shows relative convergence rate across all Stage 1 loss components.
#      Note: confidence loss (conf_loss/4) = pts3d/4 numerically — not plotted separately.
#      Note: ray consistency loss is NOT present in Stage 1 (later stage only).
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5.5))

components = [
    ('Regr3DPose_pts3d/4',      STEP_COLS[3],  'Cross-view pts3d',  'o'),
    ('Regr3DPose_self_pts3d/4', SELF_COLS[3],  'Self pts3d',        's'),
    ('pose_loss',               POSE_COL,       'Pose',              '^'),
    ('RGBLoss_rgb/4',           RGB_COLS[3],    'RGB',               'D'),
]

for key, col, label, marker in components:
    ep, v = get_test(key)
    if len(v) == 0:
        continue
    v_norm = v / v[0]
    ax.plot(ep, v_norm, color=col, linewidth=2.3, marker=marker,
            markersize=4.5, label=label, zorder=3)

ax.axhline(1.0, color='#aaaaaa', linewidth=1.0, linestyle=':', zorder=1)
ax.set_ylabel('Loss (normalized to epoch 1 = 1.0)', labelpad=6)
ax.set_title('Stage 1 — All Loss Components, Normalized Convergence')
set_xticks(ax, ep); xlim(ax, ep)
annotate_events(ax)
top_legend(fig, ax, ncol=4)
save('11_all_losses_normalized.png')

print(f'\nAll plots saved to {OUT_DIR}/')
