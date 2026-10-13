"""
Stage 1 full convergence plot across r1, r2, r3 runs.

Timeline:
  r1 (ep 1-16): RAPTOR → TBPTL bug. ScaleInv 4.7→154 in ONE epoch.
  r2 (ep 17-18): resumed from r1-end, low LR — flat at ~163, TBPTL still broken.
  r3 (ep 17-22): resumed from r1-end, dec_use_tbptl=False — phase transition at ep20.
"""

import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

LOG_BASE = Path(__file__).parent.parent / "checkpoints"
R1_LOG = LOG_BASE / "recurrent_stage1" / "train.log"
R2_LOG = LOG_BASE / "recurrent_stage1_r2" / "train.log"
R3_LOG = LOG_BASE / "recurrent_stage1_r3" / "train.log"


def _parse_log(log_path: Path, final_step: str) -> dict[int, dict]:
    """Parse one log: for each Test Epoch, extract the last-step running averages.

    Anchor on 'eta:.*  loss:' to unambiguously grab the total loss (not pose_loss).
    For duplicate epoch numbers, last occurrence wins (handles auto-resume re-evals).
    """
    si_re = re.compile(
        r"Regr3DPose_ScaleInv_pts3d/1: [\d.]+ \(([\d.]+)\).*?"
        r"Regr3DPose_ScaleInv_pts3d/2: [\d.]+ \(([\d.]+)\).*?"
        r"Regr3DPose_ScaleInv_pts3d/3: [\d.]+ \(([\d.]+)\).*?"
        r"Regr3DPose_ScaleInv_pts3d/4: [\d.]+ \(([\d.]+)\)"
    )
    results: dict[int, dict] = {}
    tag = re.escape(final_step)

    for line in log_path.read_text().splitlines():
        # Match the specific final step for this log
        hdr = re.search(rf"Test Epoch: \[(\d+)\]\s+\[{tag}\]", line)
        if not hdr:
            continue
        epoch = int(hdr.group(1))

        # Total loss: anchored after 'eta: ...'
        loss_m = re.search(r"eta:\s[\d:]+\s+loss:\s[\d.]+\s\(([\d.]+)\)", line)
        if not loss_m:
            continue
        total_loss = float(loss_m.group(1))

        # ScaleInv (present only when test_criterion includes it, i.e. r2/r3 and r1 ep2+)
        si_m = si_re.search(line)
        scale_inv = sum(float(si_m.group(i)) for i in range(1, 5)) if si_m else float("nan")

        results[epoch] = {"loss": total_loss, "scale_inv": scale_inv}

    return results


def _arrays(d: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ep = np.array(sorted(d.keys()), dtype=float)
    loss = np.array([d[int(e)]["loss"] for e in ep])
    si = np.array([d[int(e)]["scale_inv"] for e in ep])
    return ep, loss, si


def parse_all():
    r1 = _parse_log(R1_LOG, "62/63")
    r2 = _parse_log(R2_LOG, "124/125")
    r3 = _parse_log(R3_LOG, "124/125")
    return r1, r2, r3


def _project_r3(r3_ep, r3_loss, future_eps=15):
    """Fit exponential decay on post-transition r3 data and project to future_eps."""
    mask = r3_ep >= 4
    x = r3_ep[mask].astype(float)
    y = r3_loss[mask]
    # fit log(y - floor) ~ a + b*x  →  y = floor + A*exp(b*x)
    floor = max(y[-1] * 0.5, 1.0)
    log_y = np.log(np.maximum(y - floor, 1e-3))
    coeffs = np.polyfit(x, log_y, 1)
    proj_x = np.arange(x[-1] + 1, future_eps + 1)
    proj_y = np.exp(np.polyval(coeffs, proj_x)) + floor
    return proj_x, proj_y


def plot():
    r1, r2, r3 = parse_all()
    r1_ep, r1_loss, r1_si = _arrays(r1)
    r2_ep, r2_loss, r2_si = _arrays(r2)
    r3_ep, r3_loss, r3_si = _arrays(r3)

    BRANCH = r1_ep[-1]          # 16 — r2 and r3 both branch here
    r2_abs = r2_ep + BRANCH
    r3_abs = r3_ep + BRANCH

    C1, C2, C3 = "#4878CF", "#D65F0E", "#2CA02C"

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(13, 8.5),
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.05},
    )

    # ── Top panel: total val loss ─────────────────────────────────────────────
    ax.plot(r1_ep, r1_loss, "o-", color=C1, lw=2, ms=5.5, label="r1 (TBPTL bug, lr=1e-4)", zorder=3)
    ax.plot(r2_abs, r2_loss, "s--", color=C2, lw=2, ms=7, label="r2 (low LR, bug persists)", zorder=3)
    ax.plot(r3_abs, r3_loss, "^-", color=C3, lw=2.5, ms=7, label="r3 (dec_use_tbptl=False)", zorder=3)

    # Projection for r3 future epochs
    proj_x, proj_y = _project_r3(r3_ep, r3_loss, future_eps=15)
    proj_abs = proj_x + BRANCH
    ax.plot(proj_abs, proj_y, "--", color=C3, lw=1.5, alpha=0.45, zorder=2)
    ax.fill_between(proj_abs, proj_y * 0.88, proj_y * 1.12,
                    color=C3, alpha=0.08, zorder=1)
    ax.annotate(
        f"proj. ep15\n~{proj_y[-1]:.1f}",
        xy=(proj_abs[-1], proj_y[-1]),
        xytext=(proj_abs[-1] - 1.2, proj_y[-1] * 0.65),
        fontsize=8, color=C3, alpha=0.75,
        arrowprops=dict(arrowstyle="->", color=C3, lw=0.9, alpha=0.6),
    )

    # Dashed connectors from r1 endpoint to branches
    ax.plot([BRANCH, r2_abs[0]], [r1_loss[-1], r2_loss[0]], "--", color=C2, lw=1.1, alpha=0.35, zorder=1)
    ax.plot([BRANCH, r3_abs[0]], [r1_loss[-1], r3_loss[0]], "-",  color=C3, lw=1.1, alpha=0.35, zorder=1)

    # Branch vertical
    ax.axvline(BRANCH, color="gray", lw=1.1, ls=":", alpha=0.6, zorder=1)

    # TBPTL-bug zone (r1 ep2 → r2 end)
    ax.axvspan(1.5, BRANCH + r2_ep[-1] + 0.5, alpha=0.06, color="#CC0000", zorder=0)

    # Phase transition annotation  (r3 local ep4 = abs 20)
    pt_idx = np.where(r3_ep == 4)[0]
    if pt_idx.size:
        pt_x = r3_abs[pt_idx[0]]
        pt_y = r3_loss[pt_idx[0]]
        ax.annotate(
            "Phase transition\n(ScaleInv: 147→3.6)",
            xy=(pt_x, pt_y),
            xytext=(pt_x - 3.5, pt_y * 5),
            fontsize=8.5, color=C3,
            arrowprops=dict(arrowstyle="->", color=C3, lw=1.3, connectionstyle="arc3,rad=-0.25"),
            ha="center", va="bottom",
        )

    # r1 ep1 annotation: RAPTOR intact
    ax.annotate(
        "RAPTOR intact\n(ScaleInv=4.7)",
        xy=(1, r1_loss[0]),
        xytext=(3, r1_loss[0] * 1.5),
        fontsize=8, color=C1,
        arrowprops=dict(arrowstyle="->", color=C1, lw=1.1, connectionstyle="arc3,rad=0.3"),
        ha="left", va="bottom",
    )

    ax.set_yscale("log")
    ax.set_ylabel("Val Loss (log scale)", fontsize=11)
    ax.set_title(
        "LoopCUT3R Stage 1 Convergence  —  r1 → {r2, r3} branches from epoch 16",
        fontsize=12.5, pad=8,
    )
    tbptl_patch = mpatches.Patch(color="#CC0000", alpha=0.2, label="TBPTL bug active")
    ax.legend(handles=ax.get_legend_handles_labels()[0] + [tbptl_patch],
              labels=ax.get_legend_handles_labels()[1] + ["TBPTL bug active"],
              loc="upper right", fontsize=9, framealpha=0.85)
    ax.grid(True, which="both", ls="--", alpha=0.3)

    xmin, xmax = 0.5, BRANCH + 15 + 1.2   # extend to projected ep 15
    ax.set_xlim(xmin, xmax)

    # ── X ticks shared ───────────────────────────────────────────────────────
    # r1: every 2 epochs; branch: every actual r3 epoch + proj key points
    r1_ticks = [int(e) for e in r1_ep if int(e) % 2 == 0 or int(e) == 1]
    branch_ticks = [int(BRANCH) + e for e in list(map(int, r3_ep)) + [10, 15]]
    all_ticks = sorted(set(r1_ticks + branch_ticks))

    def tick_label(x):
        x = int(x)
        if x <= int(BRANCH):
            return f"r1·{x}"
        return f"+{x - int(BRANCH)}"

    tick_labels = [tick_label(x) for x in all_ticks]

    ax.set_xticks(all_ticks)
    ax.set_xticklabels(tick_labels, fontsize=7.5, rotation=45, ha="right")
    ax.tick_params(labelbottom=False)   # hide on top panel, shown on bottom

    # ── Bottom panel: ScaleInv cross (sum of 4 views) ─────────────────────────
    si_mask1 = ~np.isnan(r1_si)
    ax2.plot(r1_ep[si_mask1], r1_si[si_mask1], "o-", color=C1, lw=1.8, ms=5, label="r1")
    ax2.plot(r2_abs, r2_si, "s--", color=C2, lw=1.8, ms=6)
    ax2.plot(r3_abs, r3_si, "^-", color=C3, lw=2, ms=6)

    ax2.axvline(BRANCH, color="gray", lw=1.1, ls=":", alpha=0.6)
    ax2.axvspan(1.5, BRANCH + r2_ep[-1] + 0.5, alpha=0.06, color="#CC0000")
    ax2.set_ylabel("ScaleInv cross\n(Σ 4 views)", fontsize=9.5)
    ax2.set_xlabel("Epoch  [r1·N = global epoch N  |  +N = N epochs after branch]", fontsize=10)
    ax2.grid(True, which="both", ls="--", alpha=0.3)
    ax2.set_xlim(xmin, xmax)
    ax2.set_xticks(all_ticks)
    ax2.set_xticklabels(tick_labels, fontsize=7.5, rotation=45, ha="right")

    # Key ScaleInv values annotated
    ax2.annotate(f"4.7 → 154\n(1 epoch)", xy=(2, r1_si[1]), xytext=(4.5, 100),
                 fontsize=7.5, color=C1,
                 arrowprops=dict(arrowstyle="->", color=C1, lw=1, connectionstyle="arc3,rad=0.2"))

    out = Path(__file__).parent / "stage1_full_convergence.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    plot()
