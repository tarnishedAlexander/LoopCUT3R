#!/usr/bin/env python3
"""
Figuras del capitulo 5 de la tesis, en estilo de articulo: PDF vectorial,
tipografia tipo Times, dos colores fijos que tambien se distinguen en gris.

Todo sale de datos ya medidos (docs/results/*.json); no corre el modelo.
La figura de la sonda necesita los .npz de experiments/ y se omite si faltan.

    python tools/tesis/figuras.py            # escribe docs/figures/*.pdf
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.3,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
C_CUT, C_LOOP, C_GRIS = "#1a1a1a", "#c8553d", "#8a8a8a"
OUT = "docs/figures"
os.makedirs(OUT, exist_ok=True)
W = 5.9  # ancho de texto en pulgadas


def guardar(fig, nombre):
    fig.savefig(f"{OUT}/{nombre}.pdf")
    plt.close(fig)
    print("escrita", f"{OUT}/{nombre}.pdf")


# ---------------------------------------------------------------- curvas por fase
def curvas():
    d = json.load(open("docs/results/training_curves.json"))
    fases = [("Fase 1", "recurrent_stage1_r3", 0), ("Fase 2", "recurrent_stage2", 2), ("Fase 3", "recurrent_stage3", 0),
             ("Fase 3", "recurrent_stage3b", 0)]
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.2))
    x0, bordes, etiquetas = 0, [], []
    for i, (nom, run, saltar) in enumerate(fases):
        tr = d[run]["train"][saltar:]          # stage2: las 2 primeras son un arranque fallido
        xs = [x0 + k + 1 for k in range(len(tr))]
        ax[0].plot(xs, [t[2] if t[2] < 5 else np.nan for t in tr], color=C_LOOP, marker="o", ms=2.5)
        va = {}
        for e, v in d[run]["val"]:
            va[e] = v                          # ultima medida de cada epoca
        if run == "recurrent_stage2":
            va = {e: v for e, v in va.items()}
        ve = sorted(va)
        ax[1].plot([x0 + e for e in ve], [va[e] for e in ve], color=C_LOOP, marker="o", ms=2.5)
        if run != "recurrent_stage3":
            bordes.append(x0 + len(tr))
        if run != "recurrent_stage3b":
            etiquetas.append((nom, x0))
        x0 += len(tr)
    ini = [0] + bordes[:-1]
    for a in ax:
        for b in bordes[:-1]:
            a.axvline(b + 0.5, color=C_GRIS, lw=0.6, ls=":")
        a.set_xlabel("Época acumulada")
        a.set_xlim(0.5, x0 + 0.5)
    for (nom, _), a0, b0 in zip([("Fase 1 (224)", 0), ("Fase 2 (224)", 0), ("Fase 3 (512)", 0)], ini, bordes):
        for a in ax:
            a.text((a0 + b0) / 2 + 0.5, 0.97, nom, transform=a.get_xaxis_transform(), ha="center", va="top", fontsize=8, color=C_GRIS)
    ax[0].set_ylabel("Pérdida de pose"); ax[0].set_title("(a) Entrenamiento", loc="left")
    ax[1].set_ylabel("Pérdida de pose"); ax[1].set_title("(b) Validación (ARKitScenes)", loc="left")
    for a in ax:
        a.set_ylim(0.15, 0.52)
    fig.tight_layout(w_pad=1.5)
    guardar(fig, "curvas_fases")


# ---------------------------------------------------------------- por escena
def por_escena():
    d = json.load(open("docs/results/pose_by_scene.json"))
    A, B = "CUT3R_fixed", "LoopCUT3R_stage3b_rerun"
    mk = {"scannet": ("o", "ScanNet"), "tum": ("s", "TUM-dynamics"), "sintel": ("^", "Sintel")}
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.15))
    for a, (k, nom) in zip(ax, [("ate", "ATE (m)"), ("rpe_t", "RPE traslación (m)"), ("rpe_r", "RPE rotación (°)")]):
        lo, hi = 1e9, 0
        for ds, (m, lab) in mk.items():
            x = [d[ds][A][s][k] for s in d[ds][A]]; y = [d[ds][B][s][k] for s in d[ds][A]]
            a.scatter(x, y, marker=m, s=14, facecolors="none" if ds != "scannet" else C_LOOP, edgecolors=C_LOOP, linewidths=0.8, label=lab)
            lo, hi = min(lo, *x, *y), max(hi, *x, *y)
        lim = (lo * 0.6, hi * 1.6)
        a.plot(lim, lim, color=C_CUT, lw=0.7, ls="--")
        a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(lim); a.set_ylim(lim); a.set_aspect("equal")
        a.set_title(nom, loc="left"); a.set_xlabel("CUT3R")
    ax[0].set_ylabel("Loop-CUT3R")
    ax[2].legend(loc="lower right", handletextpad=0.1, borderpad=0.1)
    fig.tight_layout(w_pad=0.8)
    guardar(fig, "pose_por_escena")


# ---------------------------------------------------------------- memoria y velocidad
def eficiencia():
    c = json.load(open("docs/results/efficiency_cut3r.json")); l = json.load(open("docs/results/efficiency_loop.json"))
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.2), gridspec_kw={"width_ratios": [1.5, 1]})
    for dd, col, nom, mkr in ((c, C_CUT, "CUT3R", "o"), (l, C_LOOP, "Loop-CUT3R", "s")):
        s = [r for r in dd["runs"] if r["mode"] == "stream"]
        ax[0].plot([r["frames"] for r in s], [r["peak_gb"] for r in s], color=col, marker=mkr, ms=3.5, label=nom)
        ax[0].axhline(dd["weights_gb"], color=col, lw=0.6, ls=":")
    ax[0].axhline(6, color=C_GRIS, lw=0.8, ls="--"); ax[0].text(300, 6.12, "6 GB", ha="right", va="bottom", fontsize=8, color=C_GRIS)
    ax[0].set_xlabel("Fotogramas de la secuencia"); ax[0].set_ylabel("VRAM pico (GB)"); ax[0].set_title("(a) Memoria en streaming", loc="left")
    ax[0].set_ylim(0, 8.4); ax[0].legend(loc="upper left")
    f = lambda dd: [r["fps"] for r in dd["runs"] if r["mode"] == "stream" and r["frames"] == 13][0]
    ax[1].bar([0, 1], [f(c), f(l)], color=[C_CUT, C_LOOP], width=0.55)
    for i, v in enumerate([f(c), f(l)]):
        ax[1].text(i, v + 0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["CUT3R", "Loop-CUT3R"]); ax[1].set_ylabel("FPS"); ax[1].set_ylim(0, 56)
    ax[1].set_title("(b) Velocidad, 512×288", loc="left")
    fig.tight_layout(w_pad=1.5)
    guardar(fig, "eficiencia_5090")


# ---------------------------------------------------------------- progresion por fase
def progresion():
    p = "docs/results/pose_by_phase.json"
    if not os.path.exists(p):
        print("falta", p); return
    d = json.load(open(p))
    orden = [("LoopCUT3R_F1_stage1r3_ep12_224", "F1"), ("LoopCUT3R_F2_stage2_final_224", "F2"),
             ("LoopCUT3R_recurrent_stage3_checkpoint_final", "F3\n(ép. 10)"), ("LoopCUT3R_stage3b_rerun", "F3\n(final)")]
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.1))
    est = {"scannet": ("-", "o", "ScanNet"), "tum": ("--", "s", "TUM-dynamics"), "sintel": (":", "^", "Sintel")}
    for a, (k, nom) in zip(ax, [("ate", "ATE (m)"), ("rpe_t", "RPE traslación (m)"), ("rpe_r", "RPE rotación (°)")]):
        for ds, (ls, m, lab) in est.items():
            y = [np.mean([v[k] for v in d[ds][l].values()]) for l, _ in orden]
            a.plot(range(4), y, color=C_LOOP, ls=ls, marker=m, ms=3.5, label=lab)
            a.plot([-0.25, 3.25], [np.mean([v[k] for v in d[ds]["CUT3R_fixed"].values()])] * 2, color=C_CUT, ls=ls, lw=0.9)
        a.set_yscale("log"); a.set_xticks(range(4)); a.set_xticklabels([e for _, e in orden]); a.set_title(nom, loc="left")
    ax[0].legend(loc="center right", fontsize=7)
    fig.tight_layout(w_pad=0.8)
    guardar(fig, "progresion_fases")


# ---------------------------------------------------------------- sonda de autoconsistencia
def sonda():
    D, SEQ, CONF = ".scratch/ab/recon", "rep_002_01_x8", 1.5
    if not os.path.isdir(D):
        print("sonda: faltan los datos de", D, "- se omite"); return
    def carga(m):
        d = np.load(f"{D}/{m}_{SEQ}.npz", allow_pickle=True)
        pts, conf = d["pts_posed"], d["conf_cam"]
        return pts, (conf > CONF) & np.isfinite(pts).all(-1)
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.15), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    cm = matplotlib.colormaps["copper"]
    for a, (m, nom) in zip(ax[:2], [("cut3r", "(a) CUT3R"), ("loop3b", "(b) Loop-CUT3R")]):
        pts, ok = carga(m)
        n, H, Wd = pts.shape[:3]
        base = pts[0][ok[0]]; R = np.median(np.linalg.norm(base - base.mean(0), axis=-1)); c0 = base.mean(0)
        for i in range(n):
            z = np.where(ok[i, H // 2], (pts[i, H // 2, :, 2] - c0[2]) / R, np.nan)
            a.plot(np.arange(Wd), z, lw=0.8, color=cm(i / (n - 1)))
        a.set_title(nom, loc="left"); a.set_xlabel("Columna de la imagen (px)")
    ax[0].set_ylabel("Profundidad normalizada")
    sm = matplotlib.cm.ScalarMappable(cmap=cm, norm=matplotlib.colors.Normalize(1, 8))
    cb = fig.colorbar(sm, ax=ax[1], fraction=0.05, pad=0.02); cb.set_label("Repetición", fontsize=7); cb.ax.tick_params(labelsize=6)
    for m, col, nom, mk in (("cut3r", C_CUT, "CUT3R", "o"), ("loop3b", C_LOOP, "Loop-CUT3R", "s")):
        pts, ok = carga(m)
        base = pts[0][ok[0]]; R = np.median(np.linalg.norm(base - base.mean(0), axis=-1))
        ys = [np.median(np.linalg.norm(pts[i][ok[0] & ok[i]] - pts[0][ok[0] & ok[i]], axis=-1)) / R for i in range(len(pts))]
        ax[2].plot(range(1, len(pts) + 1), ys, color=col, marker=mk, ms=3, label=nom)
    ax[2].set_title("(c) Desplazamiento acumulado", loc="left"); ax[2].set_xlabel("Repetición de la imagen")
    ax[2].set_ylabel("Desplazamiento / radio de escena"); ax[2].legend(loc="center right"); ax[2].set_xticks(range(1, 9))
    fig.tight_layout(w_pad=0.6)
    guardar(fig, "sonda_deriva")


# ---------------------------------------------------------------- barrido de pasadas del encoder
def barrido():
    # medidos con los pesos finales, cambiando solo el numero de pasadas en inferencia
    cob = {1: 0, 2: 29078, 3: 97744, 4: 104140, 5: 111319, 6: 111598, 7: 111527, 8: 110163, 9: 109525, 10: 108557, 12: 99567, 16: 19354}
    res = {3: .0517, 4: .0722, 5: .0829, 6: .0921, 7: .1061, 8: .1212, 9: .1230, 10: .1444, 12: .1653}
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.1))
    ax[0].plot(list(cob), [v / 1000 for v in cob.values()], color=C_LOOP, marker="o", ms=3)
    ax[0].set_ylabel("Píxeles confiables (miles)"); ax[0].set_title("(a) Cobertura", loc="left")
    ax[1].plot(list(res), list(res.values()), color=C_LOOP, marker="s", ms=3)
    ax[1].set_ylabel(r"Residuo de forma (resid)"); ax[1].set_title("(b) Deriva", loc="left")
    for a in ax:
        a.axvline(8, color=C_GRIS, lw=0.8, ls="--"); a.set_xlabel("Pasadas del encoder en inferencia")
        a.text(8.2, 0.04, "entrenado", transform=a.get_xaxis_transform(), fontsize=7, color=C_GRIS)
    fig.tight_layout(w_pad=1.5)
    guardar(fig, "barrido_pasadas")


if __name__ == "__main__":
    curvas(); por_escena(); eficiencia(); progresion(); sonda(); barrido()
