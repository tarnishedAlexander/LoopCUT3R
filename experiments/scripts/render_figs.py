#!/usr/bin/env python3
"""Figuras de las reconstrucciones ya volcadas. No corre el modelo.

Lee los .npz de .scratch/ab/recon, asi que las figuras salen de EXACTAMENTE los
mismos datos que produjeron las tablas, y se regeneran sin los checkpoints.

POR QUE ESTE DISENO Y NO UNA NUBE DE PUNTOS
-------------------------------------------
Primer intento: dibujar la nube entera en vista lateral, coloreada por pasada.
No sirve — la deriva es ~10% del radio de la escena y queda enterrada en el
grosor de la nube; los tres modelos salian identicos. Lo que la hace visible es
reducir una dimension:

  (a) UNA fila de pixeles: su perfil de profundidad, una curva por pasada. Ocho
      curvas que deberian coincidir. Si se abren en abanico, eso es el `resid`.
  (b) La DISTRIBUCION del desplazamiento por pasada respecto a la pasada 0.

    python tools/ab/render_figs.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = ".scratch/ab/recon"
OUT = "docs/figs"
CONF = 1.5
CMAP = matplotlib.colormaps["turbo"]


def carga(modelo, seq):
    p = f"{D}/{modelo}_{seq}.npz"
    if not os.path.isfile(p):
        return None
    d = np.load(p, allow_pickle=True)
    pts, conf = d["pts_posed"], d["conf_cam"]
    return pts, (conf > CONF) & np.isfinite(pts).all(-1)


def perfil(ax, modelo, seq, titulo, nota):
    """Una fila de pixeles, su profundidad, una curva por pasada."""
    r = carga(modelo, seq)
    if r is None:
        ax.text(.5, .5, "sin datos", ha="center", transform=ax.transAxes); return
    pts, ok = r
    n, H, W = pts.shape[:3]
    fila = H // 2
    base = pts[0][ok[0]] if ok[0].any() else pts[0].reshape(-1, 3)
    if not ok[0].any():
        ax.text(.5, .5, "CERO pixeles\nsobre el umbral", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="crimson")
        ax.set_title(titulo, fontsize=9); ax.set_xticks([]); ax.set_yticks([]); return
    R = np.median(np.linalg.norm(base - base.mean(0), axis=-1)) or 1.0
    c0 = base.mean(0)
    for i in range(n):
        z = (pts[i, fila, :, 2] - c0[2]) / R
        m = ok[i, fila]
        z = np.where(m, z, np.nan)
        ax.plot(np.arange(W), z, lw=1.1, color=CMAP(i / max(n - 1, 1)), alpha=.9)
    ax.set_title(titulo, fontsize=9)
    ax.set_xlabel(nota, fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=.15)


def deriva(ax, modelos, etiquetas, seq):
    """Mediana del desplazamiento respecto a la pasada 0, pasada a pasada."""
    for mod, et in zip(modelos, etiquetas):
        r = carga(mod, seq)
        if r is None:
            continue
        pts, ok = r
        n = len(pts)
        m0 = ok[0]
        if not m0.any():
            continue
        base = pts[0][m0]
        R = np.median(np.linalg.norm(base - base.mean(0), axis=-1)) or 1.0
        ys = []
        for i in range(n):
            m = m0 & ok[i]
            ys.append(np.median(np.linalg.norm(pts[i][m] - pts[0][m], axis=-1)) / R
                      if m.any() else np.nan)
        ax.plot(range(n), ys, "o-", ms=3.5, lw=1.4, label=et)
    ax.set_xlabel("pasada sobre la MISMA imagen", fontsize=8.5)
    ax.set_ylabel("desplazamiento mediano / radio de escena\n(SIN corregir "
                  "la escala global)", fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.grid(alpha=.2)
    ax.legend(fontsize=7.5)


REP = "rep_002_01_x8"
os.makedirs(OUT, exist_ok=True)

# fig1: el perfil de una fila + la curva de deriva
fig = plt.figure(figsize=(12.5, 6.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=.33, wspace=.22)
for j, (mod, tit, nota) in enumerate([
        ("cut3r",  "CUT3R   resid 0.0082",      "las 8 pasadas coinciden"),
        ("loop3b", "LoopCUT3R   resid 0.1141",  "las 8 pasadas se abren en abanico"),
        ("prime",  "+ priming   resid 0.0410",  "casi coinciden")]):
    perfil(fig.add_subplot(gs[0, j]), mod, REP, tit, nota)
ax = fig.add_subplot(gs[1, :])
deriva(ax, ["cut3r", "prime", "loop3b", "es12"],
       ["CUT3R", "LoopCUT3R + priming", "LoopCUT3R (8 aplicaciones)",
        "LoopCUT3R (12 aplicaciones)"], REP)
ax.set_title("Y la deriva, acumulada pasada a pasada", fontsize=9.5)
fig.suptitle("Misma imagen 8 veces: arriba el perfil de profundidad de UNA fila "
             "de pixeles, una curva por pasada", fontsize=11)
fig.savefig(f"{OUT}/fig1_deriva.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  escrito {OUT}/fig1_deriva.png")

# fig2: el barrido de aplicaciones
# OJO: aqui habia un panel con la curva de desplazamiento por aplicacion, y se
# quito porque CONTRADECIA su propio titulo: a 3 aplicaciones el desplazamiento
# crudo sale el MAS alto. No es un error de los datos. `resid` divide por la
# deriva de escala global (alpha, que llega a 1.25) y el desplazamiento crudo
# no, asi que ordenan distinto. El panel que queda usa `resid`, que es la
# metrica de las tablas.
fig, axes = plt.subplots(1, 1, figsize=(6.6, 4.4))
axes = [None, axes]
cob = {"es1": 0, "es2": 29078, "es3": 97744, "es4": 104140, "es5": 111319,
       "es6": 111598, "es7": 111527, "es8": 110163, "es9": 109525,
       "es10": 108557, "es12": 99567, "es16": 19354}
res = {"es3": .0517, "es4": .0722, "es5": .0829, "es6": .0921, "es7": .1061,
       "es8": .1212, "es9": .1230, "es10": .1444, "es12": .1653}
ks = sorted(int(k[2:]) for k in cob)
a2 = axes[1]
a2.plot(ks, [cob[f"es{k}"] / 1000 for k in ks], "o-", color="#2a6", label="cobertura (miles de px)")
a2.set_xlabel("aplicaciones del encoder", fontsize=8.5)
a2.set_ylabel("miles de pixeles confiables", fontsize=8.5, color="#2a6")
a2.axvline(8, color="k", ls="--", lw=.9)
a2.text(8.15, 10, "lo entrenado", fontsize=7.5, rotation=90)
a3 = a2.twinx()
kr = sorted(int(k[2:]) for k in res)
a3.plot(kr, [res[f"es{k}"] for k in kr], "s-", color="#c33", label="resid")
a3.set_ylabel("resid @ pasada 7", fontsize=8.5, color="#c33")
a2.tick_params(labelsize=7.5); a3.tick_params(labelsize=7.5)
a2.grid(alpha=.2)
a2.set_title("La cobertura es una meseta; la deriva no para de subir", fontsize=9.5)
fig.suptitle("Barrido del numero de aplicaciones del encoder\nmismos pesos, sin entrenar nada", fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, .94])
fig.savefig(f"{OUT}/fig2_barrido_pasos.png", dpi=150)
plt.close(fig)
print(f"  escrito {OUT}/fig2_barrido_pasos.png")

# fig3: el swap 2x2
fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
for ax, (mod, tit, nota) in zip(axes, [
        ("swapEE", "enc CUT3R + dec CUT3R", "control: reproduce CUT3R"),
        ("swapEL", "enc CUT3R + dec loop", "colapsa"),
        ("swapLE", "enc loop + dec CUT3R", "colapsa"),
        ("swapLL", "enc loop + dec loop", "control: reproduce ropecold")]):
    perfil(ax, mod, REP, tit, nota)
fig.suptitle("Swap 2x2: las dos celdas cruzadas no superan el umbral de confianza "
             "en NINGUN pixel", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, .92])
fig.savefig(f"{OUT}/fig3_swap.png", dpi=150)
plt.close(fig)
print(f"  escrito {OUT}/fig3_swap.png")

# fig4: mapas de confianza del frame 0 de la escena completa. Un heatmap 2D no
# tiene el problema de proyeccion que hundio el primer intento de nube de
# puntos: se ve directamente DONDE la red no se fia.
fig, axes = plt.subplots(2, 3, figsize=(12.5, 5.6))
for col, (mod, tit) in enumerate([
        ("cut3r",  "CUT3R   146.682 px"),
        ("loop3b", "LoopCUT3R   109.735 px   (-25%)"),
        ("prime",  "+ priming   145.826 px")]):
    d = np.load(f"{D}/{mod}_002.npz", allow_pickle=True)
    conf, rgb = d["conf_cam"][0], d["rgb"][0]
    axes[0][col].imshow(np.clip(rgb, 0, 1))
    axes[0][col].set_title(tit, fontsize=9.5)
    im = axes[1][col].imshow(np.log10(np.maximum(conf, 1.0)), cmap="magma",
                             vmin=0, vmax=np.log10(55))
    axes[1][col].contour(conf, levels=[CONF], colors="cyan", linewidths=.9)
    axes[1][col].set_xlabel("linea cian = umbral 1.5", fontsize=7.5)
    for ax in (axes[0][col], axes[1][col]):
        ax.set_xticks([]); ax.set_yticks([])
axes[0][0].set_ylabel("frame 0", fontsize=9)
axes[1][0].set_ylabel("log10 confianza", fontsize=9)
fig.colorbar(im, ax=axes[1], fraction=.02, pad=.01)
fig.suptitle("Donde la red NO se fia: frame 0 de la escena 002\n"
             "el arranque en frio se come los bordes y abre agujeros", fontsize=10.5)
fig.savefig(f"{OUT}/fig4_confianza.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  escrito {OUT}/fig4_confianza.png")
