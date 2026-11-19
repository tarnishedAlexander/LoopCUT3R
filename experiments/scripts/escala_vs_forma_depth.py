#!/usr/bin/env python3
"""
Separa el error de profundidad del benchmark en ESCALA y FORMA, sin correr el
modelo: lee los `frame_*.npy` de eval_results/video_depth y la GT.

Tres AbsRel, de menos a mas perdon, todos por mediana (sin shift):

  metrico     sin alinear nada              tamanio + deriva + forma
  1 escala    una escala por SECUENCIA      deriva + forma
  por frame   una escala por FRAME          solo forma

y dos numeros de escala:

  |ln s|      cuan lejos de 1 esta la escala global de la secuencia
  sd ln s_t   cuanto cambia la escala de un frame a otro DENTRO de la secuencia

`1 escala - por frame` es lo que cuesta la deriva de escala. Si la hipotesis de
la seccion 5f fuera cierta (stage3f = misma forma, peor tamanio), `por frame`
tendria que mejorar y las otras dos empeorar.

    python tools/ab/escala_vs_forma_depth.py bonn LoopCUT3R_recurrent_stage3b_checkpoint_last ABL_rope_solo LoopCUT3R_stage3f CUT3R_fixed
"""
import glob, os, sys
import cv2, numpy as np
from PIL import Image

data, labels = sys.argv[1], sys.argv[2:]
root = "eval_results/video_depth"


def gt_bonn(f):
    d = np.asarray(Image.open(f)).astype(np.float64) / 5000.0
    return d, d > 0


def gt_sintel(f):
    with open(f, "rb") as h:
        np.fromfile(h, dtype=np.float32, count=1)
        w = int(np.fromfile(h, dtype=np.int32, count=1)[0])
        hh = int(np.fromfile(h, dtype=np.int32, count=1)[0])
        d = np.fromfile(h, dtype=np.float32, count=-1).reshape(hh, w).astype(np.float64)
    return d, (d > 0) & (d < 70)   # mismo recorte que el eval


def gt_paths(seq):
    if data == "bonn":
        return sorted(glob.glob(f"data/bonn/rgbd_bonn_dataset/rgbd_bonn_{seq}/depth_110/*.png")), gt_bonn
    return sorted(glob.glob(f"data/sintel/training/depth/{seq}/*.dpt")), gt_sintel


seqs = sorted(d for d in os.listdir(f"{root}/{data}_{labels[0]}")
              if os.path.isdir(f"{root}/{data}_{labels[0]}/{d}"))
print(f"-- {data}: {len(seqs)} secuencias, AbsRel con alineacion por mediana")
print(f"   {'etiqueta':<46}{'metrico':>9}{'1 escala':>9}{'por frame':>10}{'deriva':>8}{'|ln s|':>8}{'sd ln s_t':>10}")
for l in labels:
    num = np.zeros(3); den = 0; lns, sds = [], []
    for seq in seqs:
        gts, rd = gt_paths(seq)
        prs = sorted(glob.glob(f"{root}/{data}_{l}/{seq}/frame_*.npy"))
        assert len(gts) >= len(prs) > 0, (l, seq, len(gts), len(prs))
        P, G, s_t = [], [], []
        for g, p in zip(gts, prs):
            gt, ok = rd(g)
            pr = cv2.resize(np.load(p), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
            ok &= pr > 1e-6
            if ok.sum() < 100:
                continue
            P.append(pr[ok]); G.append(gt[ok]); s_t.append(np.median(gt[ok]) / np.median(pr[ok]))
        s_t = np.array(s_t)
        allp, allg = np.concatenate(P), np.concatenate(G)
        s = np.median(allg) / np.median(allp)
        num[0] += np.sum(np.abs(allp - allg) / allg)
        num[1] += np.sum(np.abs(s * allp - allg) / allg)
        num[2] += sum(np.sum(np.abs(si * p - g) / g) for si, p, g in zip(s_t, P, G))
        den += len(allg)
        lns.append(abs(np.log(s))); sds.append(np.std(np.log(s_t)))
    a = num / den
    print(f"   {l:<46}{a[0]:>9.4f}{a[1]:>9.4f}{a[2]:>10.4f}{a[1]-a[2]:>8.4f}{np.median(lns):>8.3f}{np.median(sds):>10.3f}")
