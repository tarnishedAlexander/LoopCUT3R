#!/usr/bin/env python3
"""
Separa el error de pose del benchmark en ESCALA y FORMA, sin correr el modelo.

Por que: la seccion 5f deja una hipotesis sin medir — stage3f repite mejor la
forma y se equivoca mas en el tamanio (`alpha` 1.0300 -> 1.1120 en la sonda).
Pero el ATE y los RPE del benchmark YA dividen una escala global por secuencia
(Sim(3), `correct_scale=True` en eval/relpose/evo_utils.py), y la rotacion no
depende de la escala. Asi que una escala global mala NO puede explicar lo que
el benchmark perdio. Lo que si podria: una escala que DERIVA dentro de la
secuencia, que un unico factor no corrige.

Lee los `pred_traj.txt` ya guardados y la GT, y da por etiqueta:

  ATE_sim3   el del benchmark (control: tiene que reproducir relpose_table.py)
  ATE_se3    sin corregir escala: cuanto del error es tamanio global
  |ln s|     cuan lejos de 1 esta la escala global ajustada
  deriva     |ln(s_2a_mitad / s_1a_mitad)|: escala que cambia DENTRO de la secuencia
  ATE_2s     ATE con una escala por mitad: lo que queda si se perdona la deriva
  RPE_rot    control, invariante a la escala

    python tools/ab/escala_vs_forma_pose.py tum LoopCUT3R_stage3b_rerun ABL_rope_solo LoopCUT3R_stage3f CUT3R_fixed
"""
import os, sys, copy
import numpy as np

sys.path.insert(0, ".")
from eval.relpose.metadata import dataset_metadata
from eval.relpose.evo_utils import load_traj, make_traj
from evo.core import sync, geometry
from evo.core.metrics import PoseRelation, Unit
from evo.main_ape import ape
from evo.main_rpe import rpe

data, labels = sys.argv[1], sys.argv[2:]
md = dataset_metadata[data]
root = "eval_results/relpose"


def cargar(label, seq):
    p = f"{root}/{data}_{label}/{seq}/pred_traj.txt"
    if not os.path.exists(p):
        return None
    a = np.loadtxt(p)
    pred = make_traj([a[:, 1:], a[:, 0]])
    gt_file = md["gt_traj_func"](md["img_path"], md.get("anno_path"), seq)
    fmt = md.get("traj_format")
    gt = make_traj(load_traj(gt_traj_file=gt_file) if data == "sintel"
                   else load_traj(gt_traj_file=gt_file, traj_format=fmt))
    if pred.timestamps.shape[0] == gt.timestamps.shape[0]:
        pred.timestamps = gt.timestamps
    else:
        gt, pred = sync.associate_trajectories(gt, pred)
    return gt, pred


def escala(gt_xyz, pr_xyz):
    try:
        _, _, s = geometry.umeyama_alignment(pr_xyz.T, gt_xyz.T, True)
        return float(s)
    except Exception:
        return np.nan


def ate_dos_escalas(gt, pr):
    """ATE alineando cada mitad con su propio Sim(3)."""
    n = len(gt.positions_xyz); h = n // 2
    err = []
    for sl in (slice(0, h), slice(h, n)):
        g, p = gt.positions_xyz[sl], pr.positions_xyz[sl]
        try:
            r, t, s = geometry.umeyama_alignment(p.T, g.T, True)
        except Exception:
            return np.nan
        err.append(np.linalg.norm((s * (r @ p.T)).T + t - g, axis=1))
    return float(np.sqrt(np.mean(np.concatenate(err) ** 2)))


seqs = [s for s in sorted(os.listdir(f"{root}/{data}_{labels[0]}"))
        if os.path.isdir(f"{root}/{data}_{labels[0]}/{s}")]
comunes = [s for s in seqs
           if all(os.path.exists(f"{root}/{data}_{l}/{s}/pred_traj.txt") for l in labels)]
def medir(l, s):
    gt, pr = cargar(l, s)
    k = dict(est_name="t", pose_relation=PoseRelation.translation_part)
    a_sim = ape(copy.deepcopy(gt), copy.deepcopy(pr), align=True, correct_scale=True, **k).stats["rmse"]
    a_se = ape(copy.deepcopy(gt), copy.deepcopy(pr), align=True, correct_scale=False, **k).stats["rmse"]
    rot = rpe(copy.deepcopy(gt), copy.deepcopy(pr), est_name="t",
              pose_relation=PoseRelation.rotation_angle_deg, align=True, correct_scale=True,
              delta=1, delta_unit=Unit.frames, rel_delta_tol=0.01, all_pairs=True).stats["rmse"]
    g, p = gt.positions_xyz, pr.positions_xyz
    h = len(g) // 2
    sg = escala(g, p); s1 = escala(g[:h], p[:h]); s2 = escala(g[h:], p[h:])
    return (a_sim, a_se, ate_dos_escalas(gt, pr), abs(np.log(sg)), abs(np.log(s2 / s1)), rot)


# Una escena cuenta solo si se puede medir en TODAS las etiquetas: una GT casi
# estatica hace degenerado el Umeyama (pasa en sintel) y el benchmark la tira.
tabla, fuera = {l: [] for l in labels}, []
for s in comunes:
    try:
        fila = {l: medir(l, s) for l in labels}
    except Exception as e:
        fuera.append(s); continue
    for l in labels:
        tabla[l].append(fila[l])
print(f"-- {data}: {len(comunes) - len(fuera)} escenas comunes" + (f"  (fuera: {fuera})" if fuera else ""))
print(f"   {'etiqueta':<28}{'ATE_sim3':>9}{'ATE_se3':>9}{'ATE_2s':>9}{'|ln s|':>8}{'deriva':>8}{'RPE_rot':>9}")
for l in labels:
    R = np.array(tabla[l])
    m = np.nanmean(R, 0); med = np.nanmedian(R, 0)
    print(f"   {l:<28}{m[0]:>9.4f}{m[1]:>9.4f}{m[2]:>9.4f}{med[3]:>8.3f}{med[4]:>8.3f}{m[5]:>9.3f}")
print("   (ATE y RPE: media sobre escenas, como el benchmark. |ln s| y deriva: MEDIANA,")
print("    porque una mitad casi estatica da una escala degenerada y dispara la media.)")
