#!/usr/bin/env python3
"""
`resid` sobre el MISMO conjunto de pixeles en varios modelos.

Por que: exp_repeat.py enmascara con la confianza de cada modelo, asi que un
modelo que se vuelve menos confiado mide su residuo solo sobre los pixeles
faciles que le quedan y sale mejor sin haber mejorado. Esta es la medida que
separa "mas consistente" de "mide menos cosas".

Es un control, no una metrica de paper: la normalizacion es propia (mediana del
radio respecto al centroide) y NO coincide con la de exp_repeat.py, asi que los
numeros sirven para comparar modelos entre si y no para citarlos sueltos.

    python tools/ab/resid_comun.py loop3b cons_ep0 cons_ep2 prime cut3r
    AB_RECON=/otro/dir python tools/ab/resid_comun.py loop3b prime

Ver docs/attractor_mechanism.md seccion 1d, que es donde este control decidio un
veredicto: la ganancia de la epoca 0 de stage3e era real, y la de `prime` esta
en parte inflada por el cambio de cobertura.
"""
import os, sys, numpy as np

D = os.environ.get("AB_RECON", ".scratch/ab/recon")
MODS = sys.argv[1:] or ["loop3b", "cons_ep0"]
SEQ = "rep_002_01_x8"
CONF = 1.5

cargado = {}
for m in MODS:
    d = np.load(f"{D}/{m}_{SEQ}.npz", allow_pickle=True)
    pts, conf = d["pts_posed"], d["conf_cam"]
    ok = (conf > CONF) & np.isfinite(pts).all(-1)
    cargado[m] = (pts, ok)

# La mascara la definen SOLO los modelos listados en AB_MASK (por defecto,
# todos). Hace falta separarlo porque la interseccion la domina el modelo mas
# pobre: metiendo un checkpoint temprano con 12.643 pixeles confiables, los
# demas quedan medidos sobre ESE recorte y la comparacion pierde sentido.
# Fijando la mascara con los modelos buenos se mide la GEOMETRIA de todos sobre
# el mismo sitio, independientemente de en que se fie cada uno.
mask_mods = os.environ.get("AB_MASK", "").split(",") if os.environ.get("AB_MASK") else MODS
for m in mask_mods:
    if m not in cargado:
        d = np.load(f"{D}/{m}_{SEQ}.npz", allow_pickle=True)
        pts, conf = d["pts_posed"], d["conf_cam"]
        cargado[m] = (pts, (conf > CONF) & np.isfinite(pts).all(-1))
comun = np.ones_like(cargado[mask_mods[0]][1][0], dtype=bool)
for m in mask_mods:
    comun &= cargado[m][1].all(0)
print(f"mascara definida por: {mask_mods}")
print(f"pixeles comunes a {MODS} en los 8 frames: {int(comun.sum())}")
print(f"{'modelo':<12} {'px propios@0':>12} {'resid propio':>13} {'resid comun':>12} {'alpha comun':>12}")
for m in MODS:
    pts, ok = cargado[m]
    p0 = pts[0][comun]
    p7 = pts[7][comun]
    # misma normalizacion de escala que exp_repeat: alpha por mediana de norma
    a = np.median(np.linalg.norm(p7 - p7.mean(0), axis=-1)) / \
        np.median(np.linalg.norm(p0 - p0.mean(0), axis=-1))
    r = np.median(np.linalg.norm(p7 / a - p0, axis=-1)) / \
        np.median(np.linalg.norm(p0 - p0.mean(0), axis=-1))
    propio = ok[0] & ok[7]
    pp0, pp7 = pts[0][propio], pts[7][propio]
    ap = np.median(np.linalg.norm(pp7 - pp7.mean(0), axis=-1)) / \
         np.median(np.linalg.norm(pp0 - pp0.mean(0), axis=-1))
    rp = np.median(np.linalg.norm(pp7 / ap - pp0, axis=-1)) / \
         np.median(np.linalg.norm(pp0 - pp0.mean(0), axis=-1))
    print(f"{m:<12} {int(ok[0].sum()):>12} {rp:>13.4f} {r:>12.4f} {a:>12.4f}")
