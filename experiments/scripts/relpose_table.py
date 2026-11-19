#!/usr/bin/env python3
"""Promedia relpose por dataset sobre las escenas que TODAS las etiquetas puntuaron.

Por que la interseccion: si una etiqueta se cae en una escena dificil y se
promedia sobre lo que le quedo, gana por abandono. La auditoria de
docs/benchmark_audit.md ya fijo esta regla; aqui esta como codigo.

Lee `_error_log.txt`, que trae una linea por escena:
    scannet-scene0000_00  | ATE: 1.72614, RPE trans: 0.99819, RPE rot: 58.71266

    python tools/ab/relpose_table.py scannet LoopCUT3R_free_stack CUT3R_fixed ...
"""
import os, re, sys

RAIZ = "eval_results/relpose"
LINEA = re.compile(r"^\s*([\w\-]+?)-(\S+)\s*\|\s*ATE:\s*([-\d.eE]+),\s*"
                   r"RPE trans:\s*([-\d.eE]+),\s*RPE rot:\s*([-\d.eE]+)")


def leer(data, etiqueta):
    d = os.path.join(RAIZ, f"{data}_{etiqueta}")
    f = os.path.join(d, "_error_log.txt")
    if not os.path.isfile(f):
        return None
    out = {}
    for L in open(f):
        m = LINEA.match(L)
        if m:
            _, escena, ate, rt, rr = m.groups()
            out[escena] = (float(ate), float(rr), float(rt))
    return out or None


def main():
    data, etiquetas = sys.argv[1], sys.argv[2:]
    datos = {e: leer(data, e) for e in etiquetas}
    faltan = [e for e, v in datos.items() if v is None]
    for e in faltan:
        print(f"  (sin resultados: {e})")
        datos.pop(e)
    if not datos:
        sys.exit(f"ninguna etiqueta con resultados para {data}")
    comun = set.intersection(*(set(v) for v in datos.values()))
    todas = set.union(*(set(v) for v in datos.values()))
    print(f"\n-- {data}: {len(comun)} escenas comunes de {len(todas)} vistas"
          + (f"  (fuera: {sorted(todas - comun)})" if todas - comun else ""))
    print(f"   {'etiqueta':<34} {'ATE':>9} {'RPE_rot':>9} {'RPE_t':>9}")
    filas = []
    for e, v in datos.items():
        n = len(comun)
        a = sum(v[s][0] for s in comun) / n
        r = sum(v[s][1] for s in comun) / n
        t = sum(v[s][2] for s in comun) / n
        filas.append((e, a, r, t))
    for e, a, r, t in filas:
        print(f"   {e:<34} {a:>9.4f} {r:>9.3f} {t:>9.4f}")
    # multiplicadores contra la primera etiqueta que tenga 'CUT3R' en el nombre
    ref = next((f for f in filas if "CUT3R_" in f[0] and not f[0].startswith("Loop")), None)
    if ref:
        print(f"\n   multiplicadores contra {ref[0]} (>1.00x = peor)")
        for e, a, r, t in filas:
            if e == ref[0]:
                continue
            print(f"   {e:<34} {a/ref[1]:>8.2f}x {r/ref[2]:>8.2f}x {t/ref[3]:>8.2f}x")


if __name__ == "__main__":
    main()
