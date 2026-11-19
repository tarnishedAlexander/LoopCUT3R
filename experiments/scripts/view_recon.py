#!/usr/bin/env python3
"""
Visor para comparar a ojo las reconstrucciones que ya volco extract_multi.py.

No corre el modelo: lee los .npz de .scratch/ab/recon, asi que cambiar de modelo
es instantaneo y la comparacion es sobre exactamente los mismos datos que
produjeron las tablas de docs/attractor_mechanism.md.

    python tools/ab/view_recon.py                       # dentro del contenedor
    python tools/ab/view_recon.py --port 8888

Luego se abre http://localhost:8888 en el navegador del host (el puerto 8888
esta publicado por docker-compose, igual que 6006 y 5000).

LO QUE HAY QUE MIRAR, y por que
-------------------------------
`Color = frame` es el modo que importa. En las secuencias `rep_*_x8` se le da
al modelo LA MISMA imagen 8 veces, asi que un modelo autoconsistente tiene que
apilar las 8 copias en una sola superficie. Si ves 8 capas desplazadas, de
colores distintos, eso ES el `resid` de las tablas, en version visual.

Comparacion sugerida, con secuencia `rep_002_01_x8` y Color = frame:

    cut3r           resid 0.0082   una sola superficie, las 8 copias pegadas
    loop3b          resid 0.1141   capas separadas: el defecto, sin arreglos
    prime           resid 0.0410   casi pegadas: lo que gana el priming
    rope_ep1        resid 0.1532   peor que loop3b: por eso stage3d se descarto
    rope_ep1_prime  resid 0.0575   entre prime y loop3b
    cons_ep2        resid 0.1131   indistinguible de loop3b: por eso stage3e se descarto

OJO con `cons_ep0` y `cons_ep2`: tienen MENOS puntos que loop3b (100k y 102k
frente a 110k), asi que a ojo parecen mas limpios por tener menos ruido, no por
ser mas consistentes. Para eso esta tools/ab/resid_comun.py, que mide sobre los
mismos pixeles. Ver docs/attractor_mechanism.md seccion 1d.

Con la secuencia `002` (15 frames distintos) lo que se ve es otra cosa: la
cobertura y la deriva acumulada. Ahi `conf@0` y los pixeles se notan como
agujeros y bordes comidos.
"""
import argparse
import os
import re
import sys

import numpy as np

# Fila de metricas de cada modelo, para tener el numero al lado de la imagen.
# Fuente: docs/attractor_mechanism.md secciones 1c y 3b (recomputado 2026-11-07).
METRICAS = {
    "cut3r":          dict(conf=0.995, px=146682, alpha=0.9993, resid=0.0082, rot=None,  ghost="1.00x",        nota="la meta"),
    "loop3b":         dict(conf=0.744, px=109735, alpha=1.1819, resid=0.1141, rot=0.262, ghost="3.57x (10 p)", nota="LoopCUT3R sin arreglos"),
    "ropecold":       dict(conf=0.747, px=110163, alpha=1.1849, resid=0.1212, rot=0.131, ghost="3.39x (10 p)", nota="RoPE encendido, sin entrenar"),
    "rope_ep0":       dict(conf=0.707, px=104307, alpha=1.2015, resid=0.1512, rot=0.119, ghost="3.63x (10 p)", nota="stage3d epoca 0"),
    "rope_ep1":       dict(conf=0.688, px=101417, alpha=1.2174, resid=0.1532, rot=0.125, ghost="3.63x (10 p)", nota="stage3d epoca 1: NEGATIVO"),
    "prime":          dict(conf=0.989, px=145826, alpha=1.0929, resid=0.0410, rot=0.296, ghost="2.77x (11 p)", nota="mejor en geometria"),
    "rope_ep1_prime": dict(conf=0.945, px=139294, alpha=1.1027, resid=0.0575, rot=0.119, ghost="2.88x (11 p)", nota="mejor en rotacion"),
    "cons_ep0":       dict(conf=0.678, px=100001, alpha=1.1940, resid=0.1055, rot=0.097, ghost="3.80x (10 p)", nota="stage3e epoca 0"),
    "cons_ep2":       dict(conf=0.695, px=102506, alpha=1.2010, resid=0.1131, rot=0.122, ghost="3.62x (10 p)", nota="stage3e epoca 2: NEGATIVO en geometria, la mejor pose"),
}


def escanear(recon_dir):
    """{secuencia: {modelo: ruta}} a partir de los nombres `{modelo}_{seq}.npz`.

    Los nombres de modelo se toman de METRICAS y de un prefijo mas largo primero,
    porque `rope_ep1` es prefijo de `rope_ep1_prime` y partir por el primer '_'
    daria modelo='rope' para los dos.
    """
    if not os.path.isdir(recon_dir):
        sys.exit(f"no existe el directorio de reconstrucciones: {recon_dir}")
    modelos = sorted(METRICAS, key=len, reverse=True)
    out = {}
    for f in sorted(os.listdir(recon_dir)):
        if not f.endswith(".npz"):
            continue
        base = f[:-4]
        for m in modelos:
            if base.startswith(m + "_"):
                out.setdefault(base[len(m) + 1:], {})[m] = os.path.join(recon_dir, f)
                break
    if not out:
        sys.exit(f"ningun .npz reconocido en {recon_dir}")
    return out


_cache = {}


def cargar(path):
    if path not in _cache:
        d = np.load(path, allow_pickle=True)
        _cache[path] = {k: d[k] for k in ("pts_posed", "conf_cam", "pose_c2w", "rgb")}
    return _cache[path]


def radio_escena(pts, mask):
    p = pts[mask]
    if len(p) == 0:
        return np.nan
    return float(np.median(np.linalg.norm(p - p.mean(0), axis=-1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon_dir", default=".scratch/ab/recon")
    ap.add_argument("--port", type=int, default=8888)
    ap.add_argument("--host", default="0.0.0.0",
                    help="0.0.0.0 para que sea alcanzable desde el host, no solo "
                         "desde dentro del contenedor")
    args = ap.parse_args()

    import viser
    import matplotlib

    datos = escanear(args.recon_dir)
    seqs = sorted(datos)
    # La secuencia de frame repetido primero: es la que hace visible el `resid`.
    seqs.sort(key=lambda s: (not s.startswith("rep_"), s))

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("-y")

    with server.gui.add_folder("Que mirar"):
        g_seq = server.gui.add_dropdown("Secuencia", seqs, initial_value=seqs[0])
        g_mod = server.gui.add_dropdown("Modelo", sorted(datos[seqs[0]]),
                                        initial_value=sorted(datos[seqs[0]])[0])
        g_info = server.gui.add_markdown("")

    with server.gui.add_folder("Color"):
        g_color = server.gui.add_dropdown(
            "Color", ["frame", "rgb", "confianza"], initial_value="frame")
        g_ayuda = server.gui.add_markdown(
            "**frame** es el modo clave: en `rep_*_x8` es la misma imagen 8 "
            "veces, asi que un modelo consistente las apila en UNA superficie. "
            "Capas de colores separadas = `resid` alto."
        )

    with server.gui.add_folder("Frames"):
        g_hasta = server.gui.add_slider("Mostrar hasta el frame", 0, 1, 1, 0)
        g_solo = server.gui.add_checkbox("Solo este frame", False)
        g_cams = server.gui.add_checkbox("Mostrar camaras", True)

    with server.gui.add_folder("Umbral y escala"):
        g_conf = server.gui.add_slider("Umbral de confianza", 1.0, 5.0, 0.1, 1.5)
        g_norm = server.gui.add_checkbox("Normalizar escala", True)
        g_pts = server.gui.add_slider("Puntos por frame", 1000, 60000, 1000, 20000)
        g_size = server.gui.add_slider("Tamanio de punto", 0.001, 0.05, 0.001, 0.008)

    handles = []
    # Guarda de reentrada: dentro de redraw se ajusta g_hasta.max/.value cuando
    # cambia la secuencia, y asignar .value dispara el on_update -> redraw otra
    # vez. Sin esta guarda es recursion infinita.
    dibujando = [False]

    def redraw(*_):
        if dibujando[0]:
            return
        dibujando[0] = True
        try:
            _redraw()
        finally:
            dibujando[0] = False

    def _redraw():
        for h in handles:
            h.remove()
        handles.clear()

        seq, mod = g_seq.value, g_mod.value
        if mod not in datos[seq]:
            g_info.content = f"**{mod}** no tiene `{seq}`. Modelos con esta secuencia: " \
                             + ", ".join(sorted(datos[seq]))
            return

        d = cargar(datos[seq][mod])
        pts, conf, poses, rgb = d["pts_posed"], d["conf_cam"], d["pose_c2w"], d["rgb"]
        n = len(pts)

        if g_hasta.max != n - 1:
            g_hasta.max = n - 1
            g_hasta.value = n - 1

        mask = (conf > g_conf.value) & np.isfinite(pts).all(-1)
        R = radio_escena(pts, mask) if g_norm.value else 1.0
        if not np.isfinite(R) or R == 0:
            R = 1.0

        m = METRICAS[mod]
        rot = "--" if m["rot"] is None else f"{m['rot']:.3f}"
        g_info.content = (
            f"### {mod}\n*{m['nota']}*\n\n"
            f"| resid@7 | conf@0 | px@0 | alpha@7 | cam_rot@7 | GHOST |\n"
            f"|---|---|---|---|---|---|\n"
            f"| **{m['resid']:.4f}** | {m['conf']:.3f} | {m['px']} | "
            f"{m['alpha']:.4f} | {rot} | {m['ghost']} |\n\n"
            f"Visible ahora: **{int(mask[:g_hasta.value + 1].sum())}** puntos "
            f"sobre el umbral, radio de escena {R:.3f}."
        )

        rng = np.random.default_rng(0)
        lo = g_hasta.value if g_solo.value else 0
        cmap = matplotlib.colormaps["turbo"]

        for i in range(lo, g_hasta.value + 1):
            mk = mask[i]
            if mk.sum() == 0:
                continue
            p = pts[i][mk] / R
            if g_color.value == "rgb":
                c = (np.clip(rgb[i][mk], 0, 1) * 255).astype(np.uint8)
            elif g_color.value == "confianza":
                v = np.clip((conf[i][mk] - 1.0) / 4.0, 0, 1)
                c = (cmap(v)[:, :3] * 255).astype(np.uint8)
            else:  # frame
                v = i / max(n - 1, 1)
                c = np.tile((np.array(cmap(v)[:3]) * 255).astype(np.uint8), (len(p), 1))
            if len(p) > g_pts.value:
                sel = rng.choice(len(p), int(g_pts.value), replace=False)
                p, c = p[sel], c[sel]
            handles.append(server.scene.add_point_cloud(
                f"/f{i}", points=p.astype(np.float32), colors=c,
                point_size=float(g_size.value), point_shape="circle"))

        if g_cams.value:
            from scipy.spatial.transform import Rotation
            for i in range(lo, g_hasta.value + 1):
                T = poses[i]
                q = Rotation.from_matrix(T[:3, :3]).as_quat()  # x,y,z,w
                handles.append(server.scene.add_camera_frustum(
                    f"/cam{i}", fov=1.0, aspect=16 / 9, scale=0.08 / max(R, 1e-6) * R,
                    wxyz=np.array([q[3], q[0], q[1], q[2]]),
                    position=T[:3, 3] / R,
                    color=(255, 80, 80) if i == g_hasta.value else (120, 120, 120)))

    def on_seq(*_):
        opts = sorted(datos[g_seq.value])
        g_mod.options = opts
        if g_mod.value not in opts:
            g_mod.value = opts[0]
        redraw()

    g_seq.on_update(on_seq)
    for g in (g_mod, g_color, g_hasta, g_solo, g_cams, g_conf, g_norm, g_pts, g_size):
        g.on_update(redraw)

    redraw()
    print(f"\nvisor en http://localhost:{args.port}", flush=True)
    print(f"secuencias: {', '.join(seqs)}", flush=True)
    print("ctrl-c para salir\n", flush=True)
    import time
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
