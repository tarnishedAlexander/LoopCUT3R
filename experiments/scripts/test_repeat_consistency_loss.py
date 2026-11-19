#!/usr/bin/env python3
"""
Control de `RepeatConsistencyLoss` (src/dust3r/losses.py).

Se prueban las propiedades de las que depende el experimento, no que "corre".
La que mas importa es la 4: si el gradiente llegara a la cabeza de confianza,
el modelo tendria una salida trivial — dejar de estar seguro y no pagar
residuo — y el entrenamiento convergeria a un modelo que no predice nada.

    python tools/ab/test_repeat_consistency_loss.py
"""
import sys

import torch

sys.path.insert(0, "src")
from dust3r.losses import RepeatConsistencyLoss  # noqa: E402

FALLOS = []


def check(nombre, cond, extra=""):
    print(f"  {'ok  ' if cond else 'FALLA'}  {nombre}{'  ' + extra if extra else ''}")
    if not cond:
        FALLOS.append(nombre)


def escena(B=2, H=4, W=6, desvio=0.0, escala=1.0, semilla=0):
    """Dos vistas con la MISMA imagen; `desvio` separa las dos predicciones."""
    g = torch.Generator().manual_seed(semilla)
    img = torch.randn(B, 3, H, W, generator=g)
    valid = torch.ones(B, H, W, dtype=torch.bool)
    gts = [
        {"img": img, "valid_mask": valid},
        {"img": img.clone(), "valid_mask": valid.clone()},
    ]
    frio = (torch.randn(B, H, W, 3, generator=g) + 5.0) * escala
    caliente = frio + desvio * escala
    preds = [
        {"pts3d_in_self_view": frio, "conf_self": torch.ones(B, H, W)},
        {"pts3d_in_self_view": caliente, "conf_self": torch.ones(B, H, W)},
    ]
    return gts, preds


def main():
    L = RepeatConsistencyLoss()

    # 1) de acuerdo consigo mismo -> cero
    gts, preds = escena(desvio=0.0)
    p, d = L.compute_loss(gts, preds)
    check("predicciones identicas -> perdida 0", float(p) == 0.0, f"loss={float(p):.3e}")
    check("se declara activa", d["RepeatConsistencyLoss/active"] == 1.0)

    # 2) en desacuerdo -> positivo y monotono
    _, p1 = escena(desvio=0.1)
    _, p2 = escena(desvio=0.2)
    l1 = float(L.compute_loss(gts, p1)[0])
    l2 = float(L.compute_loss(gts, p2)[0])
    check("desacuerdo -> perdida > 0", l1 > 0, f"loss={l1:.4f}")
    check("mas desacuerdo -> mas perdida", l2 > l1, f"{l1:.4f} -> {l2:.4f}")

    # 3) invariante a la escala de la escena: el doble de escena y el doble de
    #    desvio tienen que dar la misma perdida, porque el defecto es relativo
    _, pa = escena(desvio=0.1, escala=1.0)
    _, pb = escena(desvio=0.1, escala=10.0)
    la, lb = float(L.compute_loss(gts, pa)[0]), float(L.compute_loss(gts, pb)[0])
    check("invariante a la escala de la escena", abs(la - lb) < 1e-5,
          f"{la:.6f} vs {lb:.6f}")

    # 4) LA IMPORTANTE: el gradiente no puede llegar a la confianza, o el
    #    modelo aprende a no estar seguro de nada en vez de a ser consistente
    gts, preds = escena(desvio=0.1)
    conf = preds[0]["conf_self"].clone().requires_grad_(True)
    preds[0]["conf_self"] = conf
    preds[0]["pts3d_in_self_view"] = preds[0]["pts3d_in_self_view"].requires_grad_(True)
    L.compute_loss(gts, preds)[0].backward()
    check("la confianza NO recibe gradiente (anti-trampa)", conf.grad is None)

    # 5) maestro-alumno: solo la pasada en frio aprende
    gts, preds = escena(desvio=0.1)
    frio = preds[0]["pts3d_in_self_view"].clone().requires_grad_(True)
    caliente = preds[1]["pts3d_in_self_view"].clone().requires_grad_(True)
    preds[0]["pts3d_in_self_view"], preds[1]["pts3d_in_self_view"] = frio, caliente
    L.compute_loss(gts, preds)[0].backward()
    check("la pasada en frio recibe gradiente", frio.grad is not None)
    check("el objetivo caliente va detached", caliente.grad is None)

    # 6) guarda: sin vista duplicada la perdida es cero y lo dice
    gts, preds = escena(desvio=0.1)
    gts[1]["img"] = torch.randn_like(gts[1]["img"])
    p, d = L.compute_loss(gts, preds)
    check("imagenes distintas -> perdida 0", float(p) == 0.0)
    check("y se declara inactiva", d["RepeatConsistencyLoss/active"] == 0.0)

    # 7) resoluble desde el string del criterio, como la usara el config
    from dust3r.losses import MultiLoss
    ns = {}
    import dust3r.losses as DL
    for k, v in vars(DL).items():
        if not k.startswith("_"):
            ns[k] = v
    obj = eval("0.5*RepeatConsistencyLoss(src=0, at=1)", {"__builtins__": {}}, ns)
    check("resoluble desde el string del criterio", isinstance(obj, MultiLoss),
          repr(obj))

    print()
    if FALLOS:
        print(f"VEREDICTO: {len(FALLOS)} control(es) fallan: {', '.join(FALLOS)}")
        return 1
    print("VEREDICTO: todos los controles pasan")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
