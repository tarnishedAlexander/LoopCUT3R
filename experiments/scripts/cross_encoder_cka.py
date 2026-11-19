#!/usr/bin/env python3
"""Compara los encoders de DOS modelos capa por capa, con CKA lineal.

Responde: ¿cuanta informacion pierde el encoder recurrente de LoopCUT3R
(3 bloques x 8 pasos = 24 aplicaciones) respecto a los 24 bloques DISTINTOS de
CUT3R, sobre las mismas imagenes y los mismos parches?

Lo que importa de verdad es la ULTIMA capa: es la que alimenta al decoder.

Por que no reutilizo `_linear_cka` de src/tools/layer_similarity_encoder.py:
esa version construye matrices de Gram n x n (K = X @ X.T).  Con n = 5376
muestras eso es 115 MB por matriz y hacen falta dos por par; para una matriz
24 x 24 son 576 pares.  La forma en espacio de features da EXACTAMENTE el
mismo numero para CKA lineal y trabaja con matrices d x d (1024 x 1024):

    CKA(X, Y) = ||Y^T X||_F^2 / ( ||X^T X||_F * ||Y^T Y||_F )

con X e Y centradas por columna.

    python tools/ab/cross_encoder_cka.py A.npz B.npz --labels CUT3R LoopCUT3R
"""
import argparse

import numpy as np
import torch


def centrar(X):
    return X - X.mean(dim=0, keepdim=True)


def cka(X, Y):
    """CKA lineal en espacio de features. X: [n, d1], Y: [n, d2], ya centradas."""
    xty = X.T @ Y                       # [d1, d2]
    hsic = (xty * xty).sum()
    nx = X.T @ X
    ny = Y.T @ Y
    den = torch.sqrt((nx * nx).sum()) * torch.sqrt((ny * ny).sum())
    return (hsic / den.clamp_min(1e-12)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="npz del modelo A (referencia, p.ej. CUT3R)")
    ap.add_argument("b", help="npz del modelo B (p.ej. LoopCUT3R)")
    ap.add_argument("--labels", nargs=2, default=["A", "B"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    LA, LB = args.labels

    da, db = np.load(args.a, allow_pickle=True), np.load(args.b, allow_pickle=True)
    A, B = da["banco"], db["banco"]
    print(f"{LA:<12} {A.shape}  enc_blocks={da['n_blocks']} num_steps={da['n_steps']}")
    print(f"{LB:<12} {B.shape}  enc_blocks={db['n_blocks']} num_steps={db['n_steps']}")
    if A.shape[1] != B.shape[1]:
        raise SystemExit("distinto numero de muestras: los volcados no son comparables")
    L = min(A.shape[0], B.shape[0])

    dev = args.device if torch.cuda.is_available() else "cpu"
    Xa = [centrar(torch.from_numpy(A[i]).to(dev)) for i in range(L)]
    Xb = [centrar(torch.from_numpy(B[i]).to(dev)) for i in range(L)]

    with torch.no_grad():
        cruz = np.zeros((L, L), dtype=np.float32)
        for i in range(L):
            for j in range(L):
                cruz[i, j] = cka(Xb[i], Xa[j])          # fila = B, columna = A
        intra_a = np.array([cka(Xa[i], Xa[i + 1]) for i in range(L - 1)])
        intra_b = np.array([cka(Xb[i], Xb[i + 1]) for i in range(L - 1)])

    diag = np.diag(cruz)

    print()
    print("=" * 74)
    print(f" CKA capa a capa: {LB} (fila) contra {LA} (columna)")
    print("=" * 74)
    print(f"{'capa':>5}  {'diagonal':>9}  {'mejor':>6}  {'con capa':>9}  "
          f"{'CKA(i,i+1) propio':>18}")
    print(f"{'':>5}  {'i vs i':>9}  {'CKA':>6}  {'de '+LA:>9}")
    for i in range(L):
        j = int(np.argmax(cruz[i]))
        prop = f"{intra_b[i]:.4f}" if i < L - 1 else "    --"
        marca = "  <-- ultima (alimenta al decoder)" if i == L - 1 else ""
        print(f"{i:>5}  {diag[i]:>9.4f}  {cruz[i, j]:>6.4f}  {j:>9}  "
              f"{prop:>18}{marca}")

    print()
    print("=" * 74)
    print(" RESUMEN")
    print("=" * 74)
    print(f"  CKA de la ULTIMA capa (lo que ve el decoder) : {diag[-1]:.4f}")
    print(f"  CKA diagonal, media                          : {diag.mean():.4f}")
    print(f"  CKA diagonal, minimo (capa {int(np.argmin(diag)):>2})              : {diag.min():.4f}")
    print(f"  CKA diagonal, maximo (capa {int(np.argmax(diag)):>2})              : {diag.max():.4f}")
    print()
    print("  Escala de referencia -- capas CONSECUTIVAS del mismo modelo")
    print("  (si el cruzado se parece a esto, los encoders son casi el mismo):")
    print(f"    {LA:<12} CKA(i, i+1) media = {intra_a.mean():.4f}  "
          f"min = {intra_a.min():.4f}")
    print(f"    {LB:<12} CKA(i, i+1) media = {intra_b.mean():.4f}  "
          f"min = {intra_b.min():.4f}")
    print()
    alineado = int((np.argmax(cruz, axis=1) == np.arange(L)).sum())
    print(f"  Capas cuyo mejor emparejamiento es la MISMA profundidad: "
          f"{alineado}/{L}")
    desp = np.argmax(cruz, axis=1) - np.arange(L)
    print(f"  Desplazamiento mediano del mejor emparejamiento: {int(np.median(desp)):+d} capas")


if __name__ == "__main__":
    main()
