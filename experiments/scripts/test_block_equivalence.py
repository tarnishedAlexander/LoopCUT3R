"""Control atomico: `Block` de CUT3R vs `RecurrentBlock(rope, gamma=1)`.

Con los MISMOS pesos renombrados tienen que dar el MISMO numero. Es la premisa
de `tools/ab/build_swap.py` y del experimento N2: si falla, ninguna celda del
swap 2x2 significa nada.

Por que son equivalentes: con `enc_rope=True` el `RecurrentBlock` construye la
misma clase `Attention` que `Block`, y con `gamma=1` el LayerScale es la
identidad, asi que su forward queda `x + attn(norm1(x), xpos)` y
`x + mlp(norm2(x))`, identico a `Block.forward` con `drop_path=Identity`.
El renombrado es `mlp.fc1 -> mlp.0`, `mlp.fc2 -> mlp.2`; la atencion ya coincide.

    python tools/ab/test_block_equivalence.py

Ver docs/attractor_mechanism.md seccion 6b.
"""
import sys, torch
sys.path.insert(0, "src")
sys.path.insert(0, "src/croco")
from models.blocks import Block, RecurrentBlock

dim, heads = 1024, 16
torch.manual_seed(0)

# el mismo objeto rope que usa el modelo real
from models.pos_embed import RoPE2D
rope = RoPE2D(freq=100.0)

a = Block(dim, heads, mlp_ratio=4.0, qkv_bias=True, rope=rope).eval()
b = RecurrentBlock(dim=dim, num_heads=heads, mlp_ratio=4.0, init_values=1.0, rope=rope).eval()

sa = a.state_dict()
mapa = {"attn.qkv.weight": "attn.qkv.weight", "attn.qkv.bias": "attn.qkv.bias",
        "attn.proj.weight": "attn.proj.weight", "attn.proj.bias": "attn.proj.bias",
        "mlp.fc1.weight": "mlp.0.weight", "mlp.fc1.bias": "mlp.0.bias",
        "mlp.fc2.weight": "mlp.2.weight", "mlp.fc2.bias": "mlp.2.bias",
        "norm1.weight": "norm1.weight", "norm1.bias": "norm1.bias",
        "norm2.weight": "norm2.weight", "norm2.bias": "norm2.bias"}
sb = b.state_dict()
nuevo = {}
for k, v in sa.items():
    assert k in mapa, f"clave de Block sin mapear: {k}"
    nuevo[mapa[k]] = v
nuevo["ls1.gamma"] = torch.ones(dim)
nuevo["ls2.gamma"] = torch.ones(dim)
falta = sorted(set(sb) - set(nuevo)); sobra = sorted(set(nuevo) - set(sb))
print(f"  claves: faltan {falta} sobran {sobra}")
b.load_state_dict(nuevo, strict=True)

x = torch.randn(2, 256, dim)
pos = torch.stack(torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij"), -1)
pos = pos.reshape(1, 256, 2).repeat(2, 1, 1)
with torch.no_grad():
    ya, yb = a(x, pos), b(x, pos)
d = (ya.double() - yb.double()).abs()
print(f"  salida: max|dif| = {d.max():.3e}   media|dif| = {d.mean():.3e}   escala {ya.abs().mean():.3e}")
print("VEREDICTO:", "EQUIVALENTES" if d.max() < 1e-4 else "NO EQUIVALENTES")
