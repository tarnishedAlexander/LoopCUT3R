import sys, torch
sys.path.insert(0, "/app/recurrentTransformer/LoopCUT3R/src")
sys.path.insert(0, "/app/recurrentTransformer/LoopCUT3R/src/croco")
torch.manual_seed(0)
from models.blocks import RecurrentBlock, Block
from models.pos_embed import RoPE2D

N, D, H = 64, 1024, 16
x = torch.randn(1, N, D)
# dos rejillas de posiciones DISTINTAS para los mismos tokens
pos1 = torch.stack(torch.meshgrid(torch.arange(8), torch.arange(8), indexing="ij"), -1).reshape(1, N, 2)
pos2 = pos1[:, torch.randperm(N)]          # las mismas posiciones, asignadas a otros tokens

print("=== 1) RecurrentBlock (encoder de LoopCUT3R) ===")
blk = RecurrentBlock(dim=D, num_heads=H, init_values=1e-5).eval()
with torch.no_grad():
    o1, o2 = blk(x, pos1), blk(x, pos2)
print(f"   max|salida(pos1) - salida(pos2)| = {(o1-o2).abs().max():.3e}")
print(f"   -> {'IGNORA la posicion' if (o1-o2).abs().max() == 0 else 'usa la posicion'}")

perm = torch.randperm(N)
with torch.no_grad():
    o3 = blk(x[:, perm], pos1)
print(f"   max|salida(tokens barajados) - salida(original)[barajada]| = {(o3 - o1[:, perm]).abs().max():.3e}")
print(f"   -> {'EQUIVARIANTE a permutacion (bolsa de parches)' if (o3-o1[:,perm]).abs().max() < 1e-5 else 'no equivariante'}")

print()
print("=== 2) Block con RoPE (encoder de CUT3R) ===")
rope = RoPE2D(freq=100.0)
blkc = Block(dim=D, num_heads=H, qkv_bias=True, rope=rope).eval()
with torch.no_grad():
    c1, c2 = blkc(x, pos1), blkc(x, pos2)
print(f"   max|salida(pos1) - salida(pos2)| = {(c1-c2).abs().max():.3e}")
print(f"   -> {'IGNORA la posicion' if (c1-c2).abs().max() == 0 else 'USA la posicion'}")
with torch.no_grad():
    c3 = blkc(x[:, perm], pos1)
print(f"   max|salida(tokens barajados) - salida(original)[barajada]| = {(c3 - c1[:, perm]).abs().max():.3e}")
print(f"   -> {'equivariante' if (c3-c1[:,perm]).abs().max() < 1e-5 else 'NO equivariante (la posicion importa)'}")
