import sys, os, torch
sys.path.insert(0, "/app/recurrentTransformer/LoopCUT3R")
os.chdir("/app/recurrentTransformer/LoopCUT3R")
from add_ckpt_path import add_path_to_dust3r
CK = "/app/recurrentTransformer/LoopCUT3R/checkpoints/recurrent_stage3b/checkpoints/checkpoint-final.pth"
CFG = open("/app/recurrentTransformer/LoopCUT3R/.scratch/cfg3b.txt").read().strip()
add_path_to_dust3r(CK)
from src.dust3r.model import ARCroco3DStereo
m = ARCroco3DStereo.from_pretrained(CK, model_cfg_str=CFG).eval()

print("=== geometria real ===")
print(f"  encoder: {len(m.enc_blocks)} bloques x num_steps={m.num_steps} = "
      f"{len(m.enc_blocks)*m.num_steps} aplicaciones   (CUT3R: 24 bloques x 1 = 24)")
print(f"  decoder img  : {m.dec_num_blocks_img} bloques distintos, "
      f"{m.dec_num_steps_img} aplicaciones -> ciclo "
      f"{[i % m.dec_num_blocks_img for i in range(m.dec_num_steps_img)]}   (CUT3R: 12 y 12)")
print(f"  decoder state: {m.dec_num_blocks_state} bloques distintos, "
      f"{m.dec_num_steps_state} aplicaciones")
print(f"  enc_pos_embed = {m.enc_pos_embed}      rope = {type(m.rope).__name__}")
print()
torch.manual_seed(0)
img = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    x, p = m.patch_embed(img)
    N = x.shape[1]
    def enc(xx, pp):
        y = xx
        for _ in range(m.num_steps):
            for blk in m.enc_blocks:
                y = blk(y, pp)
        return m.enc_norm(y)
    base = enc(x, p)
    perm = torch.randperm(N)
    shuf = enc(x[:, perm], p[:, perm])          # parches barajados, posiciones con ellos
    wrong = enc(x, p[:, perm])                  # mismos parches, posiciones MENTIROSAS
print(f"tokens N={N}")
d1 = (shuf - base[:, perm]).abs().max().item()
d2 = (wrong - base).abs().max().item()
print(f"  max|enc(parches barajados) - enc(original)[barajado]| = {d1:.3e}")
print(f"  max|enc(posiciones falsas) - enc(posiciones reales)|   = {d2:.3e}")
print()
print("VEREDICTO:", "el encoder entrenado es una BOLSA DE PARCHES: barajar no cambia nada "
      "y mentir en las posiciones no cambia nada." if d1 < 1e-4 and d2 == 0.0
      else "el encoder usa la posicion")
