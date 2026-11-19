"""Reconstruye checkpoints/loop3b_free_stack.pth: las tres palancas gratis.

El fichero se borro para hacer sitio; esto lo rehace en un minuto, identico. Se
verifico en su dia que reproduce `p6` bit a bit (max|dif| = 0.000e+00 en las dos
escenas), asi que lo que mide el benchmark es exactamente lo que medio la sonda.

OJO: las tres palancas NO mejoran el benchmark; ver docs/attractor_mechanism.md
seccion 5d. Esto se conserva para poder reproducir esas cifras, no porque sea
una configuracion recomendada.

    python tools/ab/mk_freestack.py

El eval construye el modelo desde `ck["args"].model`, asi que la unica forma de
que los benchmarks midan la configuracion buena sin tocar los scripts de eval es
grabarla ahi:

    enc_rope=True      la posicion del encoder: parte en dos cam_rot, gratis (1c)
    num_steps=6        en vez de 8: domina en las 4 metricas y 2 escenas (5c)
    prime_state=True   un paso de calentamiento en i=0 (3b)

Los PESOS no se tocan: salen de loop3b_encrope_init.pth, que es stage3b con la
atencion del encoder remapeada al layout con rope (un renombrado exacto,
tools/ab/remap_enc_attn.py). Los ARGS salen del checkpoint de stage3b, porque
loop3b_encrope_init.pth es un state dict desnudo y no los trae.
"""
import os, re, torch

PESOS = "checkpoints/loop3b_encrope_init.pth"
# -final es un state dict DESNUDO (575 tensores, sin "args"); -best si los trae
ARGS_DE = "checkpoints/recurrent_stage3b/checkpoints/checkpoint-best.pth"
SALIDA = "checkpoints/loop3b_free_stack.pth"

sd = torch.load(PESOS, map_location="cpu", weights_only=False)
sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
base = torch.load(ARGS_DE, map_location="cpu", weights_only=False)
assert "args" in base, "sin args embebidos en " + ARGS_DE
print(f"  pesos: {len(sd)} tensores de {PESOS}")

s = base["args"].model
print(f"\n  ANTES:   {s}\n")

def set_kv(s, k, v):
    pat = re.compile(r"(?<![A-Za-z_])" + re.escape(k) + r"=[^,)]+")
    hits = pat.findall(s)
    if len(hits) == 1:
        return pat.sub(f"{k}={v}", s)
    if not hits:
        assert s.endswith("))"), s[-20:]
        return s[:-2] + f", {k}={v}))"
    raise AssertionError((k, hits))

# `use_tbptl` no se toca: en inferencia `self.training` es False y la rama de
# TBPTL no corre. Y la verificacion va con el MISMO regex que la sustitucion,
# no con un conteo de subcadena: "use_tbptl=False" aparece dos veces porque
# "dec_use_tbptl=False" lo contiene.
for k, v in (("num_steps", 6), ("enc_rope", "True"), ("prime_state", "True")):
    s = set_kv(s, k, v)
for k, v in (("num_steps", "6"), ("enc_rope", "True"), ("prime_state", "True")):
    pat = re.compile(r"(?<![A-Za-z_])" + k + r"=" + re.escape(v) + r"(?![A-Za-z0-9_])")
    assert len(pat.findall(s)) == 1, (k, pat.findall(s))
print(f"  DESPUES: {s}\n")

base["args"].model = s
base["model"] = sd
for k in ("optimizer", "scaler"):
    base.pop(k, None)                 # no hacen falta para inferencia, y pesan
torch.save(base, SALIDA)
print(f"  escrito {SALIDA}  ({os.path.getsize(SALIDA)/1e9:.2f} GB)")
