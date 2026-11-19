#!/usr/bin/env bash
# Lee un checkpoint y saca las tres medidas que deciden si va bien.
#
# Se juzga por `resid` y GHOST, NUNCA por el train loss: la leccion de stage3c
# (docs/attractor_mechanism.md seccion 4) es que el loss se mueve y no dice nada.
#
#   bash tools/ab/probe_ckpt.sh CKPT ETIQUETA [CFG_FILE]
#
# CFG_FILE por defecto .scratch/cfg3b_rope.txt (model string con enc_rope=True).
# Corre DENTRO del contenedor.
set -euo pipefail

CKPT="${1:?falta el checkpoint}"
LABEL="${2:?falta la etiqueta}"
CFG="${3:-/app/recurrentTransformer/LoopCUT3R/.scratch/cfg3b_rope.txt}"

W=/app/recurrentTransformer/LoopCUT3R
AB=$W/.scratch/ab
cd "$W"

[[ -f "$CKPT" ]] || { echo "no existe el checkpoint: $CKPT" >&2; exit 1; }
[[ -f "$CFG"  ]] || { echo "no existe el cfg: $CFG" >&2; exit 1; }

SEQS=$(ls -d "$AB"/ab_seqs/*/ | sed 's#/$##' | tr '\n' ' ')

echo "### extrayendo  $LABEL  <- $CKPT"
python tools/ab/extract_multi.py \
    --repo "$W" --model_path "$CKPT" --model_cfg "$(cat "$CFG")" \
    --out_dir "$AB/recon" --prefix "$LABEL" \
    --seqs examples/002 $SEQS 2>&1 | grep -E "^  ok|^  !!|prime_state|no_enc_rope"

echo
echo "### autoconsistencia (frame repetido 8 veces): lo que importa es resid@7"
python tools/ab/exp_repeat.py "$AB/recon" --seqs rep_002_01_x8 \
    --models cut3r loop3b ropecold "$LABEL" 2>&1 | grep -E "^  [a-z]|^       0 |^       7 "

echo
echo "### GHOST contra CUT3R: lo que importa es el multiplicador y cuantos pares sobreviven"
python tools/ab/exp_pairs.py "$AB/recon" --ref cut3r --cmp "$LABEL" 2>&1 | tail -3

echo
echo "### referencia a batir"
cat <<'REF'
                 conf@0   px@0     alpha@7   resid@7   cam_t@7   cam_rot@7
  loop3b          0.744  109735    1.1819    0.1141    0.0089     0.262
  ropecold        0.747  110163    1.1849    0.1212    0.0116     0.131
  rope_ep0        0.707  104307    1.2015    0.1512    0.0065     0.119
  rope_ep1        0.688  101417    1.2174    0.1532    0.0089     0.125
  prime           0.989  145826    1.0929    0.0410    0.0064     0.296
  rope_ep1_prime  0.945  139294    1.1027    0.0575    0.0115     0.119
  cons_ep0        0.678  100001    1.1940    0.1055    0.0062     0.097
  cons_ep2        0.695  102506    1.2010    0.1131    0.0038     0.122
  cut3r           0.995  146682    0.9993    0.0082    0.0023       --

  GHOST: declarar SIEMPRE contra quien y sobre cuantos pares. No es un numero
  suelto, y esta tabla daba antes 3.57x y 3.07x juntos sin decir que son
  referencias distintas.

    vs cut3r (frio)      aislados  pares      vs cut3rbi (calentado)   aislados
      loop3b               3.57x    (10)        prime                    3.07x
      ropecold             3.39x    (10)        rope_ep1_prime           3.20x
      rope_ep1             3.63x    (10)        loop3bbi   (control)     3.13x
      prime                2.77x    (11)
      rope_ep1_prime       2.88x    (11)
      cons_ep0             3.80x    (10)
      cons_ep2             3.62x    (10)

  La JUSTA para un modelo calentado es vs cut3rbi: el burn-in tambien ayuda a
  CUT3R, asi que medirse contra un CUT3R frio infla el resultado (seccion 3b).
  Y la mediana va sobre los pares que AMBOS resuelven, 10 u 11 segun el modelo:
  comparar 3.57x con 2.77x mezcla dos poblaciones.

  Y `resid` tampoco se compara solo: un modelo menos confiado lo mide sobre
  menos pixeles y sale mejor sin haber mejorado. Si `resid` baja y `px@0`
  tambien, correr tools/ab/resid_comun.py antes de creerselo (seccion 1d).
REF
