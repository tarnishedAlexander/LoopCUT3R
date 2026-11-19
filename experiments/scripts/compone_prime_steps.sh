set -uo pipefail
W=/app/recurrentTransformer/LoopCUT3R; AB=$W/.scratch/ab
cd $W
python tools/ab/build_swap.py --enc loop --dec loop --out $AB/swap_LL.pth 2>&1 | grep escrito
# se suman priming (seccion 3b) y 6 pasos (seccion 5c)? Atacan cosas distintas:
# el estado inicial y el numero de iteraciones, y los dos son gratis.
for N in 5 6 8; do
  python tools/ab/extract_multi.py --repo $W --model_path $AB/swap_LL.pth \
    --model_cfg "$(cat $W/.scratch/cfg_es$N.txt)" --prime_state \
    --out_dir $AB/recon --prefix p$N --seqs $AB/ab_seqs/rep_002_01_x8 \
    $AB/ab_seqs/rep_001_01_x8 2>&1 | grep -cE "^  ok" | sed "s/^/  p$N ok: /"
done
rm -f $AB/swap_LL.pth
echo "=== ESCENA 002: con y sin priming, a 5/6/8 pasos ==="
python tools/ab/exp_repeat.py $AB/recon --seqs rep_002_01_x8 \
  --models cut3r es5 es6 es8 p5 p6 p8 2>&1 | grep -E "^  [a-z]|^       0 |^       7 "
echo "=== ESCENA 001 ==="
python tools/ab/exp_repeat.py $AB/recon --seqs rep_001_01_x8 \
  --models es5 es6 es8 p5 p6 p8 2>&1 | grep -E "^  [a-z]|^       0 |^       7 "
echo "=== resid sobre los MISMOS pixeles (002) ==="
python tools/ab/resid_comun.py es8 es6 p8 p6 cut3r
echo "=== LISTO ==="
