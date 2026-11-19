set -uo pipefail
W=/app/recurrentTransformer/LoopCUT3R; AB=$W/.scratch/ab
cd $W
python tools/ab/build_swap.py --enc loop --dec loop --out $AB/swap_LL.pth 2>&1 | grep escrito
REP=$AB/ab_seqs/rep_002_01_x8
for P in es1 es2 es3 es4 es5 es6 es7 es8 es9 es10 es12 es16 \
         ds1 ds2 ds3 ds4 ds5 ds6 ds8 ds10 ds12; do
  rm -f $AB/recon/${P}_rep_002_01_x8.npz
  python tools/ab/extract_multi.py --repo $W --model_path $AB/swap_LL.pth \
    --model_cfg "$(cat $W/.scratch/cfg_$P.txt)" \
    --out_dir $AB/recon --prefix $P --seqs $REP 2>&1 | grep -cE "^  ok" >/dev/null \
    && echo "  $P listo" || echo "  $P FALLO"
done
rm -f $AB/swap_LL.pth
echo "=== BARRIDO DEL ENCODER (decoder fijo en 4) ==="
python tools/ab/exp_repeat.py $AB/recon --seqs rep_002_01_x8 \
  --models es1 es2 es3 es4 es5 es6 es7 es8 es9 es10 es12 es16 2>&1 \
  | grep -E "^  [a-z]|^       7 "
echo "=== BARRIDO DEL DECODER (encoder fijo en 8) ==="
python tools/ab/exp_repeat.py $AB/recon --seqs rep_002_01_x8 \
  --models ds1 ds2 ds3 ds4 ds5 ds6 ds8 ds10 ds12 2>&1 \
  | grep -E "^  [a-z]|^       7 "
echo "=== LISTO ==="
