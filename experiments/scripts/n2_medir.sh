set -uo pipefail
W=/app/recurrentTransformer/LoopCUT3R
AB=$W/.scratch/ab
cd $W
SEQS=$(ls -d $AB/ab_seqs/*/ | sed 's#/$##' | tr '\n' ' ')
for C in EE EL LE LL; do
  echo "### extrayendo swap$C"
  rm -f $AB/recon/swap${C}_*.npz
  python tools/ab/extract_multi.py --repo $W \
    --model_path $AB/swap_$C.pth --model_cfg "$(cat $W/.scratch/cfg_swap_$C.txt)" \
    --out_dir $AB/recon --prefix swap$C \
    --seqs examples/002 $SEQS 2>&1 | grep -cE "^  ok" | sed 's/^/  secuencias ok: /'
  rm -f $AB/swap_$C.pth
  df -h /home 2>/dev/null | tail -1 || true
done
echo "=== AUTOCONSISTENCIA (frame repetido 8 veces) ==="
python tools/ab/exp_repeat.py $AB/recon --seqs rep_002_01_x8 \
  --models cut3r swapEE swapEL swapLE swapLL 2>&1 | grep -E "^  [a-z]|^       0 |^       7 "
echo "=== GHOST contra cut3r ==="
for C in EE EL LE LL; do
  python tools/ab/exp_pairs.py $AB/recon --ref cut3r --cmp swap$C 2>&1 | grep "GHOST"
done
echo "=== LISTO ==="
