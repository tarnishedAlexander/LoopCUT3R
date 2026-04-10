# CUT3R Encoder Fine-Tune Runbook

This runbook is a practical execution path for fine-tuning `CUT3R_encoder` on local data.
It assumes you want fast validation first, then controlled scale-up.

## 1) Environment and Repo Sanity

From repo root (`CUT3R_encoder`):

```bash
python -V
pip install -r requirements.txt
```

Compile CroCo RoPE kernels if needed:

```bash
cd src/croco/models/curope
python setup.py build_ext --inplace
cd ../../../../
```

Quick import sanity:

```bash
python - <<'PY'
from add_ckpt_path import add_path_to_dust3r
add_path_to_dust3r('src/cut3r_512_dpt_4_64.pth')
from src.dust3r.model import ARCroco3DStereo
print('OK')
PY
```

## 2) Data Prerequisites

### Expected processed layout (MVS_Synth example)

```text
processed_mvs_synth/
  0000/
    rgb/000000.jpg
    depth/000000.npy
    cam/000000.npz   # keys: intrinsics, pose
```

Preprocess raw MVS-Synth style data:

```bash
python datasets_preprocess/preprocess_mvs_synth.py \
  --root_dir /path/to/raw/GTAV_720 \
  --out_dir /path/to/processed_mvs_synth \
  --num_workers 8
```

Set dataset root env used by local configs:

```bash
export CUT3R_DATA_ROOT=/absolute/path/to/dataset
```

## 3) Pick First Training Mode

Use this order:

1. **Encoder-only quick validation** (`config/encoder_only_mydata.yaml`)
2. **Small recurrent decoder smoke** (`config/decoder_recurrent_k43_smoke.yaml`)
3. **Task-specific branch** (`dualdpt_ray_aware` or baseline stage lineage)

Reason:

1. Encoder-only catches data contract issues fast.
2. Smoke decoder catches state/TBPTL bugs before expensive runs.
3. Full branches become safer once first two pass.

## 4) Step-by-Step Execution

### Step A: Inference smoke check

```bash
python demo.py \
  --model_path src/cut3r_512_dpt_4_64.pth \
  --seq_path examples/001 \
  --size 512 \
  --output_dir demo_tmp
```

Pass criteria:

1. No import/path errors.
2. Depth/conf/color/camera outputs appear in `demo_tmp`.

### Step B: Encoder-only local training

```bash
cd src
accelerate launch train.py --config-name encoder_only_mydata
```

Pass criteria:

1. Training starts without key/dtype errors.
2. Checkpoint folder and `log.txt` created.
3. Loss decreases over first epochs (trend, not absolute value).

### Step C: Decoder recurrent smoke

```bash
cd src
accelerate launch train.py --config-name decoder_recurrent_k43_smoke
```

Pass criteria:

1. TBPTL path runs without backward graph errors.
2. State update/reset path works through multiple views.
3. Checkpoint save/restore works.

### Step D: Chosen full fine-tune

Examples:

```bash
# Variable-view DPT fine-tune lineage
cd src
accelerate launch train.py --config-name dpt_512_vary_4_64

# Or local ray-aware DualDPT branch
accelerate launch train.py --config-name dualdpt_ray_aware
```

## 5) Evaluation Path

Start with one pipeline first (`mv_recon`) before running all.

```bash
bash eval/mv_recon/run.sh
```

Then run others:

```bash
bash eval/relpose/run.sh
bash eval/video_depth/run.sh
bash eval/monodepth/run.sh
```

## 6) Debug Checklist (Most Common)

1. **Wrong code imported**:
   - Check `add_ckpt_path.py` behavior and current working directory.
2. **Dataset key mismatch**:
   - Confirm view dict has `img`, `ray_map`, `img_mask`, `ray_mask`, `camera_pose`, `valid_mask`.
3. **OOM or unstable gradients**:
   - Lower `batch_size`, increase `accum_iter`, reduce `num_views`.
4. **TBPTL issues**:
   - Temporarily disable `use_tbptl` and `dec_use_tbptl` to isolate.
5. **Path portability issues**:
   - Replace absolute `pretrained` paths with env-based or relative paths.

## 7) Config Knobs That Matter Most

Primary training knobs:

1. `freeze`: `none`, `encoder`, `decoder`.
2. `num_views`: memory + temporal context.
3. `resolution` and patch embed class (`PatchEmbedDust3R` vs `ManyAR_PatchEmbed`).
4. `gradient_checkpointing` and TBPTL flags.
5. Loss composition (`ConfLoss + Regr3DPose + RGBLoss + optional RayConsistencyLoss`).

## 8) Minimal Validation Ladder

Do not skip this order:

1. Demo inference on known example.
2. 1 short epoch encoder-only.
3. 1 short epoch decoder smoke.
4. 1 eval subset.
5. Then long multi-epoch run.

## 9) Output Artifacts to Track

Training:

1. `checkpoints/<exp_name>/checkpoint-last.pth`
2. `checkpoints/<exp_name>/checkpoint-final.pth`
3. `checkpoints/<exp_name>/logs/log.txt`

Evaluation:

1. `eval_results/mv_recon/.../logs_all.txt`
2. `eval_results/relpose/...`
3. `eval_results/video_depth/.../result_scale.json`
4. `eval_results/monodepth/.../metric.json`
