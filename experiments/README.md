# Experiments

Configs and scripts of experiments that are **not** part of the released model.
They are kept so the thesis can be traced back to code; they are not maintained.
Most scripts here were run inside the training container and still contain its
absolute paths (`/app/...`).  To run one of the configs:

```bash
cd src
python train.py --config-path ../experiments/configs --config-name NAME
```

The released model is the one trained by `config/` (phases 1-3).  Every
experiment below was judged by the pose and video-depth benchmarks
(`tools/eval/run_benchmarks.sh`), not by the training loss, and none of them
beat the phase 3 checkpoint (`recurrent_stage3b`).

## Phase 4: long sequences

`recurrent_stage4.yaml`, `recurrent_stage4_32v.yaml`, `recurrent_stage4_64v.yaml`,
`recurrent_stage4b_32v.yaml`, `recurrent_stage4_vary.yaml`

CUT3R's last stage trains on long sequences (32 views, then 4-64).  Three
attempts (fixed 32 views; 32 views continued; variable 4-64 views) all made the
reconstruction worse than phase 3, the more so the longer they trained, while
the per-view training loss kept improving.  Phase 4 was discarded.

## Positional encoding in the encoder

`recurrent_stage3d_encrope.yaml`, `recurrent_stage3f_encrope_real.yaml`,
flag `enc_rope` in `ARCroco3DStereoConfig`; scripts `remap_enc_attn.py`,
`check_enc_rope.py`, `test_enc_pos_invariance.py`, `test_block_equivalence.py`

The recurrent blocks inherited from Raptor ignore token positions, so the
encoder treats each image as a set of patches (shuffling the patches permutes
the output exactly).  `enc_rope=True` gives the blocks CUT3R's RoPE attention;
the weights map one to one.  A short adaptation (stage3d, 8 h) did not help.
A full epoch (stage3f, 33 h) reduced the self-consistency drift by 41 % but
made the pose benchmark worse than phase 3.  Off by default.

## Self-consistency of the recurrent state

`recurrent_stage3c_*.yaml`, `recurrent_stage3e_consistency.yaml`; flags
`dec_t_scale`, `prime_state`, `duplicate_first_view`; loss
`RepeatConsistencyLoss`; scripts `exp_repeat.py`, `extract_multi.py`,
`resid_comun.py`, `barrido_steps.sh`

Feeding LoopCUT3R the same frame several times moves its points by 6-18 % of
the median depth (CUT3R: 0.6-2 %).  Interventions tried against this drift:

- `dec_t_scale` (stage3c): per-step FiLM conditioning of the shared decoder
  blocks, zero-initialised so training starts exactly from phase 3.
- `RepeatConsistencyLoss` with `duplicate_first_view` (stage3e): an explicit
  loss on a duplicated view.  It lowered its own objective but not the drift
  over 8 repetitions.
- `prime_state`: one warm-up decoder step on the first frame, no training.
  It halved the drift and made the rotation error worse on all three pose
  datasets.
- Fewer encoder passes at inference: changes of a few percent whose sign
  depends on the dataset.

Lower drift did not translate into better pose or depth; on these runs the two
moved in opposite directions.  All flags are off by default.

## Earlier pipeline

`mvp_stage*.yaml`, `encoder_only_mydata*.yaml`, `encoder_only_multi_dataset.yaml`,
`decoder_*.yaml`, `production_raptor_recurrent_cut3r.yaml`,
`raptor_full_finetune.yaml`, `recurrent_stage1_*h*.yaml`, `recurrent_stage1_r2.yaml`,
`recurrent_stage1_smoke.yaml`, `recurrent_dual_variant_original_cut3r.yaml`

The minimum viable product (a dual DPT head with its own losses, later removed)
and the first phase 1 runs with other budgets and learning rates.
`recurrent_stage1_r3` in `config/` is the run that was kept.

`stage1-4.yaml`, `dpt_512_vary_4_64.yaml` and `linear_224_fixed_16.yaml` are
CUT3R's own training configs, edited for the training machine.  LoopCUT3R's
`train.py` would build a recurrent encoder from them, not CUT3R.

## Evaluation pitfalls

`run_baseline_eval.sh`, `benchmark_cut3r_baseline.py`, `eval_compare.sh`,
`relpose_table.py`, `benchmark_table.py`

`eval_compare.sh` and `benchmark_cut3r_baseline.py` evaluated CUT3R's checkpoint
from this source tree.  `add_ckpt_path.py` always puts the local `src/` on the
path, so CUT3R's weights were loaded into LoopCUT3R's architecture with
`strict=False`: all 24 CUT3R encoder blocks were dropped and the encoder ran at
random initialisation, which made LoopCUT3R look better than CUT3R.  Every
number from those scripts is invalid.  `tools/eval/run_benchmarks.sh` runs each
model in its own tree and `tools/eval/check_weights.py` checks the pairing.

The old relpose tables averaged what each run had scored, and on Sintel CUT3R
had run full sequences against 30 frames for LoopCUT3R.
`tools/eval/pose_table.py` re-scores all labels on the same frames.

## Infrastructure

`infra/`: the launch scripts used on the training machine (NVMe staging of the
datasets, HDF5 cache building, dataset checks).  They encode that machine's
disk layout.

## Results

`results/efficiency_loop_rope.json`: efficiency of the `enc_rope` variant.

## Attribution: initialisation versus depth recurrence

`attr_control.yaml`, `attr_loop_cut3rdec.yaml`, `attr_hybrid_full.yaml`,
`scripts/init_from_cut3r.py`, `scripts/check_init.py`

Phase 1 loaded only the encoder from Raptor (`load_only_encoder: True`), so
LoopCUT3R's decoder, heads, pose memory and state tokens started from random
weights: their cosine similarity to CUT3R's is about 0, while CUT3R's came
from DUSt3R/CroCo pretraining and ~185 epochs.  The gap to CUT3R therefore mixes
initialisation and training budget with depth recurrence.  Three arms with the
same data, schedule and phase-3b encoder separate them:

| Arm | Decoder | Initialised from |
|---|---|---|
| `attr_control` | 3 shared x 4 | phase 3b (as released) |
| `attr_loop_cut3rdec` | 3 shared x 4 | CUT3R (decoder blocks 0, 4, 8), heads and pose memory from CUT3R |
| `attr_hybrid_full` | 12 independent, as CUT3R | CUT3R |

```bash
python experiments/scripts/init_from_cut3r.py --loop PHASE3B.pth --loop_cfg config/recurrent_stage3b.yaml \
    --cut3r checkpoints/cut3r_512_dpt_4_64.pth --decoder full   --out checkpoints/init_full.pth
python experiments/scripts/init_from_cut3r.py --loop PHASE3B.pth --loop_cfg config/recurrent_stage3b.yaml \
    --cut3r checkpoints/cut3r_512_dpt_4_64.pth --decoder shared --out checkpoints/init_shared.pth
python experiments/scripts/check_init.py checkpoints/init_full.pth checkpoints/init_shared.pth
```

Run from the repository root.  Datasets packed into a single `.h5` are found
through `LOOPCUT3R_H5_DIRS` (each folder and its direct subfolders are searched);
without it, a root whose images live only in the `.h5` fails with
"Could not load image".

```bash
LOOPCUT3R_H5_DIRS=/data:/data/fast HYDRA_FULL_ERROR=1 accelerate launch --num_processes 1 \
    --mixed_precision bf16 src/train.py --config-path=$PWD/experiments/configs --config-name attr_control
```

`check_init.py` loads strictly: `ARCroco3DStereo.load_state_dict` otherwise
falls back silently to dropping the decoder keys when any tensor mismatches.

### Result (3 epochs, 10 % of the phase-3b mixture each, RTX 5090)

Camera pose, `pose_table.py` with every label cut to the same frames (ATE, RPE
translation in metres, RPE rotation in degrees):

| | ScanNet ATE | RPE t | RPE r | TUM ATE | RPE t | RPE r | Sintel ATE | RPE t | RPE r |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CUT3R | 0.169 | 0.205 | 3.02 | 0.145 | 0.186 | 7.84 | 0.175 | 0.071 | 0.65 |
| LoopCUT3R (released) | 0.890 | 0.801 | 54.81 | 0.780 | 0.572 | 42.69 | 0.237 | 0.112 | 2.89 |
| `attr_control` | 0.941 | 0.778 | 55.76 | 0.802 | 0.577 | 39.66 | 0.249 | 0.110 | 2.28 |
| `attr_loop_cut3rdec` | 0.977 | 0.946 | 45.01 | 0.862 | 0.640 | 30.87 | 0.249 | 0.157 | 0.94 |
| `attr_hybrid_full` | 0.914 | 0.855 | 55.72 | 0.537 | 0.608 | 29.19 | 0.160 | 0.088 | 2.00 |

Video depth (scale & shift), Abs Rel: Sintel 0.551 / 0.620 / 0.545 and Bonn
0.116 / 0.270 / 0.206 for control / loop_cut3rdec / hybrid_full (CUT3R 0.456 /
0.053, released LoopCUT3R 0.581 / 0.110).

Reading: initialising the shared decoder from CUT3R does not help, and giving
the recurrent encoder CUT3R's whole 12-block decoder, heads and pose memory
helps on Sintel (ATE below CUT3R) and TUM but leaves ScanNet unchanged, with a
~55 degree per-step rotation error in every arm (the encoder-RoPE runs too).
The decoder is therefore not where the ScanNet/TUM pose gap comes from; the
recurrent encoder's features are.
