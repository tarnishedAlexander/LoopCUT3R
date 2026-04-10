# CUT3R Encoder Architecture Map

This document is a training-first architecture map for `CUT3R_encoder`.
It is designed to answer:

1. What runs in what order?
2. Which tensors and keys must exist at each stage?
3. Which knobs matter most for fine-tuning?

## 1) System Overview

`ARCroco3DStereo` is the core model (`src/dust3r/model.py`).

High-level flow:

1. Build per-view tokens from image and optional ray map.
2. Initialize persistent state tokens.
3. Recurrently decode each view while updating state and local memory.
4. Run downstream head (linear, DPT, or DualDPT) to produce 3D/pose/rgb/conf outputs.
5. Compute configured loss in training loop.

Primary symbols:

- `ARCroco3DStereoConfig` (`src/dust3r/model.py`)
- `ARCroco3DStereo` (`src/dust3r/model.py`)
- `_run_recurrent_encoder_blocks`
- `_assemble_view_tokens`
- `_decoder`
- `_forward_decoder_step`
- `_forward_impl`

## 2) Forward Call Graph (Training)

Entrypoint:

- `train.py: train()`
- `train_one_epoch()` chooses one of:
  - `loss_of_one_batch`
  - `loss_of_one_batch_encoder`
  - `loss_of_one_batch_tbptt`

Main path for standard training:

1. `loss_of_one_batch(batch, model, criterion, ...)`
2. `model(batch)`
3. `ARCroco3DStereo.forward()`
4. `_forward_impl(views)`
5. `_encode_views()` -> `_assemble_view_tokens()`
6. `_init_state()`
7. loop views:
   - `_forward_decoder_step(...)`
   - `_recurrent_rollout(...)`
   - `_decoder(...)`
   - `_downstream_head(...)`
8. return `ARCroco3DStereoOutput(ress=..., views=...)`
9. `criterion(batch, preds)`

## 3) Input Data Contract (Critical)

The canonical contract is produced by `BaseMultiViewDataset.__getitem__`.

Per-view keys used by model/loss path:

- `img`: tensor `[B, 3, H, W]`
- `ray_map`: tensor `[B, H, W, 6]`
- `true_shape`: tensor `[B, 2]` (height, width)
- `img_mask`: bool tensor `[B]`
- `ray_mask`: bool tensor `[B]`
- `camera_pose`: tensor `[B, 4, 4]` for gt; model predicts encoded pose in output
- `camera_intrinsics`: tensor `[B, 3, 3]`
- `depthmap`: tensor `[B, H, W]`
- `valid_mask`: bool tensor `[B, H, W]`
- `sky_mask`: bool tensor `[B, H, W]`
- `is_metric`: bool tensor or scalar flag
- `reset`: bool tensor `[B]`
- `update`: optional bool tensor `[B]`

Contract notes:

1. `img_mask` and `ray_mask` jointly control whether image/ray tokens are real or replaced with learned masked tokens.
2. `reset` hard-resets state/memory to sequence init values.
3. `update=False` keeps prior state for that sample even if a new view is seen.

## 4) Encoder Path

Functions:

- `_encode_image` and `_encode_ray_map`
- shared recurrent runner: `_run_recurrent_encoder_blocks`

Behavior:

1. Patch embedding (`PatchEmbedDust3R` or `ManyAR_PatchEmbed`) creates token grid + position.
2. Recurrent encoder blocks run for `num_steps`.
3. TBPTL may split no-grad warmup and grad steps when enabled.
4. Final norms produce encoded tokens.

TBPTL/checkpointing interaction:

- Per-block checkpointing is disabled when `use_tbptl` is active in recurrent loop.
- This avoids recurrent metadata mismatch in backward.

## 5) State + Decoder Path

State init:

1. `_encode_state` creates learnable register-token state.
2. `_init_state` projects state into decoder embedding space.

Decoder recurrence:

1. `_decoder` runs two intertwined rollouts:
   - `dec_blocks_state` for state update
   - `dec_blocks` for image-token update
2. Separate controls exist for blocks vs steps for each stream:
   - `dec_num_blocks_img`, `dec_num_steps_img`
   - `dec_num_blocks_state`, `dec_num_steps_state`
3. Optional decoder TBPTL (`dec_use_tbptl`) can truncate decoder graph.

Pose memory path (when `pose_head=True`):

1. `LocalMemory.inquire` retrieves pose feature from memory for view `i > 0`.
2. `LocalMemory.update_mem` writes back with current global image feature + output pose token.

State update logic in `_forward_decoder_step`:

1. Compute `update_mask = img_mask & update` (or `img_mask` if `update` absent).
2. Blend new and old state/memory with mask.
3. Apply `reset` mask to restore init state/memory.

## 6) Head Variants and Output Keys

Head factory: `src/dust3r/heads/__init__.py`.

### Linear head (`head_type="linear"`)

- Class: `LinearPts3dPose`.
- Produces standard CUT3R-style keys including:
  - `pts3d_in_self_view`
  - `pts3d_in_other_view`
  - `conf_self`
  - `conf`
  - `camera_pose` (if enabled)
  - `rgb` (if enabled)

### DPT head (`head_type="dpt"`)

- Class: `DPTPts3dPose`.
- Multi-scale head; computes self and cross outputs, plus pose/rgb when enabled.

### DualDPT head (`head_type="dualdpt"`)

- Class: `DualDPTHead`.
- Factorizes geometry into depth and ray direction:
  - `unit_rays` normalized
  - `pts3d = depth * unit_rays`
  - `conf_self` from combined confidence terms
- Keeps compatibility keys expected by losses.

## 7) Loss Stack

Core losses (`src/dust3r/losses.py`):

- `Regr3DPose`: 3D + pose regression with metric/non-metric normalization behavior.
- `Regr3DPose_ScaleInv`: scale-invariant variant.
- `ConfLoss`: confidence-weighted wrapper for pixel/point losses.
- `RGBLoss`: photometric term.
- `RayConsistencyLoss`: aligns predicted `unit_rays` with GT ray-map direction channels.

Encoder-only branch:

- `EncoderFeatureConsistencyLoss`
- `EncoderPoseWeightedTemporalLoss`

## 8) Training Stage Progression

Canonical pipeline:

1. `stage1.yaml`: 224 + linear, baseline multi-dataset training.
2. `stage2.yaml`: continue 224 training with expanded data mix.
3. `stage3.yaml`: switch to 512 multi-aspect + DPT head.
4. `stage4.yaml`: long-context training (`num_views=32`), encoder frozen.
5. `dpt_512_vary_4_64.yaml`: variable-view finetune up to long sequences.

Fork-specific configs in this repo include:

- `encoder_only_mydata.yaml`
- `decoder_recurrent_k43_smoke.yaml`
- `dualdpt_ray_aware.yaml`
- `linear_224_fixed_16.yaml`

## 9) Evaluation Map

Docs entrypoint: `docs/eval.md`.

Key pipelines:

1. `eval/mv_recon/run.sh` -> `eval/mv_recon/launch.py`.
2. `eval/relpose/run.sh`.
3. `eval/video_depth/run.sh`.
4. `eval/monodepth/run.sh`.

Global alignment in GA demo:

- `cloud_opt/dust3r_opt/global_aligner(...)`
- optimizer uses `compute_global_alignment(...)` loop.

## 10) Practical Risks for Fine-Tuning

1. Dirty pathing and import priority:
   - `add_ckpt_path.py` forces local `src` import precedence.
2. Absolute paths in some experiment configs can break portability.
3. TBPTL + checkpointing combinations can silently change memory/grad behavior.
4. Metric vs non-metric datasets affect normalization path in `Regr3DPose`.
5. Some local tests/scripts are smoke-level and may lag current API.

## 11) Recommended First Execution Order

1. Run small inference (`demo.py`) with known checkpoint.
2. Run tiny encoder-only experiment on your local processed data.
3. Run tiny decoder fine-tune (`decoder_recurrent_k43_smoke.yaml`-style budget).
4. Run one eval subset (mv_recon or relpose) before scaling epochs.
