# Container Readiness Report

Date: 2026-04-09
Target container: docker-dev-1
Image: docker-dev
Container project root: /app/recurrentTransformer/CUT3R_encoder

## Summary

Container is largely ready for local CUT3R_encoder development and custom training flows.
Remaining blockers are now limited to missing official large-scale dataset trees:

1. Official stage datasets such as processed_co3d, processed_scannet, processed_arkitscenes, etc. are not present in this container.
2. Full official stage1-stage4 reproduction therefore cannot run end-to-end without adding those datasets.

## Verified Passes

1. Container is running and project is correctly bind-mounted:
   - /home/theTarnished/kimera/recursiveModelsTesting -> /app
2. Core project files exist:
   - README, docs, train/eval scripts, model/loss/inference modules, main stage configs.
3. Python runtime in container is functional.
4. CUDA is visible in container:
   - torch CUDA available
   - 1 GPU detected (RTX 2060)
5. Dependency imports pass for:
   - torch, torchvision, hydra, accelerate, transformers, cv2, einops, roma, viser, evo, open3d.
6. Entrypoints that start correctly:
   - demo.py --help
   - eval/monodepth/launch.py --help
   - eval/relpose/launch.py --help
   - eval/video_depth/launch.py --help
7. Added contract validator runs successfully in container on local dataset:
   - src/tools/validate_dataset_contract.py with encoder_only_mydata and processed_mvs_synth root.
8. Config path hardening applied:
   - stage and finetune configs now use data_root interpolation with CUT3R_DATA_ROOT defaulting to /app/recurrentTransformer/dataset.
   - missing canonical pretrained defaults were set to null in affected configs to prevent immediate file-not-found failures.
9. Eval run scripts hardened:
   - eval/mv_recon/run.sh
   - eval/relpose/run.sh
   - eval/video_depth/run.sh
   - eval/monodepth/run.sh
   These now accept CUT3R_EVAL_WEIGHTS and fall back to an existing local checkpoint when src/cut3r_512_dpt_4_64.pth is absent.

## Verified Failures (Blockers)

1. Stage1 official config still fails with current filesystem because dataset is absent:
   - Missing /app/recurrentTransformer/dataset/dust3r_data/processed_co3d/selected_seqs_train.json
2. Canonical README checkpoints are missing at expected top-level paths:
   - src/cut3r_224_linear_4.pth
   - src/cut3r_512_dpt_4_64.pth

## Important Note

Many custom checkpoints are present in src/checkpoints and checkpoints. Local/custom configs can run even though canonical official checkpoint paths are absent.

## Remediation Actions

1. Provide official processed datasets under CUT3R_DATA_ROOT for stage1-stage4 reproduction.
2. Optionally add canonical CUT3R checkpoints at expected paths if you want README-default behavior.
3. Keep using the validator before long runs:
   - src/tools/validate_dataset_contract.py

## Suggested Pre-Run Gate

Before any long training or eval run, ensure all checks pass:

1. demo.py --help
2. src/train.py --config-name stage1 --cfg job
3. src/tools/validate_dataset_contract.py for selected config
4. eval pipeline help command for the exact target benchmark
