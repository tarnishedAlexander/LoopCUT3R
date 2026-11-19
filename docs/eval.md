# Evaluation

## Datasets

Follow [MonST3R](https://github.com/Junyi42/monst3r/blob/main/data/evaluation_script.md)
and [Spann3R](https://github.com/HengyiWang/spann3r/blob/main/docs/data_preprocess.md)
to prepare the evaluation data under `data/`:

```
data/
├── bonn
├── scannetv2
├── sintel
└── tum
```

(`7scenes`, `kitti`, `neural_rgbd` and `nyu-v2` are used by the monodepth and
multi-view reconstruction launchers inherited from CUT3R.)

## Protocol of the reported results

| Task | Datasets | Settings |
|---|---|---|
| Camera pose (`eval/relpose`) | ScanNet (11 seq.), TUM-dynamics (9), Sintel (13) | 512, full sequences, ATE / RPE |
| Video depth (`eval/video_depth`) | Sintel (23 seq.), Bonn (5) | 512, scale & shift alignment |

```bash
bash tools/eval/run_benchmarks.sh LoopCUT3R checkpoints/loopcut3r_512_dpt.pth
bash tools/eval/run_benchmarks.sh CUT3R CUT3R_MODEL_PATH /path/to/CUT3R
python tools/eval/pose_table.py --out docs/results/pose_by_scene.json CUT3R LoopCUT3R
```

`run_benchmarks.sh` writes to `eval_results/{relpose,video_depth}/<dataset>_<label>`
and prints the depth metrics; `pose_table.py` averages the pose metrics over the
scenes every label scored, cutting each scene to the same number of frames for
all labels.

The launchers rebuild the model from the configuration stored in the
checkpoint: use the released `loopcut3r_512_dpt.pth` or a training checkpoint
(`checkpoint-last.pth`), not a weights-only file such as `checkpoint-final.pth`.

### Run each model from its own source tree

`add_ckpt_path.py` puts this repository's `src/` on the Python path, and
`load_model()` loads weights with `strict=False`.  A CUT3R checkpoint evaluated
from this tree is therefore built as LoopCUT3R, its 24 encoder blocks are
dropped and the encoder runs with random weights, without any error.  Pass a
CUT3R checkout as the third argument of `run_benchmarks.sh`, and check any
pairing first:

```bash
python tools/eval/check_weights.py /path/to/CUT3R CUT3R_MODEL_PATH   # want MISSING 0, UNEXPECTED 0
```

## Memory and speed

```bash
python tools/eval/efficiency.py --repo . --label loopcut3r --model_path checkpoints/loopcut3r_512_dpt.pth \
    --img_dir examples/001 --frames 13 90 300 --out efficiency_loop.json
```

Reports parameters per module, block applications per frame, weights in GPU
memory, and peak VRAM and FPS in three modes: batch (all frames in one call),
stream (`inference_recurrent`, predictions kept on the GPU) and online (only the
current frame and the state on the GPU; same predictions as stream).  The load
is checked tensor by tensor.  `tools/tesis/medir_2060.sh` runs it for both
models on a 6 GB GPU (used on an RTX 2060 and, inside the `docker/` image, on
an RTX 4050 laptop; results in `docs/results/efficiency_{2060,4050}_*.json`).

The script prints `[rope] cuda` or `[rope] pytorch`.  Without the compiled RoPE
kernel (`src/croco/models/curope`) CroCo falls back to a slow PyTorch version.
CUT3R applies RoPE in all 48 blocks and LoopCUT3R only in its decoder, so a
missing kernel slows CUT3R much more than LoopCUT3R: compare speeds only when
both runs report `[rope] cuda`.  `add_path_to_dust3r` also puts the
checkpoint's folder on the Python path, so keep checkpoints out of another
source tree's `src/`.

## Other launchers from CUT3R

```bash
bash eval/monodepth/run.sh     # eval_results/monodepth/<data>_<model>/metric.json
bash eval/video_depth/run.sh   # eval_results/video_depth/<data>_<model>/result_scale.json
bash eval/relpose/run.sh       # eval_results/relpose/<data>_<model>/_error_log.txt
bash eval/mv_recon/run.sh      # eval_results/mv_recon/<model>_<ckpt>/logs_all.txt
```
