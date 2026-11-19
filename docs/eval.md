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
bash tools/eval/run_benchmarks.sh LoopCUT3R MODEL_PATH
bash tools/eval/run_benchmarks.sh CUT3R CUT3R_MODEL_PATH /path/to/CUT3R
python tools/eval/pose_table.py --out docs/results/pose_by_scene.json CUT3R LoopCUT3R
```

`run_benchmarks.sh` writes to `eval_results/{relpose,video_depth}/<dataset>_<label>`
and prints the depth metrics; `pose_table.py` averages the pose metrics over the
scenes every label scored, cutting each scene to the same number of frames for
all labels.

Use a training checkpoint (`checkpoint-last.pth`): the launchers rebuild the
model from the model string stored in it.

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
python tools/eval/efficiency.py --repo . --label loopcut3r --model_path MODEL_PATH \
    --img_dir examples/001 --frames 13 90 300 --out efficiency_loop.json
```

Reports parameters per module, block applications per frame, weights in GPU
memory, and peak VRAM and FPS in batch and streaming mode.  The load is checked
tensor by tensor.  `tools/tesis/medir_2060.sh` runs it for both models on a
6 GB GPU.

## Other launchers from CUT3R

```bash
bash eval/monodepth/run.sh     # eval_results/monodepth/<data>_<model>/metric.json
bash eval/video_depth/run.sh   # eval_results/video_depth/<data>_<model>/result_scale.json
bash eval/relpose/run.sh       # eval_results/relpose/<data>_<model>/_error_log.txt
bash eval/mv_recon/run.sh      # eval_results/mv_recon/<model>_<ckpt>/logs_all.txt
```
