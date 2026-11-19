# LoopCUT3R

LoopCUT3R is a depth-recurrent, weight-shared version of
[CUT3R](https://github.com/CUT3R/CUT3R) for online 3D reconstruction from
video.  The 24 independent blocks of CUT3R's ViT-L encoder become 3 shared
blocks applied for 8 passes, and each decoder branch runs a few shared blocks
instead of 12 independent ones.  The recurrent encoder is initialised by
distillation from CUT3R's encoder with Raptor and the whole model is then
trained in three phases on up to 31 datasets.

The goal was to fit CUT3R on a 6 GB GPU.  The model is less than half the size
and runs faster with less memory, but it is **not** as accurate as CUT3R; the
numbers below are the measured ones, with both models evaluated under the same
protocol.

## Results

Parameters, memory and speed (RTX 5090, 512x288 input, frame-by-frame):

| | CUT3R | LoopCUT3R | change |
|---|---:|---:|---:|
| Parameters | 793.31 M | 358.64 M | -54.8 % |
| Encoder | 302.31 M (24 blocks) | 37.79 M (3 blocks x 8 passes) | -87.5 % |
| Decoder, both branches | 226.88 M | 56.72 M | -75.0 % |
| Weights in GPU memory | 3.01 GB | 1.39 GB | -53.8 % |
| Peak VRAM, stream, 13 frames | 3.48 GB | 1.85 GB | -46.8 % |
| Peak VRAM, stream, 300 frames | 7.69 GB | 6.05 GB | -21.3 % |
| Peak VRAM, online, any length | 3.31 GB | 1.68 GB | -49.3 % |
| Speed, stream, 13 frames | 34.5 FPS | 46.4 FPS | +35 % |

*Stream* is the reference implementation (`inference_recurrent`), which keeps
every frame's predictions on the GPU, so its memory grows with the sequence.
*Online* keeps only the current frame and the recurrent state on the GPU, as a
live application would, and its memory does not grow.

On a 6 GB GPU (RTX 2060, 512x288):

| | CUT3R | LoopCUT3R | change |
|---|---:|---:|---:|
| Peak VRAM, online, 300 frames | 3.24 GB | 1.60 GB | -50.5 % |
| Peak VRAM, stream, 90 frames | 4.79 GB | 3.15 GB | -34.2 % |
| Stream, 300 frames | out of memory | out of memory | |
| Speed, online, 300 frames | 3.4 FPS | 4.2 FPS | +22 % |
| Speed, 224x224, batch, 13 frames | 6.7 FPS | 10.4 FPS | +54 % |

Both models run in 6 GB in online mode; neither is real-time on this GPU.
The pose memory
(152.31 M) and the prediction heads (81.89 M) are unchanged and make up 65 % of
LoopCUT3R's parameters.

Camera pose (ATE and RPE translation in metres, RPE rotation in degrees; lower
is better):

| | ScanNet ATE | RPE t | RPE r | TUM-dyn. ATE | RPE t | RPE r | Sintel ATE | RPE t | RPE r |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CUT3R | 0.169 | 0.205 | 3.02 | 0.145 | 0.186 | 7.84 | 0.175 | 0.071 | 0.65 |
| LoopCUT3R | 0.890 | 0.801 | 54.81 | 0.780 | 0.572 | 42.69 | 0.237 | 0.112 | 2.89 |

Video depth (scale & shift alignment):

| | Sintel AbsRel | Sintel δ<1.25 | Bonn AbsRel | Bonn δ<1.25 |
|---|---:|---:|---:|---:|
| CUT3R | 0.456 | 56.07 % | 0.053 | 97.43 % |
| LoopCUT3R | 0.581 | 42.24 % | 0.110 | 89.72 % |

Per-scene values are in [`docs/results/`](docs/results) and the figures in
[`docs/figures/`](docs/figures).

## Installation

```bash
git clone https://github.com/tarnishedAlexander/LoopCUT3R.git
cd LoopCUT3R
conda create -n loopcut3r python=3.11 cmake=3.14.0
conda activate loopcut3r
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # match your CUDA
pip install -r requirements.txt
# evaluation
pip install --no-deps evo==1.30.6
pip install open3d
```

Optionally compile the CUDA kernels for RoPE (as in CroCo v2):

```bash
cd src/croco/models/curope/ && python setup.py build_ext --inplace && cd -
```

## Checkpoints

| Model | Training resolutions | Head | Size |
|---|---|---|---|
| [`loopcut3r_512_dpt.pth`](https://drive.google.com/file/d/1ZShDKb_0NKOPSMvboJFOc5CqCFJkODkj/view?usp=sharing) | 512x384 ... 256x512 (10 aspect ratios) | DPT | 1.43 GB |

This is the phase 3 model whose results are reported above.  The file holds the
weights and the model configuration, so it loads with `--model_path` alone.

```bash
mkdir -p checkpoints
gdown --fuzzy https://drive.google.com/file/d/1ZShDKb_0NKOPSMvboJFOc5CqCFJkODkj/view -O checkpoints/loopcut3r_512_dpt.pth
sha256sum checkpoints/loopcut3r_512_dpt.pth
# 6983bfa849224dfa2e672b6e71a288d348950e4a0226d8adade944c1015b62e9
```

CUT3R's checkpoints are available from the
[CUT3R repository](https://github.com/CUT3R/CUT3R).

## Inference

```bash
python demo.py --model_path checkpoints/loopcut3r_512_dpt.pth --seq_path examples/001 --size 512 \
    --vis_threshold 1.5 --output_dir tmp
```

`--seq_path` can be a folder of frames or a video.  The reconstruction is
shown with viser on port 8080 and written to `--output_dir`.

## Architecture

| | CUT3R | LoopCUT3R |
|---|---|---|
| Encoder | 24 independent blocks | 3 shared blocks x 8 passes (24 applications) |
| Decoder (image and state branch) | 12 independent blocks each | 3 shared blocks each, 4 applications |
| Pose memory, heads | as CUT3R | unchanged |

The recurrent encoder is `RecurrentBlock` in
[`src/croco/models/blocks.py`](src/croco/models/blocks.py), applied by
`CroCoNet._encode_image` in [`src/croco/models/croco.py`](src/croco/models/croco.py).
The configuration fields (`num_blocks`, `num_steps`, `dec_num_blocks_*`,
`dec_num_steps_*`, TBPTL settings) are in `ARCroco3DStereoConfig`,
[`src/dust3r/model.py`](src/dust3r/model.py).

Training uses truncated backpropagation through the passes (TBPTL): the first
`tbptl_warmup` passes of the encoder run without gradient.

The recurrent encoder does not use positional encoding: like the Raptor blocks
it is distilled from, its attention ignores token positions (CUT3R's encoder
uses RoPE).  Position enters the model in the decoder, which keeps RoPE.
Adding RoPE to the encoder was tried and is documented in
[`experiments/README.md`](experiments/README.md).

## Training

All training ran on one RTX 5090.  Dataset preparation follows CUT3R; see
[`docs/preprocess.md`](docs/preprocess.md).  Set `LOOPCUT3R_DATA` to the
folder holding the preprocessed datasets.

| Step | Config | Starts from |
|---|---|---|
| Encoder distillation | done with Raptor (separate repository) | CUT3R encoder |
| Check of the distilled encoder | `encoder_only_raptor_backbone`, `encoder_only_raptor_3day_multidata` | Raptor encoder |
| Phase 1, 224, linear head | `recurrent_stage1`, then `recurrent_stage1_r3` | Raptor encoder |
| Phase 2, 224, 31 datasets | `recurrent_stage2` | phase 1 |
| Phase 3, 512, DPT head | `recurrent_stage3`, then `recurrent_stage3b` | phase 2 |

```bash
cd src
HYDRA_FULL_ERROR=1 accelerate launch train.py --config-name recurrent_stage1
```

See [`docs/train.md`](docs/train.md) for details.

## Evaluation

```bash
# a checkpoint must be run by the source tree that built it
python tools/eval/check_weights.py . checkpoints/loopcut3r_512_dpt.pth

bash tools/eval/run_benchmarks.sh LoopCUT3R checkpoints/loopcut3r_512_dpt.pth
bash tools/eval/run_benchmarks.sh CUT3R CUT3R_MODEL_PATH /path/to/CUT3R
python tools/eval/pose_table.py CUT3R LoopCUT3R

python tools/eval/efficiency.py --repo . --label loopcut3r \
    --model_path checkpoints/loopcut3r_512_dpt.pth --img_dir examples/001 --frames 13 90 300
```

See [`docs/eval.md`](docs/eval.md) for the datasets and the protocol.

## Repository layout

```
config/            training configs of the released model
src/               model, losses, datasets and train.py
eval/              benchmark launchers (from CUT3R)
tools/eval/        benchmark runner, pose table, efficiency, weight check
tools/h5/          pack datasets into HDF5 archives
tools/tesis/       figures and GPU measurement script of the thesis
datasets_preprocess/
tests/             python tests/test_recurrent_encoder.py
docs/              training, evaluation, preprocessing, results, figures
experiments/       configs and scripts of experiments not in the final model
```

## Acknowledgements

LoopCUT3R is built on [CUT3R](https://github.com/CUT3R/CUT3R) and reuses its
code, data pipeline and evaluation.  CUT3R in turn builds on
[DUSt3R](https://github.com/naver/dust3r), [CroCo](https://github.com/naver/croco),
[MonST3R](https://github.com/Junyi42/monst3r), [Spann3R](https://github.com/HengyiWang/spann3r)
and [Viser](https://github.com/nerfstudio-project/viser).  We thank the authors
for releasing their code.

## License

Code inherited from CUT3R keeps its license, CC BY-NC-SA 4.0 (see `LICENSE`).
