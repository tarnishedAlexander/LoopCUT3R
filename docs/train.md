# Training

LoopCUT3R was trained on a single RTX 5090 (32 GB).  The configs in `config/`
are the ones that produced the released model; experiments that were not kept
are in `experiments/`.

## Data

Prepare the datasets as described in [preprocess.md](preprocess.md) and point
`LOOPCUT3R_DATA` at the folder that holds them (default `data/train`).  Every
config reads its dataset roots as `${data_root}/processed_<name>`.

A dataset can be packed into one HDF5 file with `tools/h5/pack_datasets_to_h5.py`
to avoid millions of small files.  The loader finds `<name>.h5` next to the
dataset root, or in any folder listed in `LOOPCUT3R_H5_DIRS` (colon-separated).

## 1. Encoder distillation (Raptor)

The 3 recurrent encoder blocks are distilled from CUT3R's 24-block encoder
with Raptor, in its own repository: the shared blocks are trained to imitate
the transitions between the hidden states of CUT3R's encoder, so that 3 blocks
applied for 8 passes approximate its 24 layers.  The result is a checkpoint
with the recurrent encoder weights.

Two configs check the distilled encoder on its own before it is used:

```bash
cd src
RAPTOR_RECURRENT_CKPT=/path/to/raptor_encoder.pt \
    python train.py --config-name encoder_only_raptor_backbone
```

They run only the encoder (`encoder_only_training: True`) with
`EncoderFeatureConsistencyLoss` (in `src/dust3r/losses.py`): consecutive frames
of a video must give descriptors that change smoothly without collapsing to a
constant.  `src/tools/cka_triplet.py`, `layer_similarity_encoder.py` and
`layer_similarity_decoder.py` compare the representations of CUT3R and
LoopCUT3R layer by layer with CKA.

## 2. Phases 1-3

| Phase | Config | Resolution, head | Starts from |
|---|---|---|---|
| 1 | `recurrent_stage1` | 224, linear | Raptor checkpoint (`RAPTOR_PRETRAINED`) |
| 1 | `recurrent_stage1_r3` | 224, linear | `recurrent_stage1` epoch 16 |
| 2 | `recurrent_stage2` | 224, linear, 31 datasets | phase 1 |
| 3 | `recurrent_stage3` | 512 multi-aspect, DPT | phase 2 |
| 3 | `recurrent_stage3b` | 512 multi-aspect, DPT, LR restart | `recurrent_stage3` |

Each config names its starting checkpoint in `pretrained:`.  Run from `src/`:

```bash
cd src
HYDRA_FULL_ERROR=1 accelerate launch train.py --config-name recurrent_stage1
HYDRA_FULL_ERROR=1 accelerate launch train.py --config-name recurrent_stage1_r3
HYDRA_FULL_ERROR=1 accelerate launch train.py --config-name recurrent_stage2
HYDRA_FULL_ERROR=1 accelerate launch train.py --config-name recurrent_stage3
HYDRA_FULL_ERROR=1 accelerate launch train.py --config-name recurrent_stage3b
```

If the Raptor checkpoint pickles its model class, set `RAPTOR_SRC` to the
Raptor source tree so that `torch.load` can import it.

Outputs go to `checkpoints/<exp_name>/`.  The final model is
`checkpoints/recurrent_stage3b/checkpoints/checkpoint-last.pth`.

## Recurrence settings

In the `model:` string of each config:

- `num_blocks=3, num_steps=8`: 3 shared encoder blocks applied for 8 passes.
- `dec_num_blocks_img=3, dec_num_steps_img=4` (and the same for `_state`):
  3 shared decoder blocks per branch, applied 4 times.
- `use_tbptl`, `tbptl_warmup`, `tbptl_thinking`, `tbptl_grad_steps`: truncated
  backpropagation through the encoder passes.  The first `tbptl_warmup` passes
  run without gradient.

The training loss is not a reliable signal of reconstruction quality for this
model: judge a run by the benchmarks in [eval.md](eval.md).
