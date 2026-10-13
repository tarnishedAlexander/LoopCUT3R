# LoopCUT3R


## Table of Contents
- [Datasets](#datasets)
- [Training and Fine-tuning](#training-and-fine-tuning)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## Datasets
Our training data includes 32 datasets listed below. We provide processing scripts for all of them. Please download the datasets from their official sources, and refer to [preprocess.md](docs/preprocess.md) for processing scripts and more information about the datasets.

  - [ARKitScenes](https://github.com/apple/ARKitScenes) 
  - [BlendedMVS](https://github.com/YoYo000/BlendedMVS)
  - [CO3Dv2](https://github.com/facebookresearch/co3d)
  - [MegaDepth](https://www.cs.cornell.edu/projects/megadepth/)
  - [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/) 
  - [ScanNet](http://www.scan-net.org/ScanNet/)
  - [WayMo Open dataset](https://github.com/waymo-research/waymo-open-dataset)
  - [WildRGB-D](https://github.com/wildrgbd/wildrgbd/)
  - [Map-free](https://research.nianticlabs.com/mapfree-reloc-benchmark/dataset)
  - [TartanAir](https://theairlab.org/tartanair-dataset/)
  - [UnrealStereo4K](https://github.com/fabiotosi92/SMD-Nets) 
  - [Virtual KITTI 2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/)
  - [3D Ken Burns](https://github.com/sniklaus/3d-ken-burns.git)
  - [BEDLAM](https://bedlam.is.tue.mpg.de/)
  - [COP3D](https://github.com/facebookresearch/cop3d)
  - [DL3DV](https://github.com/DL3DV-10K/Dataset)
  - [Dynamic Replica](https://github.com/facebookresearch/dynamic_stereo)
  - [EDEN](https://lhoangan.github.io/eden/)
  - [Hypersim](https://github.com/apple/ml-hypersim)
  - [IRS](https://github.com/HKBU-HPML/IRS)
  - [Matterport3D](https://niessner.github.io/Matterport/)
  - [MVImgNet](https://github.com/GAP-LAB-CUHK-SZ/MVImgNet)
  - [MVS-Synth](https://phuang17.github.io/DeepMVS/mvs-synth.html)
  - [OmniObject3D](https://omniobject3d.github.io/)
  - [PointOdyssey](https://pointodyssey.com/)
  - [RealEstate10K](https://google.github.io/realestate10k/)
  - [SmartPortraits](https://mobileroboticsskoltech.github.io/SmartPortraits/)
  - [Spring](https://spring-benchmark.org/)
  - [Synscapes](https://synscapes.on.liu.se/)
  - [UASOL](https://osf.io/64532/)
  - [UrbanSyn](https://www.urbansyn.org/)
  - [HOI4D](https://hoi4d.github.io/)


## Evaluation

### Datasets
Please follow [MonST3R](https://github.com/Junyi42/monst3r/blob/main/data/evaluation_script.md) and [Spann3R](https://github.com/HengyiWang/spann3r/blob/main/docs/data_preprocess.md) to prepare **Sintel**, **Bonn**, **KITTI**, **NYU-v2**, **TUM-dynamics**, **ScanNet**, **7scenes** and **Neural-RGBD** datasets.

The datasets should be organized as follows:
```
data/
├── 7scenes
├── bonn
├── kitti
├── neural_rgbd
├── nyu-v2
├── scannetv2
├── sintel
└── tum
```

### Evaluation Scripts
Please refer to the [eval.md](docs/eval.md) for more details.

## Training and Fine-tuning
Please refer to the [train.md](docs/train.md) for more details.

LoopCUT3R also supports two selectable head/loss variants on the same recurrent backbone:

- `original_cut3r` (original CUT3R-style heads/losses)
- `loop_new` (LoopCUT3R DualDPT + new loss stack)

See [train.md](docs/train.md) for exact commands using `--variant`.

## Acknowledgements
Our code is based on the following awesome repositories:

- [DUSt3R](https://github.com/naver/dust3r)
- [MonST3R](https://github.com/Junyi42/monst3r.git)
- [Spann3R](https://github.com/HengyiWang/spann3r.git)
- [Viser](https://github.com/nerfstudio-project/viser)

We thank the authors for releasing their code!

