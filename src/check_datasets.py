"""Quick diagnostic: instantiate every Stage 2 dataset and print its length.
Run inside the container:
  cd /app/recurrentTransformer/LoopCUT3R && python src/check_datasets.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Mirror the 31 dataset definitions from recurrent_stage2.yaml
DATASETS = [
    ("Co3d_Multi",         dict(split='train', ROOT='/app/recurrentTransformer/dataset/disk14tb/processed_co3d',                          aug_crop=16, resolution=224, num_views=4, n_corres=0)),
    ("WildRGBD_Multi",     dict(split='train', ROOT='/app/recurrentTransformer/dataset/disk12tb/processed/preprocess_wildrgbd',            aug_crop=16, resolution=224, num_views=4, n_corres=0)),
    ("ARKitScenes_Multi",  dict(split='train', ROOT='/app/recurrentTransformer/fastdatasets/processed_arkitscenes',                        aug_crop=16, resolution=224, num_views=4, n_corres=0)),
    ("ARKitScenesHighRes_Multi", dict(split='train', ROOT='/app/recurrentTransformer/fastdatasets/processed_arkitscenes_highres',          aug_crop=16, resolution=224, num_views=4, n_corres=0)),
    ("ScanNetpp_Multi",    dict(split='train', ROOT='/app/recurrentTransformer/dataset/disk12tb/processed_scannetpp',                      aug_crop=16, resolution=224, num_views=4, n_corres=0)),
    ("ScanNet_Multi",      dict(split='train', ROOT='/app/recurrentTransformer/dataset/diskPool/processed_scannet',                        aug_crop=16, resolution=224, num_views=4, n_corres=0)),
    ("HyperSim_Multi",     dict(split='train', ROOT='/app/recurrentTransformer/dataset/diskPool/processed_hypersim',                       aug_crop=16, resolution=224, num_views=4, n_corres=0)),
    ("BlendedMVS_Multi",   dict(split='train', ROOT='/app/recurrentTransformer/dataset/disk14tb/processed_BlendedMVS1_full',               aug_crop=16, resolution=224, num_views=4, n_corres=0)),
    ("MegaDepth_Multi",    dict(split='train', ROOT='/app/recurrentTransformer/dataset/disk14tb/processed_megadepth',                      aug_crop=16, resolution=224, num_views=4, n_corres=0)),
    ("MapFree_Multi",      dict(split=None,    ROOT='/app/recurrentTransformer/dataset/disk14tb/processed_mapfree',                        aug_crop=16, resolution=224, num_views=4, n_corres=0)),
    ("Waymo_Multi",        dict(split=None,    ROOT='/app/recurrentTransformer/fastdatasets/waymo_processed',                              aug_crop=16, resolution=224, num_views=4, n_corres=0)),
    ("VirtualKITTI2_Multi",dict(split=None,    ROOT='/app/recurrentTransformer/dataset/diskPool/vkitti_2.0.3_mixed',                       aug_crop=16, resolution=224, num_views=4, n_corres=0)),
    ("UnReal4K_Multi",     dict(split=None,    ROOT='/app/recurrentTransformer/dataset/disk12tb/processed/processed_unrealStereo4k',       aug_crop=16, resolution=224, num_views=4, n_corres=0)),
    ("TartanAir_Multi",    dict(split=None,    ROOT='/app/recurrentTransformer/dataset/diskPool/processed_TartanAir',                      aug_crop=16, resolution=224, num_views=4, n_corres=0)),
    ("DL3DV_Multi",        dict(split='train', ROOT='/app/recurrentTransformer/dataset/disk12tb/processed_dl3dv',                          aug_crop=16, resolution=224, num_views=4, n_corres=0)),
    ("Cop3D_Multi",        dict(split='train', ROOT='/app/recurrentTransformer/dataset/disk12tb/processed/processed_cop3d',                aug_crop=16, resolution=224, num_views=4, n_corres=0)),
    ("MVImgNet_Multi",     dict(split='train', ROOT='/app/recurrentTransformer/dataset/disk12tb/processed/processed_mvimgnet',             aug_crop=16, resolution=224, num_views=4, n_corres=0)),
    ("RE10K_Multi",        dict(split=None,    ROOT='/app/recurrentTransformer/dataset/disk12tb/processed/processed_re10k',                aug_crop=16, resolution=224, num_views=4, n_corres=0)),
    ("OmniObject3D_Multi", dict(split=None,    ROOT='/app/recurrentTransformer/dataset/disk14tb/processed_omniobject3d',                   aug_crop=16, resolution=224, num_views=4, n_corres=0)),
    ("ThreeDKenBurns",     dict(split=None,    ROOT='/app/recurrentTransformer/dataset/disk14tb/processed_3dkb',                           aug_crop=16, resolution=224, num_views=4, n_corres=0)),
    ("IRS",                dict(split=None,    ROOT='/app/recurrentTransformer/dataset/disk12tb/processed_irs',                            aug_crop=16, resolution=224, num_views=4, n_corres=0)),
    ("SynScapes",          dict(split=None,    ROOT='/app/recurrentTransformer/dataset/disk14tb/processed_synscapes',                      aug_crop=16, resolution=224, num_views=4, n_corres=0)),
    ("UrbanSyn",           dict(split=None,    ROOT='/app/recurrentTransformer/dataset/disk12tb/processed_urbansyn',                       aug_crop=16, resolution=224, num_views=4, n_corres=0)),
    ("EDEN_Multi",         dict(split='train', ROOT='/app/recurrentTransformer/dataset/disk12tb/processed_eden',                           aug_crop=16, resolution=224, num_views=4, n_corres=0)),
    ("BEDLAM_Multi",       dict(split='train', ROOT='/app/recurrentTransformer/dataset/disk12tb/processed_bedlam',                         aug_crop=16, resolution=224, num_views=4, n_corres=0)),
    ("DynamicReplica",     dict(split='train', ROOT='/app/recurrentTransformer/dataset/disk14tb/processed_dynamic_replica',                aug_crop=16, resolution=224, num_views=4, n_corres=0)),
    ("Spring",             dict(split=None,    ROOT='/app/recurrentTransformer/dataset/disk12tb/processed_spring',                         aug_crop=16, resolution=224, num_views=4, n_corres=0)),
    ("MVS_Synth_Multi",    dict(split='train', ROOT='/app/recurrentTransformer/fastdatasets/processed_mvs_synth',                          aug_crop=16, resolution=224, num_views=4, n_corres=0)),
    ("PointOdyssey_Multi", dict(split='train', ROOT='/app/recurrentTransformer/dataset/diskPool/processed_point_odyssey',                  aug_crop=16, resolution=224, num_views=4, n_corres=0)),
    ("UASOL_Multi",        dict(split='train', ROOT='/app/recurrentTransformer/dataset/disk14tb/processed_uasol',                          aug_crop=16, resolution=224, num_views=4, n_corres=0)),
    ("MP3D_Multi",         dict(split=None,    ROOT='/app/recurrentTransformer/dataset/disk14tb/matterport3d/processed_mp3d',              aug_crop=16, resolution=224, num_views=4, n_corres=0)),
]

import importlib, traceback

module = importlib.import_module('dust3r.datasets')

print(f"\n{'Dataset':<30} {'Length':>10}  Status")
print("-" * 55)
any_zero = False
for cls_name, kwargs in DATASETS:
    try:
        cls = getattr(module, cls_name)
        ds = cls(**kwargs)
        n = len(ds)
        status = "*** EMPTY ***" if n == 0 else "ok"
        if n == 0:
            any_zero = True
        print(f"{cls_name:<30} {n:>10}  {status}")
    except Exception as e:
        print(f"{cls_name:<30} {'ERROR':>10}  {e}")

if any_zero:
    print("\n[!] Found zero-length datasets — these will crash set_epoch!")
else:
    print("\nAll datasets non-empty. Safe to train.")
