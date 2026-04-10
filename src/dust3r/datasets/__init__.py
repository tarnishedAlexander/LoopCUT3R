from .utils.transforms import *
from .base.batched_sampler import BatchedRandomSampler  # noqa
from .arkitscenes import ARKitScenes_Multi  # noqa
from .arkitscenes_highres import ARKitScenesHighRes_Multi
from .bedlam import BEDLAM_Multi
from .blendedmvs import BlendedMVS_Multi  # noqa
from .co3d import Co3d_Multi  # noqa
from .cop3d import Cop3D_Multi
from .dl3dv import DL3DV_Multi
from .dynamic_replica import DynamicReplica
from .eden import EDEN_Multi
from .hypersim import HyperSim_Multi
from .hoi4d import HOI4D_Multi
from .irs import IRS
from .mapfree import MapFree_Multi
from .megadepth import MegaDepth_Multi  # noqa
from .mp3d import MP3D_Multi
from .mvimgnet import MVImgNet_Multi
from .mvs_synth import MVS_Synth_Multi
from .omniobject3d import OmniObject3D_Multi
from .pointodyssey import PointOdyssey_Multi
from .realestate10k import RE10K_Multi
from .scannet import ScanNet_Multi
from .scannetpp import ScanNetpp_Multi  # noqa
from .smartportraits import SmartPortraits_Multi
from .spring import Spring
from .synscapes import SynScapes
from .tartanair import TartanAir_Multi
from .threedkb import ThreeDKenBurns
from .uasol import UASOL_Multi
from .urbansyn import UrbanSyn
from .unreal4k import UnReal4K_Multi
from .vkitti2 import VirtualKITTI2_Multi  # noqa
from .waymo import Waymo_Multi  # noqa
from .wildrgbd import WildRGBD_Multi  # noqa


from accelerate import Accelerator
from .base.easy_dataset import EasyDataset
from dust3r.utils.safe_eval import safe_eval_expr


class _SamplerSignatureError(RuntimeError):
    pass


def _build_dataset_eval_namespace():
    namespace = {}
    excluded = {
        "Accelerator",
        "EasyDataset",
        "BatchedRandomSampler",
        "get_data_loader",
        "safe_eval_expr",
    }
    for name, value in globals().items():
        if name.startswith("_") or name in excluded:
            continue
        if isinstance(value, type) and issubclass(value, EasyDataset):
            namespace[name] = value
            continue
        if callable(value):
            namespace[name] = value
            continue
        if isinstance(value, (int, float, bool, str)):
            namespace[name] = value
    return namespace


DATASET_EVAL_NAMESPACE = _build_dataset_eval_namespace()


def _get_world_size_and_rank(accelerator: Accelerator):
    world_size = 1
    rank = 0
    if accelerator is None:
        return world_size, rank

    world_size = int(getattr(accelerator, "num_processes", world_size))
    rank = int(getattr(accelerator, "process_index", rank))
    state = getattr(accelerator, "state", None)
    if state is not None:
        rank = int(getattr(state, "process_index", rank))
    return max(world_size, 1), max(rank, 0)


def _make_sampler(dataset, batch_size, shuffle, drop_last, world_size, rank, fixed_length):
    kwargs_candidates = [
        {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "drop_last": drop_last,
            "world_size": world_size,
            "rank": rank,
            "fixed_length": fixed_length,
        },
        {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "drop_last": drop_last,
            "world_size": world_size,
            "fixed_length": fixed_length,
        },
        {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "drop_last": drop_last,
            "fixed_length": fixed_length,
        },
        {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "drop_last": drop_last,
        },
    ]

    last_signature_error = None
    for kwargs in kwargs_candidates:
        try:
            return dataset.make_sampler(**kwargs)
        except TypeError as exc:
            message = str(exc)
            if "unexpected keyword argument" in message:
                last_signature_error = exc
                continue
            raise

    raise _SamplerSignatureError(str(last_signature_error))


def get_data_loader(
    dataset,
    batch_size,
    num_workers=8,
    shuffle=True,
    drop_last=True,
    pin_mem=True,
    accelerator: Accelerator = None,
    fixed_length=False,
):
    import torch

    # pytorch dataset
    if isinstance(dataset, str):
        dataset = safe_eval_expr(
            dataset,
            DATASET_EVAL_NAMESPACE,
            context="dataset",
        )

    world_size, rank = _get_world_size_and_rank(accelerator)

    try:
        sampler = _make_sampler(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            world_size=world_size,
            rank=rank,
            fixed_length=fixed_length,
        )
        shuffle = False

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_mem,
        )

    except (AttributeError, NotImplementedError, _SamplerSignatureError):
        sampler = None

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=drop_last,
        )

    return data_loader
