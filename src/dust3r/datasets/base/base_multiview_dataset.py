import PIL
import numpy as np
import torch
import random
import itertools
import os
import os.path as osp
import pickle
from dust3r.datasets.base.easy_dataset import EasyDataset
from dust3r.datasets.utils.transforms import ImgNorm, SeqColorJitter
from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates
import dust3r.datasets.utils.cropping as cropping
from dust3r.datasets.utils.corr import extract_correspondences_from_pts3d


def get_ray_map(c2w1, c2w2, intrinsics, h, w):
    c2w = np.linalg.inv(c2w1) @ c2w2
    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    grid = np.stack([i, j, np.ones_like(i)], axis=-1)
    ro = c2w[:3, 3]
    rd = np.linalg.inv(intrinsics) @ grid.reshape(-1, 3).T
    rd = (c2w @ np.vstack([rd, np.ones_like(rd[0])])).T[:, :3].reshape(h, w, 3)
    rd = rd / np.linalg.norm(rd, axis=-1, keepdims=True)
    ro = np.broadcast_to(ro, (h, w, 3))
    ray_map = np.concatenate([ro, rd], axis=-1)
    return ray_map


class BaseMultiViewDataset(EasyDataset):
    """Define all basic options.

    Usage:
        class MyDataset (BaseMultiViewDataset):
            def _get_views(self, idx, rng):
                # overload here
                views = []
                views.append(dict(img=, ...))
                return views
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if '__init__' not in cls.__dict__:
            return
        orig_init = cls.__init__

        def _wrapped_init(self, *args, **kwargs):
            root = kwargs.get('ROOT', None)
            split = kwargs.get('split', 'train') or 'all'
            # Attrs that should be excluded from the pickle and handled via custom hooks
            skip_attrs = set(getattr(cls, '_SCAN_CACHE_SKIP_ATTRS', []))

            cache_path = None
            if root:
                cache_name = f".scan_{cls.__name__}_{split}.pkl"
                cache_dir = root if osp.isdir(root) else osp.dirname(root.rstrip('/'))
                cache_path = osp.join(cache_dir, cache_name)

            if cache_path and osp.exists(cache_path):
                try:
                    with open(cache_path, 'rb') as f:
                        cached = pickle.load(f)
                    # Run orig_init but skip the expensive _load_data by patching it
                    _sentinel = object()
                    _orig_load = cls.__dict__.get('_load_data', _sentinel)
                    cls._load_data = lambda *a, **kw: None
                    try:
                        orig_init(self, *args, **kwargs)
                    finally:
                        if _orig_load is _sentinel:
                            try:
                                del cls._load_data
                            except AttributeError:
                                pass
                        else:
                            cls._load_data = _orig_load
                    # Don't restore config params — they're set by __init__ from current args.
                    # _resolutions in particular changes across stages (e.g. 224 → multi-AR).
                    CONFIG_ATTRS = {'_resolutions', 'num_views', 'allow_repeat', 'aug_crop', 'split'}
                    cached = {k: v for k, v in cached.items() if k not in CONFIG_ATTRS}
                    self.__dict__.update(cached)
                    # Allow subclass to restore skip_attrs (e.g. open a memmap)
                    if skip_attrs and hasattr(self, '_scan_cache_post_restore'):
                        self._scan_cache_post_restore(cache_path)
                    print(f"[ScanCache] Restored {cls.__name__} from {cache_path}")
                    return
                except Exception as e:
                    print(f"[ScanCache] Cache load failed for {cls.__name__}: {e}, running full scan")

            # Full init (slow path — scan from disk)
            orig_init(self, *args, **kwargs)

            # Allow subclass to handle skip_attrs before saving (e.g. write memmap)
            if skip_attrs and hasattr(self, '_scan_cache_pre_save'):
                self._scan_cache_pre_save(cache_path)

            # Save scan state for future restarts
            if cache_path:
                try:
                    to_save = {}
                    for k, v in self.__dict__.items():
                        if k in skip_attrs:
                            continue
                        try:
                            pickle.dumps(v)
                            to_save[k] = v
                        except Exception:
                            pass
                    with open(cache_path, 'wb') as f:
                        pickle.dump(to_save, f)
                    print(f"[ScanCache] Saved {cls.__name__} scan cache → {cache_path}")
                except Exception as e:
                    print(f"[ScanCache] Could not save cache for {cls.__name__}: {e}")

        cls.__init__ = _wrapped_init

    def __init__(
        self,
        *,  # only keyword arguments
        num_views=None,
        split=None,
        resolution=None,  # square_size or (width, height) or list of [(width,height), ...]
        transform=ImgNorm,
        aug_crop=False,
        n_corres=0,
        nneg=0,
        seed=None,
        allow_repeat=False,
        seq_aug_crop=False,
    ):
        assert num_views is not None, "undefined num_views"
        self.num_views = num_views
        self.split = split
        self._set_resolutions(resolution)

        self.n_corres = n_corres
        self.nneg = nneg
        assert (
            self.n_corres == "all"
            or isinstance(self.n_corres, int)
            or (
                isinstance(self.n_corres, list) and len(self.n_corres) == self.num_views
            )
        ), f"Error, n_corres should either be 'all', a single integer or a list of length {self.num_views}"
        assert (
            self.nneg == 0 or self.n_corres != "all"
        ), "nneg should be 0 if n_corres is all"

        self.is_seq_color_jitter = False
        if isinstance(transform, str):
            transform = eval(transform)
        if transform == SeqColorJitter:
            transform = SeqColorJitter()
            self.is_seq_color_jitter = True
        self.transform = transform

        self.aug_crop = aug_crop
        self.seed = seed
        self.allow_repeat = allow_repeat
        self.seq_aug_crop = seq_aug_crop

        if hasattr(self, "ROOT"):
            self._register_h5_vfs()

    def _register_h5_vfs(self):
        import os
        if not self.ROOT:
            return
        root_name = os.path.basename(self.ROOT.rstrip("/"))
        
        # Build base names to check
        names = [root_name]
        if root_name.startswith("processed_"):
            names.append(root_name[len("processed_"):])
        elif root_name.startswith("preprocess_"):
            names.append(root_name[len("preprocess_"):])
            
        possible_paths = []
        
        # Check subdirectories under /app/recurrentTransformer/dataset/
        dataset_base = "/app/recurrentTransformer/dataset"
        if os.path.isdir(dataset_base):
            for d in os.listdir(dataset_base):
                sub = os.path.join(dataset_base, d)
                if os.path.isdir(sub):
                    for n in names:
                        possible_paths.append(os.path.join(sub, f"{n}.h5"))
                        possible_paths.append(os.path.join(sub, f"{n}_dataset.h5"))
                        
        # Check under fastdatasets
        fast_base = "/app/recurrentTransformer/fastdatasets"
        if os.path.isdir(fast_base):
            for n in names:
                possible_paths.append(os.path.join(fast_base, f"{n}.h5"))
                possible_paths.append(os.path.join(fast_base, f"{n}_dataset.h5"))
                
        # Fallback to local files in dataset root
        possible_paths.append(os.path.join(self.ROOT, f"{root_name}.h5"))
        possible_paths.append(os.path.join(self.ROOT, "dataset.h5"))
        
        for h5_path in possible_paths:
            if os.path.exists(h5_path):
                try:
                    from dust3r.datasets.utils.h5_vfs import H5VFSRegistry
                    H5VFSRegistry.register(self.ROOT, h5_path)
                    break
                except Exception as e:
                    print(f"[H5VFS] Failed to register {h5_path}: {e}")

    def __len__(self):
        return len(self.scenes)

    @staticmethod
    def efficient_random_intervals(
        start,
        num_elements,
        interval_range,
        fixed_interval_prob=0.8,
        weights=None,
        seed=42,
    ):
        if random.random() < fixed_interval_prob:
            intervals = random.choices(interval_range, weights=weights) * (
                num_elements - 1
            )
        else:
            intervals = [
                random.choices(interval_range, weights=weights)[0]
                for _ in range(num_elements - 1)
            ]
        return list(itertools.accumulate([start] + intervals))

    def sample_based_on_timestamps(self, i, timestamps, num_views, interval=1):
        time_diffs = np.abs(timestamps - timestamps[i])
        ids_candidate = np.where(time_diffs < interval)[0]
        ids_candidate = np.sort(ids_candidate)
        if (self.allow_repeat and len(ids_candidate) < num_views // 3) or (
            len(ids_candidate) < num_views
        ):
            return []
        ids_sel_list = []
        ids_candidate_left = ids_candidate.copy()
        while len(ids_candidate_left) >= num_views:
            ids_sel = np.random.choice(ids_candidate_left, num_views, replace=False)
            ids_sel_list.append(sorted(ids_sel))
            ids_candidate_left = np.setdiff1d(ids_candidate_left, ids_sel)

        if len(ids_candidate_left) > 0 and len(ids_candidate) >= num_views:
            ids_sel = np.concatenate(
                [
                    ids_candidate_left,
                    np.random.choice(
                        np.setdiff1d(ids_candidate, ids_candidate_left),
                        num_views - len(ids_candidate_left),
                        replace=False,
                    ),
                ]
            )
            ids_sel_list.append(sorted(ids_sel))

        if self.allow_repeat:
            ids_sel_list.append(
                sorted(np.random.choice(ids_candidate, num_views, replace=True))
            )

        # add sequences with fixed intervals (all possible intervals)
        pos_i = np.where(ids_candidate == i)[0][0]
        curr_interval = 1
        stop = len(ids_candidate) < num_views
        while not stop:
            pos_sel = [pos_i]
            count = 0
            while len(pos_sel) < num_views:
                if count % 2 == 0:
                    curr_pos_i = pos_sel[-1] + curr_interval
                    if curr_pos_i >= len(ids_candidate):
                        stop = True
                        break
                    pos_sel.append(curr_pos_i)
                else:
                    curr_pos_i = pos_sel[0] - curr_interval
                    if curr_pos_i < 0:
                        stop = True
                        break
                    pos_sel.insert(0, curr_pos_i)
                count += 1
            if not stop and len(pos_sel) == num_views:
                ids_sel = sorted([ids_candidate[pos] for pos in pos_sel])
                if ids_sel not in ids_sel_list:
                    ids_sel_list.append(ids_sel)
            curr_interval += 1
        return ids_sel_list

    @staticmethod
    def blockwise_shuffle(x, rng, block_shuffle):
        if block_shuffle is None:
            return rng.permutation(x).tolist()
        else:
            assert block_shuffle > 0
            blocks = [x[i : i + block_shuffle] for i in range(0, len(x), block_shuffle)]
            shuffled_blocks = [rng.permutation(block).tolist() for block in blocks]
            shuffled_list = [item for block in shuffled_blocks for item in block]
            return shuffled_list

    def get_seq_from_start_id(
        self,
        num_views,
        id_ref,
        ids_all,
        rng,
        min_interval=1,
        max_interval=25,
        video_prob=0.5,
        fix_interval_prob=0.5,
        block_shuffle=None,
    ):
        """
        args:
            num_views: number of views to return
            id_ref: the reference id (first id)
            ids_all: all the ids
            rng: random number generator
            max_interval: maximum interval between two views
        returns:
            pos: list of positions of the views in ids_all, i.e., index for ids_all
            is_video: True if the views are consecutive
        """
        assert min_interval > 0, f"min_interval should be > 0, got {min_interval}"
        assert (
            min_interval <= max_interval
        ), f"min_interval should be <= max_interval, got {min_interval} and {max_interval}"
        assert id_ref in ids_all
        pos_ref = ids_all.index(id_ref)
        all_possible_pos = np.arange(pos_ref, len(ids_all))

        remaining_sum = len(ids_all) - 1 - pos_ref

        if remaining_sum >= num_views - 1:
            if remaining_sum == num_views - 1:
                assert ids_all[-num_views] == id_ref
                return [pos_ref + i for i in range(num_views)], True
            max_interval = min(max_interval, 2 * remaining_sum // (num_views - 1))
            intervals = [
                rng.choice(range(min_interval, max_interval + 1))
                for _ in range(num_views - 1)
            ]

            # if video or collection
            if rng.random() < video_prob:
                # if fixed interval or random
                if rng.random() < fix_interval_prob:
                    # regular interval
                    fixed_interval = rng.choice(
                        range(
                            1,
                            min(remaining_sum // (num_views - 1) + 1, max_interval + 1),
                        )
                    )
                    intervals = [fixed_interval for _ in range(num_views - 1)]
                is_video = True
            else:
                is_video = False

            pos = list(itertools.accumulate([pos_ref] + intervals))
            pos = [p for p in pos if p < len(ids_all)]
            pos_candidates = [p for p in all_possible_pos if p not in pos]
            pos = (
                pos
                + rng.choice(
                    pos_candidates, num_views - len(pos), replace=False
                ).tolist()
            )

            pos = (
                sorted(pos)
                if is_video
                else self.blockwise_shuffle(pos, rng, block_shuffle)
            )
        else:
            # assert self.allow_repeat
            uniq_num = remaining_sum
            new_pos_ref = rng.choice(np.arange(pos_ref + 1))
            new_remaining_sum = len(ids_all) - 1 - new_pos_ref
            new_max_interval = min(max_interval, new_remaining_sum // (uniq_num - 1))
            new_intervals = [
                rng.choice(range(1, new_max_interval + 1)) for _ in range(uniq_num - 1)
            ]

            revisit_random = rng.random()
            video_random = rng.random()

            if rng.random() < fix_interval_prob and video_random < video_prob:
                # regular interval
                fixed_interval = rng.choice(range(1, new_max_interval + 1))
                new_intervals = [fixed_interval for _ in range(uniq_num - 1)]
            pos = list(itertools.accumulate([new_pos_ref] + new_intervals))

            is_video = False
            if revisit_random < 0.5 or video_prob == 1.0:  # revisit, video / collection
                is_video = video_random < video_prob
                pos = (
                    self.blockwise_shuffle(pos, rng, block_shuffle)
                    if not is_video
                    else pos
                )
                num_full_repeat = num_views // uniq_num
                pos = (
                    pos * num_full_repeat
                    + pos[: num_views - len(pos) * num_full_repeat]
                )
            elif revisit_random < 0.9:  # random
                pos = rng.choice(pos, num_views, replace=True)
            else:  # ordered
                pos = sorted(rng.choice(pos, num_views, replace=True))
        assert len(pos) == num_views
        return pos, is_video

    def get_img_and_ray_masks(self, is_metric, v, rng, p=[0.8, 0.15, 0.05]):
        # generate img mask and raymap mask
        if v == 0 or (not is_metric):
            img_mask = True
            raymap_mask = False
        else:
            rand_val = rng.random()
            if rand_val < p[0]:
                img_mask = True
                raymap_mask = False
            elif rand_val < p[0] + p[1]:
                img_mask = False
                raymap_mask = True
            else:
                img_mask = True
                raymap_mask = True
        return img_mask, raymap_mask

    def get_stats(self):
        return f"{len(self)} groups of views"

    def __repr__(self):
        resolutions_str = "[" + ";".join(f"{w}x{h}" for w, h in self._resolutions) + "]"
        return (
            f"""{type(self).__name__}({self.get_stats()},
            {self.num_views=},
            {self.split=},
            {self.seed=},
            resolutions={resolutions_str},
            {self.transform=})""".replace(
                "self.", ""
            )
            .replace("\n", "")
            .replace("   ", "")
        )

    def _get_views(self, idx, resolution, rng, num_views):
        raise NotImplementedError()

    def __getitem__(self, idx, retry_count=0):
        # print("Receiving:" , idx)
        orig_idx = idx
        if isinstance(idx, (tuple, list, np.ndarray)):
            # the idx is specifying the aspect-ratio
            idx, ar_idx, nview = idx
        else:
            assert len(self._resolutions) == 1
            ar_idx = 0
            nview = self.num_views

        try:
            assert nview >= 1 and nview <= self.num_views
            # set-up the rng
            if self.seed:  # reseed for each __getitem__
                self._rng = np.random.default_rng(seed=self.seed + idx)
            elif not hasattr(self, "_rng"):
                seed = torch.randint(0, 2**32, (1,)).item()
                self._rng = np.random.default_rng(seed=seed)

            if self.aug_crop > 1 and self.seq_aug_crop:
                self.delta_target_resolution = self._rng.integers(0, self.aug_crop)

            # over-loaded code
            resolution = self._resolutions[
                ar_idx
            ]  # DO NOT CHANGE THIS (compatible with BatchedRandomSampler)
            views = self._get_views(idx, resolution, self._rng, nview)
            assert len(views) == nview

            if "camera_pose" not in views[0]:
                views[0]["camera_pose"] = np.ones((4, 4), dtype=np.float32)
            first_view_camera_pose = views[0]["camera_pose"]
            transform = SeqColorJitter() if self.is_seq_color_jitter else self.transform

            for v, view in enumerate(views):
                assert (
                    "pts3d" not in view
                ), f"pts3d should not be there, they will be computed afterwards based on intrinsics+depthmap for view {view_name(view)}"
                view["idx"] = (idx, ar_idx, v)

                # encode the image
                width, height = view["img"].size

                view["true_shape"] = np.int32((height, width))
                view["img"] = transform(view["img"])
                view["sky_mask"] = view["depthmap"] < 0

                assert "camera_intrinsics" in view
                if "camera_pose" not in view:
                    view["camera_pose"] = np.full((4, 4), np.nan, dtype=np.float32)
                else:
                    assert np.isfinite(
                        view["camera_pose"]
                    ).all(), f"NaN in camera pose for view {view_name(view)}"

                ray_map = get_ray_map(
                    first_view_camera_pose,
                    view["camera_pose"],
                    view["camera_intrinsics"],
                    height,
                    width,
                )
                view["ray_map"] = ray_map.astype(np.float32)

                assert "pts3d" not in view
                assert "valid_mask" not in view
                assert np.isfinite(
                    view["depthmap"]
                ).all(), f"NaN in depthmap for view {view_name(view)}"
                pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)

                view["pts3d"] = pts3d
                view["valid_mask"] = valid_mask & np.isfinite(pts3d).all(axis=-1)

                # check all datatypes
                for key, val in view.items():
                    res, err_msg = is_good_type(key, val)
                    assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"
                K = view["camera_intrinsics"]

            if self.n_corres > 0:
                ref_view = views[0]
                for view in views:
                    corres1, corres2, valid = extract_correspondences_from_pts3d(
                        ref_view, view, self.n_corres, self._rng, nneg=self.nneg
                    )
                    view["corres"] = (corres1, corres2)
                    view["valid_corres"] = valid

            # last thing done!
            for view in views:
                view["rng"] = int.from_bytes(self._rng.bytes(4), "big")
            return views
        except Exception as e:
            if retry_count >= 20:
                import traceback
                print(f"\n[ERROR] Data loading failed at index {idx} after {retry_count} retries. Exception: {e}")
                traceback.print_exc()
                raise e
            print(f"\n[WARNING] Caught exception during data loading __getitem__ for index {idx}: {e}. Retrying with another random index...")
            if isinstance(orig_idx, (tuple, list, np.ndarray)):
                random_idx_val = np.random.randint(0, len(self))
                return self.__getitem__((random_idx_val, ar_idx, nview), retry_count=retry_count + 1)
            else:
                random_idx = np.random.randint(0, len(self))
                return self.__getitem__(random_idx, retry_count=retry_count + 1)

    def _set_resolutions(self, resolutions):
        assert resolutions is not None, "undefined resolution"

        if not isinstance(resolutions, list):
            resolutions = [resolutions]

        self._resolutions = []
        for resolution in resolutions:
            if isinstance(resolution, int):
                width = height = resolution
            else:
                width, height = resolution
            assert isinstance(
                width, int
            ), f"Bad type for {width=} {type(width)=}, should be int"
            assert isinstance(
                height, int
            ), f"Bad type for {height=} {type(height)=}, should be int"
            self._resolutions.append((width, height))

    def _crop_resize_if_necessary(
        self, image, depthmap, intrinsics, resolution, rng=None, info=None
    ):
        """This function:
        - first downsizes the image with LANCZOS inteprolation,
          which is better than bilinear interpolation in
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # downscale with lanczos interpolation so that image.size == resolution
        # cropping centered on the principal point
        W, H = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W - cx)
        min_margin_y = min(cy, H - cy)
        assert min_margin_x > W / 5, f"Bad principal point in view={info}"
        assert min_margin_y > H / 5, f"Bad principal point in view={info}"
        # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)
        image, depthmap, intrinsics = cropping.crop_image_depthmap(
            image, depthmap, intrinsics, crop_bbox
        )

        # transpose the resolution if necessary
        W, H = image.size  # new size

        # high-quality Lanczos down-scaling
        target_resolution = np.array(resolution)
        if self.aug_crop > 1:
            target_resolution += (
                rng.integers(0, self.aug_crop)
                if not self.seq_aug_crop
                else self.delta_target_resolution
            )
        image, depthmap, intrinsics = cropping.rescale_image_depthmap(
            image, depthmap, intrinsics, target_resolution
        )

        # actual cropping (if necessary) with bilinear interpolation
        intrinsics2 = cropping.camera_matrix_of_crop(
            intrinsics, image.size, resolution, offset_factor=0.5
        )
        crop_bbox = cropping.bbox_from_intrinsics_in_out(
            intrinsics, intrinsics2, resolution
        )
        image, depthmap, intrinsics2 = cropping.crop_image_depthmap(
            image, depthmap, intrinsics, crop_bbox
        )

        return image, depthmap, intrinsics2


def is_good_type(key, v):
    """returns (is_good, err_msg)"""
    if isinstance(v, (str, int, tuple)):
        return True, None
    if v.dtype not in (np.float32, torch.float32, bool, np.int32, np.int64, np.uint8):
        return False, f"bad {v.dtype=}"
    return True, None


def view_name(view, batch_index=None):
    def sel(x):
        return x[batch_index] if batch_index not in (None, slice(None)) else x

    db = sel(view["dataset"])
    label = sel(view["label"])
    instance = sel(view["instance"])
    return f"{db}/{label}/{instance}"


def transpose_to_landscape(view):
    height, width = view["true_shape"]

    if width < height:
        # rectify portrait to landscape
        assert view["img"].shape == (3, height, width)
        view["img"] = view["img"].swapaxes(1, 2)

        assert view["valid_mask"].shape == (height, width)
        view["valid_mask"] = view["valid_mask"].swapaxes(0, 1)

        assert view["depthmap"].shape == (height, width)
        view["depthmap"] = view["depthmap"].swapaxes(0, 1)

        assert view["pts3d"].shape == (height, width, 3)
        view["pts3d"] = view["pts3d"].swapaxes(0, 1)

        # transpose x and y pixels
        view["camera_intrinsics"] = view["camera_intrinsics"][[1, 0, 2]]

        assert view["ray_map"].shape == (height, width, 6)
        view["ray_map"] = view["ray_map"].swapaxes(0, 1)

        assert view["sky_mask"].shape == (height, width)
        view["sky_mask"] = view["sky_mask"].swapaxes(0, 1)

        if "corres" in view:
            # transpose correspondences x and y
            view["corres"][0] = view["corres"][0][:, [1, 0]]
            view["corres"][1] = view["corres"][1][:, [1, 0]]
