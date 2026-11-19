import os.path as osp
import numpy as np
import cv2
import numpy as np
import itertools
import os
import sys
import pickle
import h5py
from tqdm import tqdm

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))

from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.utils.image import imread_cv2


class MapFree_Multi(BaseMultiViewDataset):
    # 'groups' is stored as a numpy.memmap on disk instead of in the scan cache pickle.
    # Halves startup RAM from 3.4 GB to near-zero for this attr.
    _SCAN_CACHE_SKIP_ATTRS = ['groups']

    def __init__(self, ROOT, *args, **kwargs):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = True
        self.max_interval = 30
        super().__init__(*args, **kwargs)

        self._load_data()

    # ── memmap helpers ──────────────────────────────────────────────────────────

    def _mm_path(self):
        """Physical path for the groups memmap file (outside VFS root)."""
        return osp.join(osp.dirname(self.ROOT.rstrip('/')), '.mapfree_groups_mm.dat')

    def _scan_cache_pre_save(self, cache_path):
        """Called before scan cache is saved (slow path). Writes groups to memmap."""
        if not hasattr(self, 'groups') or isinstance(self.groups, np.memmap):
            return
        mm_path = self._mm_path()
        shape = self.groups.shape
        expected = int(shape[0]) * int(shape[1]) * 4
        if not osp.exists(mm_path) or osp.getsize(mm_path) != expected:
            print(f"[MapFree] Writing groups memmap ({expected/1e9:.2f} GB) → {mm_path}")
            mm = np.memmap(mm_path, dtype='float32', mode='w+', shape=shape)
            mm[:] = self.groups.astype('float32')
            mm.flush()
            del mm
        self._groups_mm_shape = tuple(int(x) for x in shape)
        del self.groups
        import gc; gc.collect()
        self.groups = np.memmap(mm_path, dtype='float32', mode='r', shape=shape)

    def _scan_cache_post_restore(self, cache_path):
        """Called after scan cache is loaded (fast path). Opens groups memmap."""
        # Legacy: old cache still had groups as a numpy array — migrate on the fly.
        if hasattr(self, 'groups') and isinstance(self.groups, np.ndarray) and \
                not isinstance(self.groups, np.memmap):
            self._scan_cache_pre_save(cache_path)
            # Re-save the slim cache (groups now excluded, _groups_mm_shape added)
            try:
                skip = set(type(self)._SCAN_CACHE_SKIP_ATTRS)
                to_save = {}
                for k, v in self.__dict__.items():
                    if k in skip:
                        continue
                    try:
                        pickle.dumps(v)
                        to_save[k] = v
                    except Exception:
                        pass
                with open(cache_path, 'wb') as f:
                    pickle.dump(to_save, f)
                print(f"[MapFree] Migrated scan cache to memmap layout ({osp.getsize(cache_path)/1e6:.1f} MB)")
            except Exception as e:
                print(f"[MapFree] Could not migrate scan cache: {e}")
            return
        # Normal fast path: open the pre-existing memmap file.
        mm_path = self._mm_path()
        shape = getattr(self, '_groups_mm_shape', None)
        if shape is None or not osp.exists(mm_path):
            raise RuntimeError(
                f"[MapFree] groups memmap missing: {mm_path}. "
                "Delete the scan cache pkl and re-run to regenerate."
            )
        self.groups = np.memmap(mm_path, dtype='float32', mode='r', shape=tuple(shape))

    def imgid2path(self, img_id, scene):
        first_seq_id, first_frame_id = img_id
        return os.path.join(
            self.ROOT,
            scene,
            f"dense{first_seq_id}",
            "rgb",
            f"frame_{first_frame_id:05d}.jpg",
        )

    def path2imgid(self, subscene, filename):
        first_seq_id = int(subscene[5:])
        first_frame_id = int(filename[6:-4])
        return [first_seq_id, first_frame_id]

    def _open_h5_file(self, h5_file_path):
        """Open an h5 file — works whether stored on disk or as a nested key
        inside a VFS-backed outer .h5 archive."""
        import io
        try:
            from dust3r.datasets.utils.h5_vfs import H5VFSRegistry
            for root, outer_h5_path in H5VFSRegistry._registries:
                if h5_file_path.startswith(root + os.sep) or h5_file_path.startswith(root + "/"):
                    rel = os.path.relpath(h5_file_path, root)
                    handle = H5VFSRegistry.get_handle(outer_h5_path)
                    if rel in handle:
                        obj = handle[rel]
                        if hasattr(obj, 'dtype'):  # Dataset → raw bytes
                            raw = bytes(obj[()])
                            return h5py.File(io.BytesIO(raw), "r")
        except Exception:
            pass
        return h5py.File(h5_file_path, "r")

    def _load_data(self):
        cache_file = f"{self.ROOT}/cached_metadata_50_col_only.h5"
        if os.path.exists(cache_file):
            print(f"Loading cached metadata from {cache_file}")
            with self._open_h5_file(cache_file) as hf:
                self.scenes = list(map(lambda x: x.decode("utf-8"), hf["scenes"][:]))
                self.sceneids = hf["sceneids"][:]
                self.scope = hf["scope"][:]
                self.video_flags = hf["video_flags"][:]
                self.groups = hf["groups"][:]
                self.id_ranges = hf["id_ranges"][:]
                self.images = hf["images"][:]
        else:
            scene_dirs = sorted(
                [
                    d
                    for d in os.listdir(self.ROOT)
                    if os.path.isdir(os.path.join(self.ROOT, d))
                ]
            )
            scenes = []
            sceneids = []
            groups = []
            scope = []
            images = []
            id_ranges = []
            is_video = []
            start = 0
            j = 0
            offset = 0

            for scene in tqdm(scene_dirs):
                scenes.append(scene)
                # video sequences
                subscenes = sorted(
                    [
                        d
                        for d in os.listdir(os.path.join(self.ROOT, scene))
                        if d.startswith("dense")
                    ]
                )
                id_range_subscenes = []
                for subscene in subscenes:
                    rgb_paths = sorted(
                        [
                            d
                            for d in os.listdir(
                                os.path.join(self.ROOT, scene, subscene, "rgb")
                            )
                            if d.endswith(".jpg")
                        ]
                    )
                    assert (
                        len(rgb_paths) > 0
                    ), f"{os.path.join(self.ROOT, scene, subscene)} is empty."
                    num_imgs = len(rgb_paths)
                    images.extend(
                        [self.path2imgid(subscene, rgb_path) for rgb_path in rgb_paths]
                    )
                    id_range_subscenes.append((offset, offset + num_imgs))
                    offset += num_imgs

                # image collections
                metadata_path = os.path.join(self.ROOT, scene, "metadata.pkl")
                if os.path.exists(metadata_path):
                    metadata = pickle.load(open(metadata_path, "rb"))
                else:
                    raw_scene_dir = os.path.join(os.path.dirname(self.ROOT), "train", scene)
                    overlaps_npz = os.path.join(raw_scene_dir, "overlaps.npz")
                    if not os.path.exists(overlaps_npz):
                        raise FileNotFoundError(f"Neither metadata.pkl nor overlaps.npz found for scene {scene}")
                    overlaps_data = np.load(overlaps_npz, allow_pickle=True)
                    idxs = overlaps_data["idxs"]
                    overlaps = overlaps_data["overlaps"]
                    metadata = {}
                    for idx_idx in range(len(overlaps)):
                        score = float(overlaps[idx_idx])
                        if score <= 0.0:
                            continue
                        s1, f1, s2, f2 = map(int, idxs[idx_idx])
                        ref_img = (s1, f1)
                        other_img = ((s2, f2), score)
                        metadata.setdefault(ref_img, []).append(other_img)
                ref_imgs = list(metadata.keys())
                img_groups = []
                for ref_img in ref_imgs:
                    other_imgs = metadata[ref_img]
                    if len(other_imgs) + 1 < self.num_views:
                        continue
                    group = [(*other_img[0], other_img[1]) for other_img in other_imgs]
                    group.insert(0, (*ref_img, 1))
                    img_groups.append(np.array(group))
                    id_ranges.append(id_range_subscenes[ref_img[0]])
                    scope.append(start)
                    start = start + len(group)

                num_groups = len(img_groups)
                sceneids.extend([j] * num_groups)
                groups.extend(img_groups)
                is_video.extend([False] * num_groups)
                j += 1

            self.scenes = np.array(scenes)
            self.sceneids = np.array(sceneids)
            self.scope = np.array(scope)
            self.video_flags = np.array(is_video)
            self.groups = np.concatenate(groups, 0)
            self.id_ranges = np.array(id_ranges)
            self.images = np.array(images)

            data = dict(
                scenes=self.scenes,
                sceneids=self.sceneids,
                scope=self.scope,
                video_flags=self.video_flags,
                groups=self.groups,
                id_ranges=self.id_ranges,
                images=self.images,
            )

            with h5py.File(cache_file, "w") as h5f:
                h5f.create_dataset(
                    "scenes",
                    data=data["scenes"].astype(object),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                    compression="lzf",
                    chunks=True,
                )
                h5f.create_dataset(
                    "sceneids", data=data["sceneids"], compression="lzf", chunks=True
                )
                h5f.create_dataset(
                    "scope", data=data["scope"], compression="lzf", chunks=True
                )
                h5f.create_dataset(
                    "video_flags",
                    data=data["video_flags"],
                    compression="lzf",
                    chunks=True,
                )
                h5f.create_dataset(
                    "groups", data=data["groups"], compression="lzf", chunks=True
                )
                h5f.create_dataset(
                    "id_ranges", data=data["id_ranges"], compression="lzf", chunks=True
                )
                h5f.create_dataset(
                    "images", data=data["images"], compression="lzf", chunks=True
                )

    def __len__(self):
        return len(self.scope)

    def get_image_num(self):
        return len(self.images)

    def get_stats(self):
        return f"{len(self)} groups of views"

    def _get_views(self, idx, resolution, rng, num_views):
        scene = self.scenes[self.sceneids[idx]]
        if rng.random() < 0.6:
            ids = np.arange(self.id_ranges[idx][0], self.id_ranges[idx][1])
            cut_off = num_views if not self.allow_repeat else max(num_views // 3, 3)
            start_ids = ids[: len(ids) - cut_off + 1]
            start_id = rng.choice(start_ids)
            pos, ordered_video = self.get_seq_from_start_id(
                num_views,
                start_id,
                ids.tolist(),
                rng,
                max_interval=self.max_interval,
                video_prob=0.8,
                fix_interval_prob=0.5,
                block_shuffle=16,
            )
            ids = np.array(ids)[pos]
            image_idxs = self.images[ids]
        else:
            ordered_video = False
            seq_start_index = self.scope[idx]
            seq_end_index = self.scope[idx + 1] if idx < len(self.scope) - 1 else None
            image_idxs = (
                self.groups[seq_start_index:seq_end_index]
                if seq_end_index is not None
                else self.groups[seq_start_index:]
            )
            image_idxs, overlap_scores = image_idxs[:, :2], image_idxs[:, 2]
            replace = (
                True
                if self.allow_repeat
                or len(overlap_scores[overlap_scores > 0]) < num_views
                else False
            )
            image_idxs = rng.choice(
                image_idxs,
                num_views,
                replace=replace,
                p=overlap_scores / np.sum(overlap_scores),
            )
            image_idxs = image_idxs.astype(np.int64)

        views = []
        for v, view_idx in enumerate(image_idxs):
            img_path = self.imgid2path(view_idx, scene)
            depth_path = img_path.replace("rgb", "depth").replace(".jpg", ".npy")
            cam_path = img_path.replace("rgb", "cam").replace(".jpg", ".npz")
            sky_mask_path = img_path.replace("rgb", "sky_mask")
            image = imread_cv2(img_path)
            
            if os.path.exists(depth_path):
                depthmap = np.load(depth_path)
            else:
                depthmap = np.zeros(image.shape[:2], dtype=np.float32)
                
            camera_params = np.load(cam_path)
            intrinsics = camera_params["intrinsic"].astype(np.float32)
            camera_pose = camera_params["pose"].astype(np.float32)
            
            if os.path.exists(sky_mask_path):
                sky_mask = cv2.imread(sky_mask_path, cv2.IMREAD_UNCHANGED) >= 127
            else:
                sky_mask = np.zeros(image.shape[:2], dtype=bool)

            depthmap[sky_mask] = -1.0
            depthmap[depthmap > 400.0] = 0.0
            depthmap = np.nan_to_num(depthmap, nan=0, posinf=0, neginf=0)
            threshold = (
                np.percentile(depthmap[depthmap > 0], 98)
                if depthmap[depthmap > 0].size > 0
                else 0
            )
            depthmap[depthmap > threshold] = 0.0

            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image, depthmap, intrinsics, resolution, rng, info=(img_path)
            )
            # generate img mask and raymap mask
            img_mask, ray_mask = self.get_img_and_ray_masks(
                self.is_metric, v, rng, p=[0.75, 0.2, 0.05]
            )

            views.append(
                dict(
                    img=image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,  # cam2world
                    camera_intrinsics=intrinsics,
                    dataset="MapFree",
                    label=img_path,
                    is_metric=self.is_metric,
                    instance=img_path,
                    is_video=ordered_video,
                    quantile=np.array(0.96, dtype=np.float32),
                    img_mask=img_mask,
                    ray_mask=ray_mask,
                    camera_only=False,
                    depth_only=False,
                    single_view=False,
                    reset=False,
                )
            )
        assert len(views) == num_views
        return views
