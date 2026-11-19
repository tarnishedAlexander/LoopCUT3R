import os.path as osp
import os
import sys
import itertools

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
import cv2
import numpy as np

from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.utils.image import imread_cv2


class DL3DV_Multi(BaseMultiViewDataset):
    def __init__(self, *args, split, ROOT, **kwargs):
        self.ROOT = ROOT
        self.video = True
        self.max_interval = 20
        self.is_metric = False
        super().__init__(*args, **kwargs)

        self.loaded_data = self._load_data()

    def _load_data(self):
        # Register H5VFS before any filesystem access so os.listdir/exists go through the VFS.
        # Needed because _load_data is called after super().__init__(), bypassing the base-class
        # VFS registration that other datasets get via _scan_cache_pre_save.
        h5_path = self.ROOT.rstrip('/') + '.h5'
        import os as _os
        if not _os.path.isdir(self.ROOT):
            _real_h5 = h5_path if _os.path.isfile(h5_path) else None
            if _real_h5 is None:
                # Try one level up (e.g. disk14tb/processed_dl3dv.h5)
                _real_h5 = h5_path
            try:
                from dust3r.datasets.utils.h5_vfs import H5VFSRegistry
                H5VFSRegistry.register(self.ROOT, _real_h5)
            except Exception:
                pass

        cache_path = osp.join(self.ROOT, "dl3dv_metadata_cache.pkl")
        if osp.exists(cache_path):
            print(f"[DL3DV] Loading cached metadata from {cache_path}...")
            import pickle
            with open(cache_path, "rb") as f:
                cache_data = pickle.load(f)
            self.scenes = cache_data["scenes"]
            self.sceneids = cache_data["sceneids"]
            self.images = cache_data["images"]
            self.start_img_ids = cache_data["start_img_ids"]
            self.scene_img_list = cache_data["scene_img_list"]
            return

        self.all_scenes = sorted(
            [f for f in os.listdir(self.ROOT) if os.path.isdir(osp.join(self.ROOT, f))]
        )
        subscenes = []
        for scene in self.all_scenes:
            # not empty
            subscenes.extend(
                [
                    osp.join(scene, f)
                    for f in os.listdir(osp.join(self.ROOT, scene))
                    if os.path.isdir(osp.join(self.ROOT, scene, f))
                    and len(os.listdir(osp.join(self.ROOT, scene, f))) > 0
                ]
            )

        offset = 0
        scenes = []
        sceneids = []
        images = []
        scene_img_list = []
        start_img_ids = []
        j = 0

        for scene_idx, scene in enumerate(subscenes):
            scene_dir = osp.join(self.ROOT, scene, "dense")
            rgb_dir = osp.join(scene_dir, "rgb")
            if not osp.isdir(rgb_dir):
                print(f"Skipping {scene}: missing dense/rgb")
                continue
            rgb_paths = []
            for f in sorted(os.listdir(rgb_dir)):
                if not f.endswith(".png"):
                    continue
                basename = f[:-4]
                depth_path = osp.join(scene_dir, "depth", basename + ".npy")
                cam_path = osp.join(scene_dir, "cam", basename + ".npz")
                sky_path = osp.join(scene_dir, "sky_mask", f)
                outlier_path = osp.join(scene_dir, "outlier_mask", f)
                if all(
                    osp.isfile(p)
                    for p in (depth_path, cam_path, sky_path, outlier_path)
                ):
                    rgb_paths.append(f)
            if len(rgb_paths) == 0:
                print(f"Skipping {scene}: no frames with complete dense assets")
                continue
            num_imgs = len(rgb_paths)
            cut_off = (
                self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)
            )

            if num_imgs < cut_off:
                print(f"Skipping {scene}")
                continue

            img_ids = list(np.arange(num_imgs) + offset)
            start_img_ids_ = img_ids[: num_imgs - cut_off + 1]

            scenes.append(scene)
            scene_img_list.append(img_ids)
            sceneids.extend([j] * num_imgs)
            images.extend(rgb_paths)
            start_img_ids.extend(start_img_ids_)
            offset += num_imgs
            j += 1

        self.scenes = scenes
        self.sceneids = sceneids
        self.images = images
        self.start_img_ids = start_img_ids
        self.scene_img_list = scene_img_list

    def __len__(self):
        return len(self.start_img_ids)

    def get_image_num(self):
        return len(self.images)

    def _get_views(self, idx, resolution, rng, num_views):
        start_id = self.start_img_ids[idx]
        scene_id = self.sceneids[start_id]
        all_image_ids = self.scene_img_list[scene_id]
        pos, ordered_video = self.get_seq_from_start_id(
            num_views,
            start_id,
            all_image_ids,
            rng,
            max_interval=self.max_interval,
            block_shuffle=25,
        )
        image_idxs = np.array(all_image_ids)[pos]

        views = []
        for view_idx in image_idxs:
            scene_id = self.sceneids[view_idx]
            scene_dir = osp.join(self.ROOT, self.scenes[scene_id], "dense")

            rgb_path = self.images[view_idx]
            basename = rgb_path[:-4]

            rgb_image = imread_cv2(
                osp.join(scene_dir, "rgb", rgb_path), cv2.IMREAD_COLOR
            )
            depthmap = np.load(osp.join(scene_dir, "depth", basename + ".npy")).astype(
                np.float32
            )
            depthmap[~np.isfinite(depthmap)] = 0  # invalid
            cam_file = np.load(osp.join(scene_dir, "cam", basename + ".npz"))
            sky_mask = (
                cv2.imread(
                    osp.join(scene_dir, "sky_mask", rgb_path), cv2.IMREAD_UNCHANGED
                )
                >= 127
            )
            outlier_mask = cv2.imread(
                osp.join(scene_dir, "outlier_mask", rgb_path), cv2.IMREAD_UNCHANGED
            )
            depthmap[sky_mask] = -1.0
            depthmap[outlier_mask >= 127] = 0.0
            depthmap = np.nan_to_num(depthmap, nan=0, posinf=0, neginf=0)
            threshold = (
                np.percentile(depthmap[depthmap > 0], 98)
                if depthmap[depthmap > 0].size > 0
                else 0
            )
            depthmap[depthmap > threshold] = 0.0

            intrinsics = cam_file["intrinsic"].astype(np.float32)
            camera_pose = cam_file["pose"].astype(np.float32)

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
            )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap.astype(np.float32),
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset="dl3dv",
                    label=self.scenes[scene_id] + "_" + rgb_path,
                    instance=osp.join(scene_dir, "rgb", rgb_path),
                    is_metric=self.is_metric,
                    is_video=ordered_video,
                    quantile=np.array(0.9, dtype=np.float32),
                    img_mask=True,
                    ray_mask=False,
                    camera_only=False,
                    depth_only=False,
                    single_view=False,
                    reset=False,
                )
            )
        return views
