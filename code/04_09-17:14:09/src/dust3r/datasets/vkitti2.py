import os.path as osp
import numpy as np
import cv2
import numpy as np
import itertools
import os
import sys

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))

from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.utils.image import imread_cv2


class VirtualKITTI2_Multi(BaseMultiViewDataset):

    def __init__(self, ROOT, *args, **kwargs):
        self.ROOT = ROOT
        self.rgb_root_candidates = [
            ROOT.replace("vkitti_2.0.3_mixed", "vkitti_2.0.3_rgb"),
            ROOT.replace("dataset/vkitti_2.0.3_mixed", "dataset/distill_croco/vkitti_2.0.3_rgb"),
            osp.join(osp.dirname(ROOT), "vkitti_2.0.3_rgb"),
            osp.join(osp.dirname(ROOT), "distill_croco", "vkitti_2.0.3_rgb"),
        ]
        self.depth_root_candidates = [
            ROOT.replace("vkitti_2.0.3_mixed", "vkitti_2.0.3_depth"),
            ROOT.replace("dataset/vkitti_2.0.3_mixed", "dataset/vkitti_2.0.3_depth"),
            osp.join(osp.dirname(ROOT), "vkitti_2.0.3_depth"),
        ]
        self.video = True
        self.is_metric = True
        self.max_interval = 5
        super().__init__(*args, **kwargs)
        # loading all
        self._load_data(self.split)

    def _resolve_vkitti_paths(self, scene_rel, cam, basename):
        mixed_scene_dir = osp.join(self.ROOT, scene_rel, cam)
        mixed_rgb = osp.join(mixed_scene_dir, basename + "_rgb.jpg")
        mixed_depth = osp.join(mixed_scene_dir, basename + "_depth.png")

        if osp.exists(mixed_rgb) and osp.exists(mixed_depth):
            return mixed_rgb, mixed_depth

        raw_rgb_name = f"rgb_{basename}.jpg"
        raw_depth_name = f"depth_{basename}.png"

        for rgb_root in self.rgb_root_candidates:
            for depth_root in self.depth_root_candidates:
                rgb_path = osp.join(rgb_root, scene_rel, "frames", "rgb", cam, raw_rgb_name)
                depth_path = osp.join(depth_root, scene_rel, "frames", "depth", cam, raw_depth_name)
                if osp.exists(rgb_path) and osp.exists(depth_path):
                    return rgb_path, depth_path

        return mixed_rgb, mixed_depth

    def _load_data(self, split=None):
        scene_dirs = sorted(
            [
                d
                for d in os.listdir(self.ROOT)
                if os.path.isdir(os.path.join(self.ROOT, d))
            ]
        )
        if split == "train":
            scene_dirs = scene_dirs[:-1]
        elif split == "test":
            scene_dirs = scene_dirs[-1:]
        seq_dirs = []
        for scene in scene_dirs:
            seq_dirs += sorted(
                [
                    os.path.join(scene, d)
                    for d in os.listdir(os.path.join(self.ROOT, scene))
                    if os.path.isdir(os.path.join(self.ROOT, scene, d))
                ]
            )
        offset = 0
        scenes = []
        sceneids = []
        images = []
        scene_img_list = []
        start_img_ids = []
        j = 0

        for seq_idx, seq in enumerate(seq_dirs):
            seq_path = osp.join(self.ROOT, seq)
            for cam in ["Camera_0", "Camera_1"]:
                cam_dir = osp.join(seq_path, cam)
                if not osp.isdir(cam_dir):
                    continue

                basenames = sorted(
                    {f[:5] for f in os.listdir(cam_dir) if f.endswith("_cam.npz")}
                )
                if len(basenames) == 0:
                    basenames = sorted(
                        {f[:5] for f in os.listdir(cam_dir) if f.endswith(".jpg")}
                    )

                num_imgs = len(basenames)
                cut_off = (
                    self.num_views
                    if not self.allow_repeat
                    else max(self.num_views // 3, 3)
                )
                if num_imgs < cut_off:
                    print(f"Skipping {seq}/{cam}: frames={num_imgs}")
                    continue
                img_ids = list(np.arange(num_imgs) + offset)
                start_img_ids_ = img_ids[: num_imgs - cut_off + 1]

                scenes.append(seq + "/" + cam)
                scene_img_list.append(img_ids)
                sceneids.extend([j] * num_imgs)
                images.extend(basenames)
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

    def get_stats(self):
        return f"{len(self)} groups of views"

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
            video_prob=1.0,
            fix_interval_prob=0.9,
        )
        image_idxs = np.array(all_image_ids)[pos]

        views = []

        for v, view_idx in enumerate(image_idxs):
            scene_id = self.sceneids[view_idx]
            scene_rel = self.scenes[scene_id].rsplit("/", 1)[0]
            cam = self.scenes[scene_id].rsplit("/", 1)[1]
            scene_dir = osp.join(self.ROOT, self.scenes[scene_id])
            basename = self.images[view_idx]

            img = basename + "_rgb.jpg"
            rgb_path, depth_path = self._resolve_vkitti_paths(scene_rel, cam, basename)
            image = imread_cv2(rgb_path)
            depthmap = (
                cv2.imread(
                    depth_path,
                    cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,
                ).astype(np.float32)
                / 100.0
            )
            camera_params = np.load(osp.join(scene_dir, basename + "_cam.npz"))

            intrinsics = camera_params["camera_intrinsics"]
            camera_pose = camera_params["camera_pose"]

            sky_mask = depthmap >= 655
            depthmap[sky_mask] = -1.0  # sky

            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                image, depthmap, intrinsics, resolution, rng, info=(scene_dir, img)
            )

            # generate img mask and raymap mask
            img_mask, ray_mask = self.get_img_and_ray_masks(
                self.is_metric, v, rng, p=[0.85, 0.1, 0.05]
            )

            views.append(
                dict(
                    img=image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,  # cam2world
                    camera_intrinsics=intrinsics,
                    dataset="VirtualKITTI2",
                    label=scene_dir,
                    is_metric=self.is_metric,
                    instance=scene_dir + "_" + img,
                    is_video=ordered_video,
                    quantile=np.array(1.0, dtype=np.float32),
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
