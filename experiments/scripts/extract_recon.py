#!/usr/bin/env python3
"""
Dump everything a CUT3R-family model predicts for one image sequence to a .npz,
so two models can be compared with numbers instead of by eye.

Runs unchanged in both the CUT3R and LoopCUT3R trees: it reuses the local
demo.py for image loading and preprocessing, so both models see byte-identical
inputs and any difference in the output is the model.

    python extract_recon.py --model_path CKPT --seq_path examples/002 --out out.npz
"""
import argparse
import os
import sys

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--seq_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model_cfg", default=None,
                    help="only needed for checkpoints without embedded args")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    from add_ckpt_path import add_path_to_dust3r

    add_path_to_dust3r(args.model_path)

    import demo as D  # local demo.py -> identical preprocessing in both repos
    from src.dust3r.model import ARCroco3DStereo
    from src.dust3r.inference import inference_recurrent
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from src.dust3r.utils.geometry import geotrf

    img_paths, tmpdir = D.parse_seq_path(args.seq_path)
    if not img_paths:
        raise SystemExit(f"no images found in {args.seq_path}")
    print(f"{len(img_paths)} frames from {args.seq_path}")
    views = D.prepare_input(img_paths, [True] * len(img_paths), args.size)

    print(f"loading {args.model_path}")
    # CUT3R's from_pretrained has no model_cfg_str parameter; LoopCUT3R's does.
    # Only pass it when it was actually asked for, so one script serves both.
    kw = {"model_cfg_str": args.model_cfg} if args.model_cfg else {}
    model = ARCroco3DStereo.from_pretrained(args.model_path, **kw).to(args.device)
    model.eval()

    outputs, _ = inference_recurrent(views, model, args.device)
    preds, vws = outputs["pred"], outputs["views"]
    n = len(preds)

    pts_cam = torch.cat([p["pts3d_in_self_view"].cpu() for p in preds], 0)
    pts_world = torch.cat([p["pts3d_in_other_view"].cpu() for p in preds], 0)
    conf_cam = torch.cat([p["conf_self"].cpu() for p in preds], 0)
    conf_world = torch.cat([p["conf"].cpu() for p in preds], 0)
    poses = torch.cat(
        [pose_encoding_to_camera(p["camera_pose"].clone()).cpu() for p in preds], 0
    )

    # demo.py renders geotrf(pose, pts_cam), NOT the pts3d_in_other_view head.
    # Keep both: it is the only way to tell whether an artifact comes from the
    # pose head or from the world-points head.
    pts_posed = torch.cat(
        [geotrf(poses[i : i + 1], pts_cam[i : i + 1]) for i in range(n)], 0
    )

    rgb = torch.cat([0.5 * (v["img"].permute(0, 2, 3, 1).cpu() + 1.0) for v in vws], 0)

    f32 = lambda t: t.numpy().astype(np.float32)
    np.savez_compressed(
        args.out,
        pts_cam=f32(pts_cam),
        pts_world=f32(pts_world),
        pts_posed=f32(pts_posed),
        conf_cam=f32(conf_cam),
        conf_world=f32(conf_world),
        pose_c2w=f32(poses),
        rgb=f32(rgb),
        img_paths=np.array([str(p) for p in img_paths]),
        seq=np.array(args.seq_path),
        model_path=np.array(args.model_path),
        size=np.array(args.size),
    )
    mb = os.path.getsize(args.out) / 1e6
    print(f"wrote {args.out}  ({mb:.0f} MB)  frames={n}  HxW={tuple(pts_cam.shape[1:3])}")

    if tmpdir:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
