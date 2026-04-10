import os
import os.path as osp
import numpy as np
from tqdm import tqdm

def main():
    root = "/home/theTarnished/kimera/recursiveModelsTesting/recurrentTransformer/dataset"
    rgb_dir = osp.join(root, "vkitti_2.0.3_rgb")
    depth_dir = osp.join(root, "vkitti_2.0.3_depth")
    text_dir = osp.join(root, "vkitti_2.0.3_textgt")
    out_dir = osp.join(root, "vkitti_2.0.3_mixed")

    os.makedirs(out_dir, exist_ok=True)

    scenes = sorted([s for s in os.listdir(rgb_dir) if s.startswith("Scene")])

    for scene in scenes:
        seqs = sorted([s for s in os.listdir(osp.join(rgb_dir, scene)) if osp.isdir(osp.join(rgb_dir, scene, s))])
        for seq in seqs:
            # We need to process both Camera_0 and Camera_1
            for cam in ["Camera_0", "Camera_1"]:
                out_seq_cam = osp.join(out_dir, scene, seq, cam)
                os.makedirs(out_seq_cam, exist_ok=True)

                raw_rgb_cam = osp.join(rgb_dir, scene, seq, "frames", "rgb", cam)
                raw_depth_cam = osp.join(depth_dir, scene, seq, "frames", "depth", cam)
                
                if not osp.exists(raw_rgb_cam):
                    continue

                images = sorted([i for i in os.listdir(raw_rgb_cam) if i.endswith(".jpg")])
                
                # Load intrinsics
                intrinsic_lines = []
                intrinsic_path = osp.join(text_dir, scene, seq, "intrinsic.txt")
                with open(intrinsic_path, "r") as f:
                    for line in f.readlines()[1:]:
                        parts = line.strip().split()
                        intrinsic_lines.append(parts)
                
                # Load extrinsics
                extrinsic_lines = []
                extrinsic_path = osp.join(text_dir, scene, seq, "extrinsic.txt")
                with open(extrinsic_path, "r") as f:
                    for line in f.readlines()[1:]:
                        parts = line.strip().split()
                        extrinsic_lines.append(parts)
                
                # Precompute dictionaries
                cam_id = "0" if cam == "Camera_0" else "1"
                
                intrinsics_dict = {}
                for parts in intrinsic_lines:
                    if parts[1] == cam_id:
                        frame_id = int(parts[0])
                        K = np.array([
                            [float(parts[2]), 0, float(parts[4])],
                            [0, float(parts[3]), float(parts[5])],
                            [0, 0, 1]
                        ]).astype(np.float32)
                        intrinsics_dict[frame_id] = K
                        
                extrinsics_dict = {}
                for parts in extrinsic_lines:
                    if parts[1] == cam_id:
                        frame_id = int(parts[0])
                        # extrinsic format: r1,1 r1,2 r1,3 t1 r2,1 ...
                        vals = [float(x) for x in parts[2:14]]
                        pose = np.array([
                            [vals[0], vals[1], vals[2], vals[3]],
                            [vals[4], vals[5], vals[6], vals[7]],
                            [vals[8], vals[9], vals[10], vals[11]],
                            [0, 0, 0, 1]
                        ]).astype(np.float32)
                        extrinsics_dict[frame_id] = pose

                for img in tqdm(images, desc=f"{scene}/{seq}/{cam}"):
                    # rgb_00000.jpg
                    prefix = img[:4] # "rgb_"
                    frame_num_str = img[4:9]
                    frame_idx = int(frame_num_str)
                    
                    # Target names
                    target_rgb = f"{frame_num_str}_rgb.jpg"
                    target_depth = f"{frame_num_str}_depth.png"
                    target_npz = f"{frame_num_str}_cam.npz"
                    
                    src_rgb = osp.join(raw_rgb_cam, img)
                    src_depth = osp.join(raw_depth_cam, f"depth_{frame_num_str}.png")
                    
                    dst_rgb = osp.join(out_seq_cam, target_rgb)
                    dst_depth = osp.join(out_seq_cam, target_depth)
                    dst_npz = osp.join(out_seq_cam, target_npz)
                    
                    rel_src_rgb = os.path.relpath(src_rgb, start=out_seq_cam)
                    rel_src_depth = os.path.relpath(src_depth, start=out_seq_cam)
                    
                    if not osp.exists(dst_rgb):
                        os.symlink(rel_src_rgb, dst_rgb)
                        
                    if not osp.exists(dst_depth):
                        os.symlink(rel_src_depth, dst_depth)
                        
                    if frame_idx in intrinsics_dict and frame_idx in extrinsics_dict:
                        K = intrinsics_dict[frame_idx]
                        pose = extrinsics_dict[frame_idx]
                        if not osp.exists(dst_npz):
                            np.savez(dst_npz, camera_pose=pose, camera_intrinsics=K)

if __name__ == "__main__":
    main()
