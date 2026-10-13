#!/usr/bin/env python3
import os
import json
import numpy as np
from tqdm import tqdm

def load_colmap_images(sfm_dir):
    images_txt_path = os.path.join(sfm_dir, "images.txt")
    if not os.path.exists(images_txt_path):
        return set()
    with open(images_txt_path, "r") as f:
        raw = f.read().splitlines()
        raw = [line for line in raw if not line.startswith("#")]  # skip header
    valid_names = set()
    # Each image has two lines, the first line is the image info
    for image in raw[0::2]:
        image = image.split(" ")
        img_name = image[-1]
        valid_names.add(img_name)
    return valid_names

def align_scene_pairs(scene_id, scannetpp_dir):
    data_dir = os.path.join(scannetpp_dir, "data", scene_id)
    sfm_dir_dslr = os.path.join(data_dir, "dslr", "colmap")
    sfm_dir_iphone = os.path.join(data_dir, "iphone", "colmap")
    
    # Load valid colmap reconstructed frame names
    dslr_valid = load_colmap_images(sfm_dir_dslr)
    iphone_valid = load_colmap_images(sfm_dir_iphone)
    
    pairs_dir_scene_selected_pairs = os.path.join(
        scannetpp_dir, "scannetpp_pairs", scene_id, "selected_pairs.npz"
    )
    if not os.path.exists(pairs_dir_scene_selected_pairs):
        return

    # Load original pairs
    selected_npz = np.load(pairs_dir_scene_selected_pairs)
    selection, pairs = selected_npz["selection"], selected_npz["pairs"]
    
    # Create mask of elements that exist in COLMAP reconstruction
    valid_mask = []
    for name in selection:
        if name.startswith("DSC"):
            is_valid = (name + ".JPG") in dslr_valid
        elif name.startswith("frame_"):
            is_valid = (name + ".jpg") in iphone_valid
        else:
            is_valid = False
        valid_mask.append(is_valid)
        
    # Map old selection indices to new selection indices
    old_to_new = {}
    new_selection = []
    new_idx = 0
    for old_idx, is_valid in enumerate(valid_mask):
        if is_valid:
            old_to_new[old_idx] = new_idx
            new_selection.append(selection[old_idx])
            new_idx += 1
            
    # Filter and re-index the pairs matrix
    new_pairs_list = []
    for p in pairs:
        idx1, idx2, score = p[0], p[1], p[2]
        if int(idx1) in old_to_new and int(idx2) in old_to_new:
            new_pairs_list.append([old_to_new[int(idx1)], old_to_new[int(idx2)], score])
            
    if new_pairs_list:
        new_pairs = np.array(new_pairs_list, dtype=pairs.dtype)
    else:
        new_pairs = np.empty((0, 3), dtype=pairs.dtype)
        
    # Write back the aligned selection and pairs
    np.savez(
        pairs_dir_scene_selected_pairs,
        selection=np.array(new_selection, dtype=selection.dtype),
        pairs=new_pairs
    )
    
    num_removed = len(selection) - len(new_selection)
    if num_removed > 0:
        print(f"Scene {scene_id}: Removed {num_removed} unregistered frames from selection.")

def main():
    scannetpp_dir = "/app/recurrentTransformer/dataset/disk12tb/scannet/scannetpp"
    scene_list_path = os.path.join(scannetpp_dir, "scannetpp_pairs", "scene_list.json")
    
    with open(scene_list_path, "r") as f:
        scenes = json.load(f)
        
    print(f"Aligning precomputed pairs for {len(scenes)} scenes...")
    for scene_id in tqdm(scenes):
        align_scene_pairs(scene_id, scannetpp_dir)
    print("Alignment complete!")

if __name__ == "__main__":
    main()
