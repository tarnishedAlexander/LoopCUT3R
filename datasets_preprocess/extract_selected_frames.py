#!/usr/bin/env python3
import os
import re
import json
import shutil
import tempfile
import subprocess
import numpy as np
from tqdm import tqdm

def run_command(cmd):
    # Execute a shell command and check for errors
    res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\nError: {res.stderr.decode('utf-8')}")

def extract_frames_for_scene(scene_id, scannetpp_dir):
    scene_dir = os.path.join(scannetpp_dir, "data", scene_id)
    iphone_dir = os.path.join(scene_dir, "iphone")
    
    # Check if this scene has iPhone data
    mkv_path = os.path.join(iphone_dir, "rgb.mkv")
    mask_mkv_path = os.path.join(iphone_dir, "rgb_mask.mkv")
    
    if not (os.path.exists(mkv_path) and os.path.exists(mask_mkv_path)):
        print(f"Skipping scene {scene_id}: No iPhone MKV videos found.")
        return

    # Load the selected pairs to find which frames we actually need
    pairs_dir_scene = os.path.join(scannetpp_dir, "scannetpp_pairs", scene_id)
    selected_pairs_path = os.path.join(pairs_dir_scene, "selected_pairs.npz")
    
    if not os.path.exists(selected_pairs_path):
        print(f"Skipping scene {scene_id}: No selected_pairs.npz found.")
        return
        
    selected_npz = np.load(selected_pairs_path)
    selection = selected_npz["selection"]
    
    # Filter selection for iPhone frames (names starting with 'frame_')
    selected_iphone_frames = [f for f in selection if f.startswith("frame_")]
    if not selected_iphone_frames:
        print(f"Scene {scene_id} has no selected iPhone frames.")
        return

    # Parse frame indices from frame names, e.g., 'frame_004320' -> 4320
    frame_indices = sorted([int(f.split("_")[-1]) for f in selected_iphone_frames])
    
    output_rgb_dir = os.path.join(iphone_dir, "rgb")
    output_mask_dir = os.path.join(iphone_dir, "rgb_masks")
    
    # Check if all selected files already exist
    all_rgb_exist = all(os.path.exists(os.path.join(output_rgb_dir, f"{f}.jpg")) for f in selected_iphone_frames)
    all_masks_exist = all(os.path.exists(os.path.join(output_mask_dir, f"{f}.png")) for f in selected_iphone_frames)
    
    if all_rgb_exist and all_masks_exist:
        print(f"Scene {scene_id}: All {len(selected_iphone_frames)} frames already extracted. Skipping.")
        return

    print(f"\nProcessing scene: {scene_id} ({len(frame_indices)} selected iPhone frames)")
    
    # Create temporary directories for ffmpeg outputs
    tmp_rgb_dir = os.path.join(iphone_dir, "rgb_tmp")
    tmp_mask_dir = os.path.join(iphone_dir, "rgb_masks_tmp")
    os.makedirs(tmp_rgb_dir, exist_ok=True)
    os.makedirs(tmp_mask_dir, exist_ok=True)
    
    # 1. Create the ffmpeg filter script string
    # E.g. select='eq(n,4320)+eq(n,4322)+...'
    filter_str = "select='" + "+".join([f"eq(n,{idx})" for idx in frame_indices]) + "'"
    
    # Write to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', dir='.', suffix='.txt', delete=False) as tmp_file:
        tmp_file.write(filter_str)
    tmp_filter_path = tmp_file.name

    try:
        # 2. Extract RGB JPEGs
        if not all_rgb_exist:
            os.makedirs(output_rgb_dir, exist_ok=True)
            print(f"-> Extracting RGB frames from {mkv_path}...")
            cmd = f"ffmpeg -y -i {mkv_path} -filter_script:v {tmp_filter_path} -vsync vfr -q:v 1 {tmp_rgb_dir}/frame_%06d.jpg"
            run_command(cmd)
            
            # Map and rename temporary sequential output back to their original frame indices
            extracted_rgb_files = sorted(
                [os.path.join(tmp_rgb_dir, f) for f in os.listdir(tmp_rgb_dir) if f.startswith("frame_")],
                key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0])
            )
            
            if len(extracted_rgb_files) != len(frame_indices):
                print(f"Warning: Expected {len(frame_indices)} RGB frames but extracted {len(extracted_rgb_files)}")
            
            for file_path, idx in zip(extracted_rgb_files, frame_indices):
                target_name = f"frame_{idx:06d}.jpg"
                shutil.move(file_path, os.path.join(output_rgb_dir, target_name))
                
        # 3. Extract Mask PNGs
        if not all_masks_exist:
            os.makedirs(output_mask_dir, exist_ok=True)
            print(f"-> Extracting Mask frames from {mask_mkv_path}...")
            cmd = f"ffmpeg -y -i {mask_mkv_path} -pix_fmt gray -filter_script:v {tmp_filter_path} -vsync vfr {tmp_mask_dir}/frame_%06d.png"
            run_command(cmd)
            
            # Map and rename temporary sequential output back to their original frame indices
            extracted_mask_files = sorted(
                [os.path.join(tmp_mask_dir, f) for f in os.listdir(tmp_mask_dir) if f.startswith("frame_")],
                key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0])
            )
            
            if len(extracted_mask_files) != len(frame_indices):
                print(f"Warning: Expected {len(frame_indices)} mask frames but extracted {len(extracted_mask_files)}")
                
            for file_path, idx in zip(extracted_mask_files, frame_indices):
                target_name = f"frame_{idx:06d}.png"
                shutil.move(file_path, os.path.join(output_mask_dir, target_name))
                
        print(f"Successfully extracted frames for scene {scene_id}!")
        
    finally:
        # Cleanup
        if os.path.exists(tmp_filter_path):
            os.remove(tmp_filter_path)
        shutil.rmtree(tmp_rgb_dir, ignore_errors=True)
        shutil.rmtree(tmp_mask_dir, ignore_errors=True)

def main():
    scannetpp_dir = "/app/recurrentTransformer/dataset/disk12tb/scannet/scannetpp"
    scene_list_path = os.path.join(scannetpp_dir, "scannetpp_pairs", "scene_list.json")
    
    with open(scene_list_path, "r") as f:
        scenes = json.load(f)
        
    print(f"Loaded {len(scenes)} scenes to check for iPhone extraction.")
    for scene_id in tqdm(scenes, desc="Overall Progress"):
        try:
            extract_frames_for_scene(scene_id, scannetpp_dir)
        except Exception as e:
            print(f"Error processing scene {scene_id}: {e}")

if __name__ == "__main__":
    main()
