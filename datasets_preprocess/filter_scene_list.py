#!/usr/bin/env python3
import os
import json
import shutil

def main():
    # Dynamically find the recurrentTransformer root directory relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    loopcut3r_dir = os.path.dirname(script_dir)
    recurrent_transformer_dir = os.path.dirname(loopcut3r_dir)
    
    scannetpp_dir = os.path.join(recurrent_transformer_dir, "dataset/disk12tb/scannet/scannetpp")
    pairs_dir = os.path.join(scannetpp_dir, "scannetpp_pairs")
    scene_list_path = os.path.join(pairs_dir, "scene_list.json")
    backup_path = os.path.join(pairs_dir, "scene_list.json.bak")
    
    print(f"Detected scannetpp directory: {scannetpp_dir}")
    
    # 1. Back up original file if not already backed up
    if not os.path.exists(backup_path):
        print(f"Creating backup of scene_list.json at {backup_path}")
        shutil.copy2(scene_list_path, backup_path)
    else:
        print(f"Backup already exists at {backup_path}")
        
    # 2. Read scene list
    with open(scene_list_path, "r") as f:
        scenes = json.load(f)
        
    print(f"Original scene list contains {len(scenes)} scenes.")
    
    # 3. Filter scenes based on availability of directories
    filtered_scenes = []
    missing_scenes = []
    
    for scene in scenes:
        data_dir = os.path.join(scannetpp_dir, "data", scene)
        dir_dslr = os.path.join(data_dir, "dslr")
        dir_iphone = os.path.join(data_dir, "iphone")
        dir_scans = os.path.join(data_dir, "scans")
        
        # Check if directories exist
        if (os.path.isdir(data_dir) and 
            os.path.isdir(dir_dslr) and 
            os.path.isdir(dir_iphone) and 
            os.path.isdir(dir_scans)):
            filtered_scenes.append(scene)
        else:
            reason = []
            if not os.path.isdir(data_dir):
                reason.append("missing scene folder")
            else:
                if not os.path.isdir(dir_dslr): reason.append("missing dslr")
                if not os.path.isdir(dir_iphone): reason.append("missing iphone")
                if not os.path.isdir(dir_scans): reason.append("missing scans")
            missing_scenes.append((scene, ", ".join(reason)))
            
    print(f"\nFiltered out {len(missing_scenes)} scenes due to missing directories:")
    for scene, reason in missing_scenes:
        print(f"  - {scene}: {reason}")
        
    print(f"\nKeeping {len(filtered_scenes)} scenes.")
    
    # 4. Write filtered scene list back
    with open(scene_list_path, "w") as f:
        json.dump(filtered_scenes, f)
    print(f"\nSuccessfully overwrote {scene_list_path} with {len(filtered_scenes)} scenes.")

if __name__ == "__main__":
    main()
