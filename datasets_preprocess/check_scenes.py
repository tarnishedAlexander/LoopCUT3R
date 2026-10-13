#!/usr/bin/env python3
import os
import json

def main():
    root = "/app/recurrentTransformer/dataset/disk12tb/scannet/scannetpp"
    pairsdir = "/app/recurrentTransformer/dataset/disk12tb/scannet/scannetpp/scannetpp_pairs"
    
    listfile = os.path.join(pairsdir, "scene_list.json")
    with open(listfile, "r") as f:
        scenes = json.load(f)
        
    print(f"Loaded {len(scenes)} scenes.")
    
    missing_count = 0
    for idx, scene in enumerate(scenes):
        data_dir = os.path.join(root, "data", scene)
        dir_dslr = os.path.join(data_dir, "dslr")
        dir_iphone = os.path.join(data_dir, "iphone")
        dir_scans = os.path.join(data_dir, "scans")
        
        ok = (os.path.isdir(data_dir) and 
              os.path.isdir(dir_dslr) and 
              os.path.isdir(dir_iphone) and 
              os.path.isdir(dir_scans))
              
        if not ok:
            print(f"Scene {idx}: {scene} FAILED assertion!")
            print(f"  data_dir ({os.path.isdir(data_dir)}): {data_dir}")
            print(f"  dir_dslr ({os.path.isdir(dir_dslr)}): {dir_dslr}")
            print(f"  dir_iphone ({os.path.isdir(dir_iphone)}): {dir_iphone}")
            print(f"  dir_scans ({os.path.isdir(dir_scans)}): {dir_scans}")
            missing_count += 1
            
    print(f"Total missing: {missing_count}")

if __name__ == "__main__":
    main()
