import os
import sys
import time
import pickle
import h5py

def build_cache_for_h5(h5_path):
    print(f"\n==================================================")
    print(f"Building cache for {h5_path}...")
    print(f"==================================================")
    
    cache_bin_path = h5_path + ".cache.pkl"
    start_t = time.time()
    
    files_set = set()
    dirs_set = set()
    contents_dict = {}
    
    # Register root
    dirs_set.add(".")
    contents_dict["."] = []
    
    f = h5py.File(h5_path, "r")
    
    # Iterative DFS with a stack of (rel_path, group_obj)
    # Optimised by using extension-based leaf detection to avoid any dataset opening.
    stack = [(".", f)]
    group_count = 0
    file_count = 0
    
    last_print = time.time()
    
    while stack:
        parent_path, parent_node = stack.pop()
        
        try:
            keys = list(parent_node.keys())
        except Exception as e:
            print(f"Error reading keys for {parent_path}: {e}")
            continue
            
        contents_dict[parent_path] = keys
        
        for k in keys:
            child_path = k if parent_path == "." else f"{parent_path}/{k}"
            
            # If the item ends with standard extensions, it's a leaf file dataset.
            if k.endswith((".png", ".jpg", ".jpeg", ".npy", ".npz")):
                files_set.add(child_path)
                file_count += 1
            else:
                # It is a directory group
                dirs_set.add(child_path)
                try:
                    child_node = parent_node[k]
                    stack.append((child_path, child_node))
                except Exception as e:
                    print(f"Error accessing group {child_path}: {e}")
                        
        group_count += 1
        if group_count % 500 == 0 or time.time() - last_print > 5:
            print(f"Progress: Traversed {group_count} directories, found {file_count} files... Elapsed: {time.time() - start_t:.1f}s")
            sys.stdout.flush()
            last_print = time.time()
            
    f.close()
    
    cache_data = {
        "files": files_set,
        "dirs": dirs_set,
        "contents": contents_dict
    }
    
    print(f"Saving path cache to disk for {h5_path}...")
    with open(cache_bin_path, "wb") as f_out:
        pickle.dump(cache_data, f_out, protocol=pickle.HIGHEST_PROTOCOL)
        
    duration = time.time() - start_t
    print(f"SUCCESS: Cache build complete for {h5_path} in {duration:.2f} seconds. Files: {len(files_set)}, Directories: {len(dirs_set)}")
    sys.stdout.flush()
    return cache_data

if __name__ == "__main__":
    h5_files = [
        "/app/recurrentTransformer/dataset/disk12tb/processed_dl3dv.h5",
        "/app/recurrentTransformer/dataset/disk12tb/processed_scannet.h5",
        "/app/recurrentTransformer/dataset/disk12tb/processed_TartanAir.h5"
    ]
    
    for path in h5_files:
        if os.path.exists(path):
            build_cache_for_h5(path)
        else:
            print(f"File not found: {path}")
