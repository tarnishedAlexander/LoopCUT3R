import sys
import os

def add_path_to_dust3r(ckpt):
    """Add the correct source root to sys.path so that `import dust3r` works."""
    # Always prioritize the local source code of this repo
    local_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
    if os.path.isdir(os.path.join(local_src, "dust3r")):
        sys.path.insert(0, local_src)
        print(f"[add_path_to_dust3r] Force-added local src to sys.path: {local_src}")
        return
        
    fallback = os.path.dirname(os.path.abspath(ckpt))
    sys.path.insert(0, fallback)
    print(f"[add_path_to_dust3r] Fallback — added to sys.path: {fallback}")
