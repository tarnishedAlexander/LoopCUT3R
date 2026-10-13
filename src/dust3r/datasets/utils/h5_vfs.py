# HDF5 Virtual File System (VFS) Interceptor for LoopCUT3R
# Transparently routes cv2.imread and np.load to compressed H5 archives.

import os
import time
import h5py
import cv2
import numpy as np

class H5NpzWrapper:
    """Wrapper that mimics numpy's NpzFile structure for camera parameters."""
    def __init__(self, group):
        self.files = list(group.keys())
        self._data = {}
        for k in self.files:
            dataset = group[k]
            # Check if it was pickled object (void dtype)
            if dataset.dtype.kind == 'V':
                import pickle
                data = pickle.loads(dataset[()].tobytes())
            else:
                if dataset.shape == ():
                    data = dataset[()]
                else:
                    data = dataset[:]
                # Restore to float32 if compressed as float16
                if data.dtype == np.float16:
                    data = data.astype(np.float32)
                # Convert bytes to string if it was encoded as string/bytes
                elif data.dtype.kind == 'S':
                    data = np.char.decode(data, 'utf-8')
            self._data[k] = data

    def __getitem__(self, key):
        if key not in self._data:
            raise KeyError(f"Key '{key}' not found in H5 NPZ group.")
        return self._data[key]

    def keys(self):
        return self.files

    def __iter__(self):
        return iter(self.files)

    def __contains__(self, key):
        return key in self._data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def close(self):
        pass


class H5VFSRegistry:
    _registries = []  # List of tuples: (dataset_root, h5_path)
    _handles = {}     # Process-local cache of open h5py.File handles: (pid, h5_path) -> h5py.File
    
    # Process-local lazy caches for O(1) repeated lookups
    _exists_cache = {}   # (h5_path, rel_path) -> bool
    _isfile_cache = {}   # (h5_path, rel_path) -> bool
    _isdir_cache = {}    # (h5_path, rel_path) -> bool
    _listdir_cache = {}  # (h5_path, rel_path) -> list of strings

    _original_imread = None
    _original_load = None
    _original_open = None
    _original_exists = None
    _original_isfile = None
    _original_isdir = None
    _original_listdir = None
    _patched = False

    @classmethod
    def register(cls, dataset_root, h5_path):
        dataset_root = os.path.abspath(dataset_root)
        h5_path = os.path.abspath(h5_path)
        
        # Check if already registered
        for root, path in cls._registries:
            if root == dataset_root:
                return

        cls._registries.append((dataset_root, h5_path))
        print(f"[H5VFS] Registered dataset root: {dataset_root} -> {h5_path}")
        
        # Ensure patched
        cls.patch()

    @classmethod
    def get_handle(cls, h5_path):
        pid = os.getpid()
        key = (pid, h5_path)
        if key not in cls._handles:
            orig_exists = cls._original_exists if cls._original_exists else os.path.exists
            if orig_exists(h5_path):
                try:
                    # Dynamically tune the HDF5 Raw Data Chunk Cache (rdcc) to maximize performance.
                    # This is crucial for HDD storage where chunk cache thrashing causes head thrashing.
                    if "disk12tb" in h5_path:
                        # HDD: 128 MB cache for active scene chunks (balanced memory vs perf)
                        cls._handles[key] = h5py.File(
                            h5_path, 
                            "r", 
                            rdcc_nbytes=128 * 1024 * 1024, 
                            rdcc_nslots=100003, 
                            rdcc_w0=0.0
                        )
                    else:
                        # SSD/NVMe: 16 MB cache is sufficient for fast random access devices
                        cls._handles[key] = h5py.File(
                            h5_path, 
                            "r", 
                            rdcc_nbytes=16 * 1024 * 1024, 
                            rdcc_nslots=10007, 
                            rdcc_w0=0.0
                        )
                except Exception as e:
                    print(f"[H5VFS] Error opening H5 file {h5_path}: {e}")
                    cls._handles[key] = None
            else:
                cls._handles[key] = None
        return cls._handles[key]

    @classmethod
    def patch(cls):
        if cls._patched:
            return

        import cv2
        import numpy as np
        import builtins

        cls._original_imread = cv2.imread
        cv2.imread = imread_wrapper

        cls._original_load = np.load
        np.load = load_wrapper

        cls._original_open = builtins.open
        builtins.open = open_wrapper

        cls._original_exists = os.path.exists
        os.path.exists = exists_wrapper

        cls._original_isfile = os.path.isfile
        os.path.isfile = isfile_wrapper

        cls._original_isdir = os.path.isdir
        os.path.isdir = isdir_wrapper

        cls._original_listdir = os.listdir
        os.listdir = listdir_wrapper

        cls._patched = True
        print("[H5VFS] Dynamic filesystem hooks successfully activated (lazy-caching mode).")


def exists_wrapper(path):
    if not isinstance(path, str):
        try:
            path = str(path)
        except Exception:
            path = ""
    if path:
        path_abs = os.path.abspath(path)
        for root, h5_path in H5VFSRegistry._registries:
            if path_abs == root or path_abs.startswith(root + '/'):
                rel_path = os.path.relpath(path_abs, root).replace("\\", "/")
                if rel_path == '.':
                    return True
                cache_key = (h5_path, rel_path)
                if cache_key in H5VFSRegistry._exists_cache:
                    return H5VFSRegistry._exists_cache[cache_key]
                handle = H5VFSRegistry.get_handle(h5_path)
                res = False
                if handle is not None:
                    res = rel_path in handle
                if not res:
                    # Fallback to physical disk check
                    res = H5VFSRegistry._original_exists(path)
                H5VFSRegistry._exists_cache[cache_key] = res
                return res
    return H5VFSRegistry._original_exists(path)


def isfile_wrapper(path):
    if not isinstance(path, str):
        try:
            path = str(path)
        except Exception:
            path = ""
    if path:
        path_abs = os.path.abspath(path)
        for root, h5_path in H5VFSRegistry._registries:
            if path_abs == root or path_abs.startswith(root + '/'):
                rel_path = os.path.relpath(path_abs, root).replace("\\", "/")
                cache_key = (h5_path, rel_path)
                if cache_key in H5VFSRegistry._isfile_cache:
                    return H5VFSRegistry._isfile_cache[cache_key]
                handle = H5VFSRegistry.get_handle(h5_path)
                res = False
                if handle is not None:
                    try:
                        obj = handle[rel_path]
                        res = isinstance(obj, h5py.Dataset) or rel_path.endswith(".npz")
                    except KeyError:
                        res = False
                if not res:
                    # Fallback to physical disk check
                    res = H5VFSRegistry._original_isfile(path)
                H5VFSRegistry._isfile_cache[cache_key] = res
                return res
    return H5VFSRegistry._original_isfile(path)


def isdir_wrapper(path):
    if not isinstance(path, str):
        try:
            path = str(path)
        except Exception:
            path = ""
    if path:
        path_abs = os.path.abspath(path)
        for root, h5_path in H5VFSRegistry._registries:
            if path_abs == root or path_abs.startswith(root + '/'):
                rel_path = os.path.relpath(path_abs, root).replace("\\", "/")
                if rel_path == '.':
                    return True
                cache_key = (h5_path, rel_path)
                if cache_key in H5VFSRegistry._isdir_cache:
                    return H5VFSRegistry._isdir_cache[cache_key]
                handle = H5VFSRegistry.get_handle(h5_path)
                res = False
                if handle is not None:
                    try:
                        obj = handle[rel_path]
                        res = isinstance(obj, h5py.Group) and not rel_path.endswith(".npz")
                    except KeyError:
                        res = False
                if not res:
                    # Fallback to physical disk check
                    res = H5VFSRegistry._original_isdir(path)
                H5VFSRegistry._isdir_cache[cache_key] = res
                return res
    return H5VFSRegistry._original_isdir(path)


def listdir_wrapper(path):
    if not isinstance(path, str):
        try:
            path = str(path)
        except Exception:
            path = ""
    if path:
        path_abs = os.path.abspath(path)
        for root, h5_path in H5VFSRegistry._registries:
            if path_abs == root or path_abs.startswith(root + '/'):
                rel_path = os.path.relpath(path_abs, root).replace("\\", "/")
                
                cache_key = (h5_path, rel_path)
                if cache_key in H5VFSRegistry._listdir_cache:
                    return H5VFSRegistry._listdir_cache[cache_key]
                    
                handle = H5VFSRegistry.get_handle(h5_path)
                res = []
                if handle is not None:
                    if rel_path == '.':
                        res = list(handle.keys())
                    elif rel_path in handle:
                        obj = handle[rel_path]
                        if isinstance(obj, h5py.Group):
                            res = list(obj.keys())
                # Fallback: merge with files present physically on disk
                try:
                    phys_res = H5VFSRegistry._original_listdir(path)
                    for item in phys_res:
                        if item not in res:
                            res.append(item)
                except Exception:
                    pass
                H5VFSRegistry._listdir_cache[cache_key] = res
                return res
    return H5VFSRegistry._original_listdir(path)


def open_wrapper(file, mode='r', *args, **kwargs):
    if not isinstance(file, str):
        try:
            filename = str(file)
        except Exception:
            filename = ""
    else:
        filename = file

    if filename:
        filename_abs = os.path.abspath(filename)
        for root, h5_path in H5VFSRegistry._registries:
            if filename_abs == root or filename_abs.startswith(root + '/'):
                rel_path = os.path.relpath(filename_abs, root).replace("\\", "/")
                handle = H5VFSRegistry.get_handle(h5_path)
                if handle is not None:
                    try:
                        obj = handle[rel_path]
                        if isinstance(obj, h5py.Dataset):
                            bytes_data = obj[()].tobytes()
                            import io
                            if 'b' in mode:
                                return io.BytesIO(bytes_data)
                            else:
                                encoding = kwargs.get('encoding', 'utf-8') or 'utf-8'
                                return io.StringIO(bytes_data.decode(encoding))
                    except KeyError:
                        pass
                    except Exception as e:
                        print(f"[H5VFS] Error reading text/raw file {rel_path} from H5: {e}")
                        
    return H5VFSRegistry._original_open(file, mode, *args, **kwargs)


def imread_wrapper(filename, flags=cv2.IMREAD_COLOR):
    if not isinstance(filename, str):
        filename = str(filename)
    
    filename_abs = os.path.abspath(filename)
    for root, h5_path in H5VFSRegistry._registries:
        if filename_abs == root or filename_abs.startswith(root + '/'):
            rel_path = os.path.relpath(filename_abs, root).replace("\\", "/")
            handle = H5VFSRegistry.get_handle(h5_path)
            if handle is not None:
                try:
                    obj = handle[rel_path]
                    
                    # Direct numpy array retrieval (e.g. pre-decompressed depth map or mask)
                    if len(obj.shape) > 1:
                        img = obj[:]
                        if img.dtype == np.float16:
                            img = img.astype(np.float32)
                        return img
                    
                    # Retrieve the byte array from HDF5
                    if obj.shape == ():
                        data_bytes = np.frombuffer(obj[()].tobytes(), dtype=np.uint8)
                    else:
                        data_bytes = np.frombuffer(obj[:], dtype=np.uint8)
                    img = cv2.imdecode(data_bytes, flags)
                    if img is not None:
                        return img
                except KeyError:
                    pass
                except Exception as e:
                    print(f"[H5VFS] Error decoding image {rel_path} from H5: {e}")
                    
    # Fallback to disk read
    return H5VFSRegistry._original_imread(filename, flags)


def load_wrapper(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII'):
    if not isinstance(file, str):
        try:
            filename = str(file)
        except Exception:
            filename = ""
    else:
        filename = file

    if filename:
        filename_abs = os.path.abspath(filename)
        for root, h5_path in H5VFSRegistry._registries:
            if filename_abs == root or filename_abs.startswith(root + '/'):
                rel_path = os.path.relpath(filename_abs, root).replace("\\", "/")
                handle = H5VFSRegistry.get_handle(h5_path)
                if handle is not None:
                    try:
                        obj = handle[rel_path]
                        if isinstance(obj, h5py.Group):
                            # Camera parameters (NPZ group)
                            return H5NpzWrapper(obj)
                        elif isinstance(obj, h5py.Dataset):
                            # Check if it is a raw npy file stored as void bytes
                            if obj.dtype.kind == 'V':
                                import io
                                bytes_data = obj[()].tobytes()
                                try:
                                    return np.load(io.BytesIO(bytes_data), allow_pickle=True)
                                except Exception as e:
                                    print(f"[H5VFS] Error decoding raw .npy {rel_path}: {e}. Falling back to disk.")
                                    break  # Fall through to physical disk read
                            
                            # Standard depth maps or numpy arrays (NPY dataset)
                            if obj.shape == ():
                                arr = obj[()]
                            else:
                                arr = obj[:]
                            if arr.dtype == np.float16:
                                arr = arr.astype(np.float32)
                            return arr
                    except KeyError:
                        pass

    # Fallback to disk read
    return H5VFSRegistry._original_load(file, mmap_mode=mmap_mode, allow_pickle=allow_pickle,
                                         fix_imports=fix_imports, encoding=encoding)
