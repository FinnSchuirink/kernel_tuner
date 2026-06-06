import hashlib
import json
import os
from pathlib import Path
import threading

class CompilationCache:
    """Class to build a compilation cache that re-uses kernels, if the kernel exists

        Cache directory:
            <cache_dir>/
                index.json -> Mapping from cache key to metadata
                <cache_key>.bin -> Compiled binary
    """

    def __init__(self, cache_dir: str = "compilation_cache"):
        """Initialize the cache

            :param cache_dir: Stringified directory name for the cache, default to "compilation cache
            :type cache_dir: str

            :param index_path: Stringified path to the index file
            :type index_path: str

            :param _hits: Number of cache hits
            :type _hits: int

            :param _misses: Number of cache misses
            :type _misses: int

            :param _index: Dictionary of cache entries
            :type _index: dict
        """
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = Path(os.path.join(cache_dir, "index.json"))
        self._hits = 0
        self._misses = 0
        self._index = self._load_index()
        self._lock = threading.Lock()

    def put(self, key: str, compiled_binary: bytes, metadata: dict = None):
        """Store the compiled binary in the cache

        :param key: The unique key for the compiled binary (using make_cache_key)
        :type key: str

        :param compiled_binary: The compiled binary for the kernel
        :type compiled_binary: bytes

        :param metadata: Meta data about the file, for debugging purposes (backend, device, flags, ...)
        :type metadata: dict
        """
        with self._lock:
            cache_filename = key + ".bin"

            ## Save binary in cache
            binary_path = Path(os.path.join(self._cache_dir, cache_filename))
            binary_path.write_bytes(compiled_binary)

            ## Save cache entry with metadata in a map
            self._index[key] = {
                "filename": cache_filename,
                "size": len(compiled_binary),
                **(metadata or {})
            }

            ## Update index page
            self._write_index()
            return None
        
    def get(self, key: str):
        """Get compiled binary from the cache

        :param key: The deterministic key for the compiled binary (using make_cache_key)
        :type key: str
        """

        with self._lock:
            ## Binary not in cache
            if (key not in self._index):
                self._misses += 1
                return None
            
            ## Full path to the desired binary
            cache_entry = self._index[key]
            cache_file = cache_entry["filename"]

            binary_path = Path(os.path.join(self._cache_dir, cache_file))

            ## Key exists in cache, but file is deleted -> update _index
            if (not os.path.exists(binary_path)):
                del self._index[key]
                self._misses += 1
                return None

            ## Hit -> Get binary
            self._hits += 1
            return binary_path.read_bytes()
    
    @staticmethod
    def make_cache_key(kernel_string: str, backend: str, device: str, flags: list[str], cuda_version=None, cc=None) -> str:

        """Function that creates a uniquely identifiable hash for each different kernel
        :param kernel_string: Stringified kernel
        :type kernel_string: str

        :param backend: 'PyCuda', 'CuPY', 'OpenCL', 'NVCC'
        :type backend: str

        :param device: CUDA/OpenCL device to use
        :type device: str

        :param flags: List of compilation flags
        :type flags: list[str]

        :param cuda_version: CUDA driver version
        :type cuda_version: int

        :param cc: Compute Capabilities
        :type cc: int
        """

        # FIXME: Let user determinate which parameter is essential for recompilation (higher clock frequency)
        
        # Deterministic hashing to uniquely identify kernels
        h = hashlib.sha256()

        ## Create the hash by converting string into bytes
        h.update(kernel_string.encode())
        h.update(backend.encode())
        h.update(str(device).encode())

        # Sort flags as compiler flags are commutative
        for flag in sorted(flags or []):
            h.update(flag.encode())

        if cuda_version:
            h.update(str(cuda_version).encode())
        if cc:
            h.update(str(cc).encode())

        # Create valid indexing key 
        return h.hexdigest()
    
    def log_stats(self) -> dict:
        """Return number of hits and misses for the cache
        """
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": total
        }
    
    def _load_index(self) -> dict:
        """Load index.json file into dictionary
        """
        if (os.path.exists(self._index_path)):
            with open(self._index_path) as f:
                return json.load(f)
        return {}
    
    def _write_index(self) -> None:
        """Write dictionary to index.json file
        """
        with open(self._index_path, "w") as f:
            json.dump(self._index, f, indent=2)

        
        

        


    