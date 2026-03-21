import hashlib

class CompilationCache:
    """Class to build a compilation cache that re-uses kernels, if applicable


    """
    ## Store kernel in to cache
    def put(self):
        x = 4

    ## Retrieve the cached kernel
    def get(self):
        x = 5
    
    @staticmethod
    def make_cache_key(
        kernel_string: str, backend: str, device: str, flags: list[str]):

        """Function that creates a uniquely identifieable hash for each different kernel
        :param kernel_string: Stringified kernel
        :type kernel_source: str

        :param backend: 'PyCuda', 'CuPY', 'OpenCL', 'NVCC'
        :type backend: str

        :param device: CUDA/OpenCL device to use
        :type device: str

        :param flags: List of compilation flags
        :type list[str]
        """
        
        # Deterministic hashing to uniquely identify kernels
        h = hashlib.sha256()

        ## Create the hash by coiverting string into bytes
        h.update(kernel_string.encode())
        h.update(backend.encode())
        h.update(device.encode())

        # Sort flags as compiler flags are associative
        for flag in sorted(flags):
            h.update(flag.encode())

        # Cretae valid indexing key 
        return h.hexdigest()
        
        
        


    