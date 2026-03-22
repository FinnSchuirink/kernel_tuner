import json
import os
import pytest
from pathlib import Path
from kernel_tuner.compilation_cache import CompilationCache

KERNEL_ONE = {
    "kernel_string" : """
    __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }
    """,
    "backend": "pycuda",
    "device":  "NVIDIA A1000",
    "flags": ["-Wall", "-O2", "-std=c++17"]
}

KERNEL_TWO = {
    "kernel_string" : """
        __global__ void add(float* a) { a[0] += 1; }
    """,
    "backend": "pycuda",
    "device":  "NVIDIA A1000",
    "flags": ["-O3", "-arch=sm_80"]
}

@pytest.fixture
def cache():
    return CompilationCache()

@pytest.fixture
def key_one():
    return CompilationCache.make_cache_key(
        kernel_string = KERNEL_ONE["kernel_string"],
        backend = KERNEL_ONE["backend"],
        device = KERNEL_ONE["device"],
        flags = KERNEL_ONE["flags"]
    )

@pytest.fixture
def key_two():
    return CompilationCache.make_cache_key(
        kernel_string = KERNEL_TWO["kernel_string"],
        backend = KERNEL_TWO["backend"],
        device = KERNEL_TWO["device"],
        flags = KERNEL_TWO["flags"]
    )

TEST_BINARY_ONE = b"Lorem ipsum"
TEST_BINARY_TWO = b"Dolor sit amet"

def test_miss_on_empty_cache(cache, key_one):
    assert cache.get(key_one) is None

def test_hit_after_put(cache, key_one):
    cache.put(key_one, TEST_BINARY_ONE)
    assert cache.get(key_one) == TEST_BINARY_ONE

def test_miss_after_different_put(cache, key_one, key_two):
    cache.put(key_one, TEST_BINARY_ONE)
    cache.put(key_two, TEST_BINARY_TWO)

    assert cache.get(key_one) == TEST_BINARY_ONE
    assert cache.get(key_two) == TEST_BINARY_TWO

def test_deterministic_key_generation(key_one):
    same_key = CompilationCache.make_cache_key(        
        kernel_string = KERNEL_ONE["kernel_string"],
        backend = KERNEL_ONE["backend"],
        device = KERNEL_ONE["device"],
        flags = KERNEL_ONE["flags"]
    )
    assert key_one == same_key

def test_slightly_different_kernel_different_key(key_one):
    diff_key = CompilationCache.make_cache_key(
        kernel_string = KERNEL_ONE["kernel_string"] + "a",
        backend = KERNEL_ONE["backend"],
        device = KERNEL_ONE["device"],
        flags = KERNEL_ONE["flags"]
    )
    assert diff_key != key_one

def test_different_backend_different_key(key_one):
    diff_kernel_key = CompilationCache.make_cache_key(
        kernel_string = KERNEL_ONE["kernel_string"],
        backend = "opencl",
        device = KERNEL_ONE["device"],
        flags = KERNEL_ONE["flags"]
    )
    assert diff_kernel_key != key_one

def test_different_device_different_key(key_one):
    diff_device_key = CompilationCache.make_cache_key(
        kernel_string = KERNEL_ONE["kernel_string"],
        backend = KERNEL_ONE["backend"],
        device = "NVIDIA A400",
        flags = KERNEL_ONE["flags"]
    )
    assert key_one != diff_device_key

def test_associativity_compiler_flags(key_one):
    diff_flag_key = CompilationCache.make_cache_key(
        kernel_string = KERNEL_ONE["kernel_string"],
        backend = KERNEL_ONE["backend"],
        device = KERNEL_ONE["device"],
        flags = ["-O2", "-std=c++17", "-Wall"]
    )
    assert diff_flag_key == key_one

def test_empty_compiler_flags(cache):
    empty_flags = CompilationCache.make_cache_key(
        kernel_string = KERNEL_ONE["kernel_string"],
        backend = KERNEL_ONE["backend"],
        device = "NVIDIA A400",
        flags = []
    )

    none_flags = CompilationCache.make_cache_key(
        kernel_string = KERNEL_ONE["kernel_string"],
        backend = KERNEL_ONE["backend"],
        device = "NVIDIA A400",
        flags = None
    )

    assert none_flags == empty_flags

def test_get_with_manually_deleted_binary(cache, key_one):
    cache.put(key_one, TEST_BINARY_ONE)

    binary_path = Path(cache._cache_dir, cache._index[key_one]["filename"])  
    os.remove(binary_path)

    cache.get(key_one) is None

