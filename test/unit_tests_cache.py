import json
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
    "device":  "NVIDIA A100",
    "flags": ["-Wall", "-O2", "-std=c++17"]
}

KERNEL_TWO = {
    "kernel_string" : """
        __global__ void add(float* a) { a[0] += 1; }
    """,
    "backend": "pycuda",
    "device":  "NVIDIA A100",
    "flags": ["-O3", "-arch=sm_80"]
}

@pytest.fixture
def cache():
    return CompilationCache()

@pytest.fixture
def key_one(cache):
    return CompilationCache.make_cache_key(
        kernel_string = KERNEL_ONE["kernel_string"],
        backend =KERNEL_ONE["backend"],
        device = KERNEL_ONE["device"],
        flags = KERNEL_ONE["flags"]
    )

@pytest.fixture
def key_two(cache):
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