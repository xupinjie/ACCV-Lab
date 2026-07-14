# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ctypes
import gc
import os
import random
import time

import numpy as np
import psutil
import pynvml
import torch


def is_diff_in_range(to_comp_1, to_comp_2, tolerance):
    # support input as YUV tuple or single channel
    if isinstance(to_comp_1, tuple) and isinstance(to_comp_2, tuple):
        results = [is_diff_in_range(c1, c2, tolerance) for c1, c2 in zip(to_comp_1, to_comp_2)]
        in_range = all(r[0] for r in results)
        max_diff = max(r[1] for r in results)
        count_in_range = sum(r[2] for r in results)
        return in_range, max_diff, count_in_range

    # support single channel
    if torch.is_tensor(to_comp_1):
        to_comp_1 = to_comp_1.cpu()
    if torch.is_tensor(to_comp_2):
        to_comp_2 = to_comp_2.cpu()
    to_comp_1 = np.array(to_comp_1)
    to_comp_2 = np.array(to_comp_2)
    use_int = issubclass(to_comp_1.dtype.type, np.integer) and issubclass(to_comp_2.dtype.type, np.integer)
    type_to_use = np.int64 if use_int else np.float64
    diffs = np.abs(np.array(to_comp_1).astype(type_to_use) - np.array(to_comp_2).astype(type_to_use))
    max_diff = diffs.max()
    in_range = max_diff <= tolerance
    count_in_range = np.sum(diffs > tolerance)
    return in_range, max_diff, count_in_range


def diff(gop_decoded, opencv_decoded, file_names, frames, num_files, diff_tolerance=21):
    for file_name, frame, i in zip(file_names, frames, range(num_files)):
        same_with_opencv_dec, max_diff, count_in_range = is_diff_in_range(
            gop_decoded[i], opencv_decoded[i], diff_tolerance
        )
        print(file_name, " frame_id: ", frame, " pass: ", same_with_opencv_dec, " max_diff: ", max_diff)
        if not same_with_opencv_dec:
            print(f"Error: {file_name} {frame} is not same with opencv_decoded")
            return max_diff
    return 0


def gop_decode_bgr(nv_gop_dec, file_path_list, frame_id_list):
    try:
        decoded_frames = nv_gop_dec.DecodeN12ToRGB(file_path_list, frame_id_list, True)
        torch.cuda.nvtx.range_push("unsqueeze_tensor_list")
        res = [torch.unsqueeze(torch.as_tensor(df), 0) for df in decoded_frames]
        torch.cuda.nvtx.range_pop()
        return res
    except Exception as e:
        print(f"Error: {e}")
        return None


def gop_decode_bgr_with_fast_init(nv_gop_dec, file_path_list, frame_id_list, fast_stream_infos):
    try:
        decoded_frames = nv_gop_dec.DecodeN12ToRGB(
            file_path_list, frame_id_list, as_bgr=True, fastStreamInfos=fast_stream_infos
        )
        torch.cuda.nvtx.range_push("unsqueeze_tensor_list")
        res = [torch.unsqueeze(torch.as_tensor(df), 0) for df in decoded_frames]
        torch.cuda.nvtx.range_pop()
        return res
    except Exception as e:
        print(f"Error: {e}")
        return None


def gop_decode_bgr_ddseparate_with_fast_init(
    nv_gop_dec1, nv_gop_dec2, file_path_list, frame_id_list, fast_stream_infos
):
    try:
        gop_list = nv_gop_dec1.GetGOPList(file_path_list, frame_id_list, fastStreamInfos=fast_stream_infos)
        gop_data_list = [data for data, _, _ in gop_list]
        decoded_frames = nv_gop_dec2.DecodeFromGOPListRGB(
            gop_data_list, file_path_list, frame_id_list, as_bgr=True
        )
        res = [torch.unsqueeze(torch.as_tensor(df), 0) for df in decoded_frames]
        return res
    except Exception as e:
        print(f"Error: {e}")
        return None


def gop_decode_bgr_ddseparate_from_multi_packets(nv_gop_dec, file_path_list, frame_id_list, packets_list):
    try:
        decoded_frames = nv_gop_dec.DecodeFromGOPListRGB(
            packets_list, file_path_list, frame_id_list, as_bgr=True
        )
        res = [torch.unsqueeze(torch.as_tensor(df), 0) for df in decoded_frames]
        return res
    except Exception as e:
        print(f"Error: {e}")
        return None


def get_data_dir():
    """
    Return absolute path to the test video data directory.

    This is resolved relative to this test package so that tests can be run
    from any current working directory.
    """
    test_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(test_root, "data")


# ============================================================================
# GPU / CPU Memory Measurement Utilities
# ============================================================================


def _query_cuda_pool(device_id: int = 0) -> tuple:
    """Return (reserved_mb, used_mb) for the CUDA default stream-ordered pool.

    Raises RuntimeError on any CUDA API failure.
    """
    libcuda = ctypes.CDLL("libcuda.so.1")
    if libcuda.cuCtxSynchronize() != 0:
        raise RuntimeError("cuCtxSynchronize failed")
    pool = ctypes.c_void_p()
    if libcuda.cuDeviceGetDefaultMemPool(ctypes.byref(pool), ctypes.c_int(device_id)) != 0:
        raise RuntimeError(f"cuDeviceGetDefaultMemPool failed for device {device_id}")
    reserved = ctypes.c_uint64(0)
    used = ctypes.c_uint64(0)
    # CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT = 5 (CUDA 11.2+)
    # CU_MEMPOOL_ATTR_USED_MEM_CURRENT     = 6 (CUDA 11.2+)
    if libcuda.cuMemPoolGetAttribute(pool, ctypes.c_int(5), ctypes.byref(reserved)) != 0:
        raise RuntimeError("cuMemPoolGetAttribute(RESERVED_MEM_CURRENT) failed")
    if libcuda.cuMemPoolGetAttribute(pool, ctypes.c_int(6), ctypes.byref(used)) != 0:
        raise RuntimeError("cuMemPoolGetAttribute(USED_MEM_CURRENT) failed")
    to_mb = 1 / (1024 * 1024)
    return reserved.value * to_mb, used.value * to_mb


class GPUMemoryMonitor:
    """GPU memory monitor backed by pynvml.

    Tracks all GPU memory (including CUDA Driver API allocations that
    torch.cuda.memory_allocated() cannot see).

    Two complementary leak checks:
    - get_pool_used_mb(): directly returns USED_MEM_CURRENT of the CUDA default
      pool. After cleanup this must be 0; a nonzero value means a cuMemAllocAsync
      allocation was never freed (lost pointer / missing cuMemFreeAsync).
    - get_used_memory_mb(): nvml total minus pool-cached-free blocks (RESERVED -
      USED). Catches leaks on non-pool paths (cuMemAlloc, cudaMalloc, etc.);
      pool-cached-free noise is subtracted so the baseline/final delta is clean.

    Use as a context manager; call force_cleanup() before each measurement.
    """

    def __init__(self, gpu_id: int = 0):
        self.gpu_id = gpu_id
        self._initialized = False

    def __enter__(self):
        pynvml.nvmlInit()
        self._initialized = True
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._initialized:
            pynvml.nvmlShutdown()
            self._initialized = False

    def get_pool_used_mb(self) -> float:
        """Return MB actively allocated from the CUDA default pool (USED_MEM_CURRENT).

        Should be 0 after cleanup. A nonzero value indicates a pool allocation
        leak (cuMemFreeAsync was never called for some cuMemAllocAsync allocation).
        """
        _, used_mb = _query_cuda_pool(self.gpu_id)
        return used_mb

    def get_used_memory_mb(self) -> float:
        """Return total GPU usage in MB, minus CUDA pool-cached free blocks.

        Subtracts (RESERVED - USED) so that memory freed into the pool but not
        yet returned to the OS does not inflate the reading. Active pool
        allocations (USED) are still counted as live, so pool leaks are visible.
        """
        info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        total_mb = info.used / (1024 * 1024)
        reserved_mb, used_mb = _query_cuda_pool(self.gpu_id)
        pool_freed_mb = max(0.0, reserved_mb - used_mb)
        return total_mb - pool_freed_mb

    def get_free_memory_mb(self) -> float:
        """Return current GPU free memory in MB."""
        info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        return info.free / (1024 * 1024)


class CPUMemoryMonitor:
    """CPU Memory Monitor using psutil."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())

    def get_rss_mb(self) -> float:
        """Get current process RSS (Resident Set Size) in MB."""
        return self.process.memory_info().rss / (1024 * 1024)


def force_cleanup():
    """Force GC and CUDA cleanup before taking a memory measurement.

    Flushes Python GC, PyTorch cache, and CUDA work so that all pending frees
    have settled before any memory reading is taken.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
    time.sleep(0.3)
    gc.collect()


def measure_memory_delta(baseline_gpu_mb: float, current_gpu_mb: float, tolerance_mb: float = 50.0) -> tuple:
    """
    Check if memory delta is within tolerance.

    Returns:
        (is_ok, delta_mb): Whether delta is within tolerance and the actual delta
    """
    delta = current_gpu_mb - baseline_gpu_mb
    return delta <= tolerance_mb, delta


def select_random_clip(path_base):
    # Only consider sample_clip* subdirs as eligible for the general random-clip
    # tests. Other data/ subdirs (e.g. pix_fmt_variants/) hold targeted fixtures
    # whose contents may not be RGB-decodable on the runtime GPU.
    subdirs = [
        d
        for d in os.listdir(path_base)
        if os.path.isdir(os.path.join(path_base, d)) and d.startswith("sample_clip")
    ]
    if not subdirs:
        return None
    clip_dir = os.path.join(path_base, random.choice(subdirs))
    video_names = os.listdir(clip_dir)
    files = [os.path.join(clip_dir, file) for file in video_names]
    return files
