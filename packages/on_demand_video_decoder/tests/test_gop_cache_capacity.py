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

import pytest

import accvlab.on_demand_video_decoder as nvc
import utils


def test_gop_cache_lru_capacity():
    path_base = utils.get_data_dir()
    files = utils.select_random_clip(path_base)
    if files is None:
        pytest.skip("No test video files available")
    if len(files) < 3:
        pytest.skip("Need at least 3 files for LRU cache test")

    decoder = nvc.CreateGopDecoder(maxfiles=6, iGpu=0, gopCacheCapacity=2)

    decoder.GetGOPList([files[0]], [10], useGOPCache=True)
    decoder.GetGOPList([files[1]], [10], useGOPCache=True)

    cache_info = decoder.get_cache_info()
    assert cache_info["cache_capacity"] == 2
    assert cache_info["cached_files_count"] == 2
    assert files[0] in cache_info["cached_files"]
    assert files[1] in cache_info["cached_files"]

    first_file_info = cache_info["cached_files"][files[0]]
    decoder.GetGOPList([files[0]], [first_file_info["first_frame_id"]], useGOPCache=True)
    assert decoder.isCacheHit() == [True]

    decoder.GetGOPList([files[2]], [10], useGOPCache=True)

    cache_info = decoder.get_cache_info()
    assert cache_info["cached_files_count"] == 2
    assert files[0] in cache_info["cached_files"]
    assert files[2] in cache_info["cached_files"]
    assert files[1] not in cache_info["cached_files"]
