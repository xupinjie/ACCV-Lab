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
import sys

import accvlab.on_demand_video_decoder as nvc


def test_pynvgopdecoder_rejects_direct_construction():
    """PyNvGopDecoder must be created via CreateGopDecoder; direct construction raises."""
    with pytest.raises(TypeError):
        nvc.PyNvGopDecoder()
    with pytest.raises(TypeError):
        nvc.PyNvGopDecoder(1, 0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
