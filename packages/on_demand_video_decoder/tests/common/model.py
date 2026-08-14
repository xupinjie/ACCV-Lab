# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Small data model used by the common decoder tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Tuple

import torch


@dataclass(frozen=True)
class DecodeCase:
    """A common decode request: one frame index for every video entry."""

    name: str
    videos: Tuple[str, ...]
    frame_indices: Tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.videos) != len(self.frame_indices):
            raise ValueError("videos and frame_indices must have the same length")
        if not self.videos:
            raise ValueError("a common decode case must not be empty")


@dataclass(frozen=True)
class InvalidDecodeCase:
    """A malformed request that must reach the adapter unchanged."""

    name: str
    videos: Tuple[Any, ...]
    frame_indices: Tuple[Any, ...]


@dataclass(frozen=True)
class ResourceDecodeCase:
    """One committed fixture and its exact normalized output contract."""

    name: str
    video: str
    frame_index: int
    preferred_output_format: Literal["rgb", "yuv"]
    expected_format: str
    expected_dtype: torch.dtype
    expected_plane_shapes: Tuple[Tuple[int, ...], ...]


@dataclass(frozen=True)
class OutputContract:
    """Properties shared by every frame returned through one adapter."""

    kind: str
    dtypes: Tuple[torch.dtype, ...]
    device_type: str = "cuda"


@dataclass(frozen=True)
class CanonicalFrame:
    """Owned tensors used to compare results from otherwise different APIs."""

    format: str
    planes: Tuple[torch.Tensor, ...]
    width: int
    height: int

    @property
    def dtype(self) -> torch.dtype:
        return self.planes[0].dtype

    @property
    def device(self) -> torch.device:
        if not self.planes:
            raise AssertionError("a canonical frame must expose at least one plane")
        devices = {plane.device for plane in self.planes}
        if len(devices) != 1:
            device_names = sorted(str(device) for device in devices)
            raise AssertionError(f"all planes must be on the same device, got {device_names}")
        return self.planes[0].device

    @property
    def device_type(self) -> str:
        return self.device.type


def assert_frames_equal(actual: CanonicalFrame, expected: CanonicalFrame) -> None:
    """Assert exact equality and provide a useful plane-level failure."""

    assert actual.format == expected.format
    assert actual.width == expected.width
    assert actual.height == expected.height
    assert len(actual.planes) == len(expected.planes)
    for plane_index, (actual_plane, expected_plane) in enumerate(zip(actual.planes, expected.planes)):
        assert actual_plane.shape == expected_plane.shape, f"plane {plane_index} shape differs"
        assert actual_plane.dtype == expected_plane.dtype, f"plane {plane_index} dtype differs"
        assert actual_plane.device == expected_plane.device, f"plane {plane_index} device differs"
        assert torch.equal(actual_plane, expected_plane), f"plane {plane_index} pixels differ"
