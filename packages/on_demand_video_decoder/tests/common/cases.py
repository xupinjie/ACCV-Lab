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

"""Deterministic requests shared by all file-and-frame decoder APIs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

import torch

from common.model import DecodeCase, InvalidDecodeCase, ResourceDecodeCase

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SAMPLE_CLIP_DIR = DATA_DIR / "sample_clip"
PIX_FMT_VARIANTS_DIR = DATA_DIR / "pix_fmt_variants"
OPEN_GOP_VARIANTS_DIR = DATA_DIR / "open_gop_variant"
TEMPORAL_VARIANTS_DIR = DATA_DIR / "temporal_variants"


def sample_videos() -> Tuple[str, ...]:
    videos = tuple(str(path) for path in sorted(SAMPLE_CLIP_DIR.glob("*.mp4")))
    if len(videos) < 5:
        raise RuntimeError(f"at least five common test videos are required in {SAMPLE_CLIP_DIR}")
    return videos


def common_cases() -> List[DecodeCase]:
    videos = sample_videos()
    return [
        DecodeCase("single_first", (videos[0],), (0,)),
        DecodeCase("multi_video_mixed", videos[:3], (0, 37, 200)),
        DecodeCase(
            "same_video_unsorted_duplicate",
            (videos[0], videos[0], videos[0], videos[0]),
            (60, 0, 60, 29),
        ),
        DecodeCase(
            "boundaries",
            (videos[0], videos[0], videos[0]),
            (0, 100, 200),
        ),
    ]


def resource_cases() -> Tuple[ResourceDecodeCase, ...]:
    """Committed codec fixtures that each need one decode smoke path."""

    cases = (
        ResourceDecodeCase(
            "h264_avc1",
            str(PIX_FMT_VARIANTS_DIR / "h264_avc1_yuv420p.mp4"),
            33,
            "rgb",
            "RGB",
            torch.uint8,
            ((256, 256, 3),),
        ),
        ResourceDecodeCase(
            "hevc_hev1_10bit",
            str(PIX_FMT_VARIANTS_DIR / "hevc_hev1_yuv420p10le.mp4"),
            33,
            "yuv",
            "P016",
            torch.uint16,
            ((256, 256), (128, 128, 2)),
        ),
        ResourceDecodeCase(
            "hevc_hvc1_10bit",
            str(PIX_FMT_VARIANTS_DIR / "hevc_hvc1_yuv420p10le.mp4"),
            33,
            "yuv",
            "P016",
            torch.uint16,
            ((256, 256), (128, 128, 2)),
        ),
        ResourceDecodeCase(
            "vfr_h264",
            str(TEMPORAL_VARIANTS_DIR / "vfr_h264_yuv420p.mp4"),
            33,
            "rgb",
            "RGB",
            torch.uint8,
            ((256, 256, 3),),
        ),
        # Display frame 39 is a leading RASL picture associated with CRA 40.
        ResourceDecodeCase(
            "open_gop",
            str(OPEN_GOP_VARIANTS_DIR / "moving_shape_open_gop_h265.mp4"),
            39,
            "rgb",
            "RGB",
            torch.uint8,
            ((256, 256, 3),),
        ),
    )
    for case in cases:
        if not Path(case.video).is_file():
            raise RuntimeError(f"common decode resource is missing: {case.video}")
    return cases


INVALID_CASE_NAMES = (
    "too_few_frame_indices",
    "too_many_frame_indices",
    "missing_file",
    "empty_file",
    "non_video_file",
    "truncated_video",
    "negative_frame_index",
    "out_of_range_frame_index",
    "none_frame_index",
    "string_frame_index",
    "float_frame_index",
    "too_many_inputs",
)


def _insert_middle(values: Tuple[Any, ...], invalid_value: Any) -> Tuple[Any, ...]:
    return values[:2] + (invalid_value,) + values[2:]


def invalid_case(name: str, temp_dir: Path) -> InvalidDecodeCase:
    """Build one malformed request without requiring external media tools."""

    videos = sample_videos()
    valid_videos = videos[:4]
    valid_frames = (0, 30, 60, 90, 120)

    if name == "too_few_frame_indices":
        return InvalidDecodeCase(name, valid_videos, valid_frames[:3])
    if name == "too_many_frame_indices":
        return InvalidDecodeCase(name, valid_videos, valid_frames)

    if name in {"missing_file", "empty_file", "non_video_file", "truncated_video"}:
        invalid_path = temp_dir / f"{name}.mp4"
        if name == "empty_file":
            invalid_path.write_bytes(b"")
        elif name == "non_video_file":
            invalid_path.write_bytes(b"not a video\n" * 8)
        elif name == "truncated_video":
            invalid_path.write_bytes(Path(videos[0]).read_bytes()[:64])
        return InvalidDecodeCase(
            name,
            _insert_middle(valid_videos, str(invalid_path)),
            valid_frames,
        )

    invalid_frame_indices = {
        "negative_frame_index": -1,
        "out_of_range_frame_index": 999999,
        "none_frame_index": None,
        "string_frame_index": "60",
        "float_frame_index": 60.5,
    }
    if name in invalid_frame_indices:
        return InvalidDecodeCase(
            name,
            videos[:5],
            _insert_middle((0, 30, 90, 120), invalid_frame_indices[name]),
        )

    if name == "too_many_inputs":
        return InvalidDecodeCase(
            name,
            tuple(videos[index % len(videos)] for index in range(9)),
            tuple(index * 10 for index in range(9)),
        )

    raise ValueError(f"unknown invalid decode case: {name}")
