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

"""Minimum decode behavior shared by file-and-frame decoder APIs."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

from common.adapters import DecoderTestAdapter, OutputFormat
from common.cases import sample_videos
from common.checker import BinaryCorrectnessChecker
from common.model import CanonicalFrame, DecodeCase, assert_frames_equal

DecoderInput = Tuple[DecoderTestAdapter, OutputFormat]


def _decode_and_normalize(
    adapter: DecoderTestAdapter,
    output_format: OutputFormat,
    videos: Sequence[str],
    frame_indices: Sequence[int],
) -> List[CanonicalFrame]:
    frames = adapter.decode(videos, frame_indices, output_format)
    assert frames is not None
    assert len(frames) == len(videos)
    assert all(frame is not None for frame in frames)
    return [adapter.normalize(frame, output_format) for frame in frames]


def _assert_output_contract(
    adapter: DecoderTestAdapter,
    output_format: OutputFormat,
    frame: CanonicalFrame,
) -> None:
    contract = adapter.output_contract(output_format)
    assert frame.width > 0
    assert frame.height > 0
    assert frame.planes
    assert frame.dtype in contract.dtypes
    assert frame.device_type == contract.device_type

    if contract.kind in {"RGB", "BGR"}:
        assert frame.format == contract.kind
        assert len(frame.planes) == 1
        assert frame.planes[0].shape == (frame.height, frame.width, 3)
    else:
        assert frame.format in {"NV12", "P016", "YUV444", "YUV444_16BIT"}
        if frame.format in {"NV12", "P016"}:
            assert len(frame.planes) == 2
            assert frame.planes[0].shape == (frame.height, frame.width)
            assert frame.planes[1].shape == (frame.height // 2, frame.width // 2, 2)
        else:
            assert len(frame.planes) == 3
            assert all(plane.shape == (frame.height, frame.width) for plane in frame.planes)


def test_common_decode_contract(
    decoder_input: DecoderInput,
    decode_case: DecodeCase,
    binary_correctness_checker: BinaryCorrectnessChecker,
) -> None:
    adapter, output_format = decoder_input
    frames = _decode_and_normalize(
        adapter,
        output_format,
        decode_case.videos,
        decode_case.frame_indices,
    )
    for frame in frames:
        _assert_output_contract(adapter, output_format, frame)

    if decode_case.name == "same_video_unsorted_duplicate":
        assert_frames_equal(frames[0], frames[2])

    if binary_correctness_checker is not None:
        binary_correctness_checker.validate(decode_case, output_format, frames)


def test_decode_is_deterministic(decoder_input: DecoderInput) -> None:
    adapter, output_format = decoder_input
    videos = sample_videos()[:3]
    frame_indices = (0, 37, 100)
    first = _decode_and_normalize(adapter, output_format, videos, frame_indices)
    second = _decode_and_normalize(adapter, output_format, videos, frame_indices)
    for actual, expected in zip(second, first):
        assert_frames_equal(actual, expected)


def test_request_order_is_preserved(decoder_input: DecoderInput) -> None:
    adapter, output_format = decoder_input
    video = sample_videos()[0]
    videos = (video, video, video)
    ordered = _decode_and_normalize(adapter, output_format, videos, (0, 30, 60))
    reordered = _decode_and_normalize(adapter, output_format, videos, (60, 0, 30))

    assert_frames_equal(reordered[0], ordered[2])
    assert_frames_equal(reordered[1], ordered[0])
    assert_frames_equal(reordered[2], ordered[1])


def test_same_frame_is_batch_independent(decoder_input: DecoderInput) -> None:
    adapter, output_format = decoder_input
    videos = sample_videos()
    single = _decode_and_normalize(adapter, output_format, (videos[0],), (30,))[0]
    mixed = _decode_and_normalize(
        adapter,
        output_format,
        (videos[1], videos[0], videos[2]),
        (10, 30, 50),
    )[1]
    assert_frames_equal(mixed, single)
