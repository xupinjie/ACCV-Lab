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

"""Format contracts and cross-API equality for committed video fixtures."""

from __future__ import annotations

from fractions import Fraction
from typing import Tuple

import accvlab.on_demand_video_decoder as nvc
from common.adapters import DecoderTestAdapter, adapter_factories
from common.cases import resource_cases
from common.model import CanonicalFrame, ResourceDecodeCase, assert_frames_equal

ResourceDecoderInput = Tuple[DecoderTestAdapter, ResourceDecodeCase]


def _decode_one(adapter: DecoderTestAdapter, case: ResourceDecodeCase) -> CanonicalFrame:
    frames = adapter.decode(
        (case.video,),
        (case.frame_index,),
        case.preferred_output_format,
    )

    assert frames is not None
    assert len(frames) == 1
    assert frames[0] is not None
    return adapter.normalize(frames[0], case.preferred_output_format)


def test_decode_committed_resource(resource_decoder_input: ResourceDecoderInput) -> None:
    adapter, case = resource_decoder_input
    frame = _decode_one(adapter, case)

    assert frame.format == case.expected_format
    assert frame.dtype == case.expected_dtype
    assert tuple(tuple(plane.shape) for plane in frame.planes) == case.expected_plane_shapes


def test_resource_adapters_are_pixel_identical(resource_case: ResourceDecodeCase) -> None:
    baseline = None
    compared_adapters = 0
    for factory in adapter_factories(nvc):
        if resource_case.preferred_output_format not in factory.output_formats:
            continue
        adapter = factory.create()
        try:
            actual = _decode_one(adapter, resource_case)
        finally:
            adapter.close()
        if baseline is None:
            baseline = actual
        else:
            assert_frames_equal(actual, baseline)
        compared_adapters += 1

    assert compared_adapters >= 2


def test_vfr_fixture_reports_variable_timing() -> None:
    case = next(case for case in resource_cases() if case.name == "vfr_h264")
    info = nvc.GetFastInitInfo([case.video])[0]
    average_rate = Fraction(info.avg_frame_rate_num, info.avg_frame_rate_den)
    real_rate = Fraction(info.r_frame_rate_num, info.r_frame_rate_den)

    assert (info.width, info.height) == (256, 256)
    assert info.duration > 0
    assert average_rate != real_rate
