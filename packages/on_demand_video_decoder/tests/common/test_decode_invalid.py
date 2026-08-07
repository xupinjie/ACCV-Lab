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

"""Invalid requests shared by all file-and-frame decoder APIs."""

from __future__ import annotations

from typing import Tuple

import pytest

from common.adapters import DecoderTestAdapter, OutputFormat
from common.cases import sample_videos
from common.model import InvalidDecodeCase

DecoderInput = Tuple[DecoderTestAdapter, OutputFormat]
EXPECTED_EXCEPTIONS = (RuntimeError, ValueError, TypeError)


def _assert_recovery(adapter: DecoderTestAdapter, output_format: OutputFormat) -> None:
    frames = adapter.decode((sample_videos()[0],), (0,), output_format)
    assert frames is not None
    assert len(frames) == 1
    assert frames[0] is not None
    recovered = adapter.normalize(frames[0], output_format)
    assert recovered.width > 0
    assert recovered.height > 0
    assert recovered.planes


def test_empty_request_is_safe_and_recovers(
    invalid_decoder_input: DecoderInput,
) -> None:
    adapter, output_format = invalid_decoder_input

    try:
        frames = adapter.decode((), (), output_format)
    except EXPECTED_EXCEPTIONS as error:
        assert str(error).strip(), "decode error must contain an actionable message"
    else:
        assert frames == []

    _assert_recovery(adapter, output_format)


def test_invalid_request_fails_atomically_and_recovers(
    invalid_decoder_input: DecoderInput,
    invalid_decode_case: InvalidDecodeCase,
) -> None:
    adapter, output_format = invalid_decoder_input

    with pytest.raises(EXPECTED_EXCEPTIONS) as error:
        adapter.decode(
            invalid_decode_case.videos,
            invalid_decode_case.frame_indices,
            output_format,
        )
    assert str(error.value).strip(), "decode error must contain an actionable message"

    _assert_recovery(adapter, output_format)
