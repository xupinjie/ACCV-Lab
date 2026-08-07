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

"""Fixtures for the common decoder contract."""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import pytest

import accvlab.on_demand_video_decoder as nvc
from common.adapters import DecoderTestAdapter, OutputFormat, adapter_factories
from common.cases import INVALID_CASE_NAMES, common_cases, invalid_case, resource_cases
from common.checker import BinaryCorrectnessChecker

_ADAPTER_FACTORIES = adapter_factories(nvc)
_ADAPTER_INPUTS = tuple(
    (factory, output_format) for factory in _ADAPTER_FACTORIES for output_format in factory.output_formats
)
_INVALID_ADAPTER_INPUTS = tuple(
    (factory, output_format)
    for factory in _ADAPTER_FACTORIES
    for output_format in factory.output_formats
    if output_format != "bgr"
)
_COMMON_CASES = common_cases()
_RESOURCE_CASES = resource_cases()
_RESOURCE_ADAPTER_INPUTS = tuple(
    (factory, resource_case)
    for resource_case in _RESOURCE_CASES
    for factory in _ADAPTER_FACTORIES
    if resource_case.preferred_output_format in factory.output_formats
)


@pytest.fixture(
    params=_ADAPTER_INPUTS,
    ids=lambda value: f"{value[0].name}-{value[1]}",
)
def decoder_input(request) -> Iterator[Tuple[DecoderTestAdapter, OutputFormat]]:
    factory, output_format = request.param
    adapter = factory.create()
    try:
        yield adapter, output_format
    finally:
        adapter.close()


@pytest.fixture(params=_COMMON_CASES, ids=lambda case: case.name)
def decode_case(request):
    return request.param


@pytest.fixture(
    params=_INVALID_ADAPTER_INPUTS,
    ids=lambda value: f"{value[0].name}-{value[1]}",
)
def invalid_decoder_input(request) -> Iterator[Tuple[DecoderTestAdapter, OutputFormat]]:
    factory, output_format = request.param
    adapter = factory.create()
    try:
        yield adapter, output_format
    finally:
        adapter.close()


@pytest.fixture(params=INVALID_CASE_NAMES)
def invalid_decode_case(request, tmp_path):
    return invalid_case(request.param, tmp_path)


@pytest.fixture(
    params=_RESOURCE_ADAPTER_INPUTS,
    ids=lambda value: (f"{value[0].name}-{value[1].name}-{value[1].preferred_output_format}"),
)
def resource_decoder_input(request):
    factory, resource_case = request.param
    adapter = factory.create()
    try:
        yield adapter, resource_case
    finally:
        adapter.close()


@pytest.fixture(params=_RESOURCE_CASES, ids=lambda case: case.name)
def resource_case(request):
    return request.param


@pytest.fixture
def binary_correctness_checker() -> Optional[BinaryCorrectnessChecker]:
    """Reserved for an approved binary reference checker; intentionally unset."""

    return None
