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

"""Extension point for a future independent binary correctness checker."""

from __future__ import annotations

from typing import Protocol, Sequence

from common.adapters import OutputFormat
from common.model import CanonicalFrame, DecodeCase


class BinaryCorrectnessChecker(Protocol):
    """Optional checker supplied by an environment with an approved reference."""

    def validate(
        self,
        case: DecodeCase,
        output_format: OutputFormat,
        frames: Sequence[CanonicalFrame],
    ) -> None: ...
