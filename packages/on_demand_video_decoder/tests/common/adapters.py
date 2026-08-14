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

"""Thin adapters for APIs that naturally decode video paths and frame indices."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, List, Literal, Sequence, Tuple

import torch

from common.model import CanonicalFrame, OutputContract

OutputFormat = Literal["rgb", "bgr", "yuv"]


_PIXEL_FORMAT_NAMES = {
    3: "NV12",
    4: "YUV444",
    5: "P016",
    6: "YUV444_16BIT",
}


class DecoderTestAdapter(ABC):
    """Minimum interface required by the common decode tests."""

    name: str
    output_formats: Tuple[OutputFormat, ...]

    def __init__(self, decoder: object) -> None:
        self.decoder = decoder

    @abstractmethod
    def decode(
        self,
        videos: Sequence[str],
        frame_indices: Sequence[int],
        output_format: OutputFormat,
    ) -> List[object]:
        """Decode one requested frame for every video entry."""

    def normalize(self, frame: object, output_format: OutputFormat) -> CanonicalFrame:
        """Take an owned snapshot suitable for exact comparison."""

        if output_format in {"rgb", "bgr"}:
            tensor = torch.as_tensor(frame, device="cuda").clone()
            if tensor.ndim != 3 or tensor.shape[-1] != 3:
                raise AssertionError(f"RGB/BGR output must have shape (H, W, 3), got {tuple(tensor.shape)}")
            return CanonicalFrame(
                format=output_format.upper(),
                planes=(tensor,),
                width=int(tensor.shape[1]),
                height=int(tensor.shape[0]),
            )

        planes = []
        for view in frame.cuda():
            plane = torch.as_tensor(view, device="cuda").clone()
            if plane.ndim == 3 and plane.shape[-1] == 1:
                plane = plane.squeeze(-1)
            planes.append(plane)
        if not planes:
            raise AssertionError("YUV output did not expose any planes")

        format_name = _PIXEL_FORMAT_NAMES.get(int(frame.format), f"UNKNOWN_{int(frame.format)}")
        return CanonicalFrame(
            format=format_name,
            planes=tuple(planes),
            width=int(planes[0].shape[1]),
            height=int(planes[0].shape[0]),
        )

    def output_contract(self, output_format: OutputFormat) -> OutputContract:
        # Native YUV output is a format family: its concrete layout and dtype
        # depend on the source pixel format and bit depth.
        kind = "YUV" if output_format == "yuv" else output_format.upper()
        dtypes = (torch.uint8, torch.uint16) if output_format == "yuv" else (torch.uint8,)
        return OutputContract(kind=kind, dtypes=dtypes)

    def close(self) -> None:
        self.decoder = None


class RandomAdapter(DecoderTestAdapter):
    name = "random"
    output_formats = ("rgb", "bgr", "yuv")

    def decode(
        self,
        videos: Sequence[str],
        frame_indices: Sequence[int],
        output_format: OutputFormat,
    ) -> List[object]:
        if output_format == "yuv":
            return self.decoder.Decode(list(videos), list(frame_indices))
        return self.decoder.DecodeN12ToRGB(
            list(videos),
            list(frame_indices),
            output_format == "bgr",
        )


class RandomFastInitAdapter(DecoderTestAdapter):
    name = "random_fast_init"
    output_formats = ("rgb", "bgr", "yuv")

    def __init__(
        self,
        decoder: object,
        get_fast_init_info: Callable[[Sequence[str]], object],
    ) -> None:
        super().__init__(decoder)
        self.get_fast_init_info = get_fast_init_info

    def decode(
        self,
        videos: Sequence[str],
        frame_indices: Sequence[int],
        output_format: OutputFormat,
    ) -> List[object]:
        videos = list(videos)
        frame_indices = list(frame_indices)
        stream_infos = self.get_fast_init_info(videos)
        if output_format == "yuv":
            return self.decoder.Decode(
                videos,
                frame_indices,
                fastStreamInfos=stream_infos,
            )
        return self.decoder.DecodeN12ToRGB(
            videos,
            frame_indices,
            output_format == "bgr",
            fastStreamInfos=stream_infos,
        )


class StreamAdapter(DecoderTestAdapter):
    name = "stream"
    output_formats = ("rgb", "bgr", "yuv")

    def decode(
        self,
        videos: Sequence[str],
        frame_indices: Sequence[int],
        output_format: OutputFormat,
    ) -> List[object]:
        if output_format == "yuv":
            return self.decoder.Decode(list(videos), list(frame_indices))
        return self.decoder.DecodeN12ToRGB(
            list(videos),
            list(frame_indices),
            output_format == "bgr",
        )


class StreamAsyncAdapter(DecoderTestAdapter):
    name = "stream_async"
    output_formats = ("rgb", "bgr")

    def decode(
        self,
        videos: Sequence[str],
        frame_indices: Sequence[int],
        output_format: OutputFormat,
    ) -> List[object]:
        videos = list(videos)
        frame_indices = list(frame_indices)
        as_bgr = output_format == "bgr"
        self.decoder.DecodeN12ToRGBAsync(videos, frame_indices, as_bgr)
        return self.decoder.DecodeN12ToRGBAsyncGetBuffer(
            videos,
            frame_indices,
            as_bgr,
        )


class BatchStreamAsyncAdapter(DecoderTestAdapter):
    name = "batch_stream_async"
    output_formats = ("rgb", "bgr")

    def decode(
        self,
        videos: Sequence[str],
        frame_indices: Sequence[int],
        output_format: OutputFormat,
    ) -> List[object]:
        videos = list(videos)
        frame_indices_2d = [[frame_index] for frame_index in frame_indices]
        as_bgr = output_format == "bgr"
        self.decoder.Decode(videos, frame_indices_2d, as_bgr)
        frames_2d = self.decoder.GetBuffer(videos, frame_indices_2d, as_bgr)
        for video_index, frames in enumerate(frames_2d):
            if len(frames) != 1:
                raise AssertionError(
                    f"batch output {video_index} must contain exactly one frame, got {len(frames)}"
                )
        return [frames[0] for frames in frames_2d]


def _decode_gop_data(
    decoder: object,
    gop_data: Sequence[object],
    videos: Sequence[str],
    frame_indices: Sequence[int],
    output_format: OutputFormat,
) -> List[object]:
    if output_format == "yuv":
        return decoder.DecodeFromGOPList(
            list(gop_data),
            list(videos),
            list(frame_indices),
        )
    return decoder.DecodeFromGOPListRGB(
        list(gop_data),
        list(videos),
        list(frame_indices),
        output_format == "bgr",
    )


class BatchGopAsyncAdapter(DecoderTestAdapter):
    name = "batch_gop_async"
    output_formats = ("rgb", "bgr", "yuv")

    def __init__(self, demuxer: object, decoder: object) -> None:
        super().__init__(decoder)
        self.demuxer = demuxer

    def decode(
        self,
        videos: Sequence[str],
        frame_indices: Sequence[int],
        output_format: OutputFormat,
    ) -> List[object]:
        videos = list(videos)
        frame_indices = list(frame_indices)
        gop_list = self.demuxer.GetGOPList(
            videos,
            frame_indices,
            useGOPCache=False,
        )
        gop_data_2d = [[bundle[0]] for bundle in gop_list]
        frame_indices_2d = [[frame_index] for frame_index in frame_indices]

        if output_format == "yuv":
            self.decoder.DecodeFromGOPList(
                gop_data_2d,
                videos,
                frame_indices_2d,
            )
            frames_2d = self.decoder.DecodeFromGOPListGetBuffer(
                videos,
                frame_indices_2d,
            )
        else:
            as_bgr = output_format == "bgr"
            self.decoder.DecodeFromGOPListRGB(
                gop_data_2d,
                videos,
                frame_indices_2d,
                as_bgr,
            )
            frames_2d = self.decoder.DecodeFromGOPListRGBGetBuffer(
                videos,
                frame_indices_2d,
                as_bgr,
            )

        for video_index, frames in enumerate(frames_2d):
            if len(frames) != 1:
                raise AssertionError(
                    f"batch output {video_index} must contain exactly one frame, got {len(frames)}"
                )
        return [frames[0] for frames in frames_2d]

    def close(self) -> None:
        self.demuxer = None
        super().close()


class GopListAdapter(DecoderTestAdapter):
    name = "gop_list"
    output_formats = ("rgb", "bgr", "yuv")

    def __init__(self, demuxer: object, decoder: object) -> None:
        super().__init__(decoder)
        self.demuxer = demuxer

    def decode(
        self,
        videos: Sequence[str],
        frame_indices: Sequence[int],
        output_format: OutputFormat,
    ) -> List[object]:
        gop_list = self.demuxer.GetGOPList(
            list(videos),
            list(frame_indices),
            useGOPCache=False,
        )
        gop_data = [bundle[0] for bundle in gop_list]
        return _decode_gop_data(
            self.decoder,
            gop_data,
            videos,
            frame_indices,
            output_format,
        )

    def close(self) -> None:
        self.demuxer = None
        super().close()


class GopListFastInitAdapter(DecoderTestAdapter):
    name = "gop_list_fast_init"
    output_formats = ("rgb", "bgr", "yuv")

    def __init__(
        self,
        demuxer: object,
        decoder: object,
        get_fast_init_info: Callable[[Sequence[str]], object],
    ) -> None:
        super().__init__(decoder)
        self.demuxer = demuxer
        self.get_fast_init_info = get_fast_init_info

    def decode(
        self,
        videos: Sequence[str],
        frame_indices: Sequence[int],
        output_format: OutputFormat,
    ) -> List[object]:
        videos = list(videos)
        frame_indices = list(frame_indices)
        gop_list = self.demuxer.GetGOPList(
            videos,
            frame_indices,
            fastStreamInfos=self.get_fast_init_info(videos),
            useGOPCache=False,
        )
        return _decode_gop_data(
            self.decoder,
            [bundle[0] for bundle in gop_list],
            videos,
            frame_indices,
            output_format,
        )

    def close(self) -> None:
        self.demuxer = None
        super().close()


class GroupAdapter(DecoderTestAdapter):
    name = "group"
    output_formats = ("rgb", "bgr")

    def __init__(self, demuxer: object, decoder: object) -> None:
        super().__init__(decoder)
        self.demuxer = demuxer

    def decode(
        self,
        videos: Sequence[str],
        frame_indices: Sequence[int],
        output_format: OutputFormat,
    ) -> List[object]:
        if len(videos) != len(frame_indices):
            raise ValueError("videos and frame_indices must have the same length")
        requests = [
            {"filepath": video, "frame_ids": [frame_index]}
            for video, frame_index in zip(videos, frame_indices)
        ]
        groups = self.demuxer.GetGOPGroups(requests)
        decoded_groups = self.decoder.DecodeFromGOPGroupsRGB(
            groups,
            output_format == "bgr",
        )

        frames = [None] * len(videos)
        for group in decoded_groups:
            source_index = group["source_index"]
            for frame, positions in zip(group["frames"], group["frame_positions"]):
                for position in positions:
                    if position != 0:
                        raise AssertionError(
                            f"source {source_index} returned unexpected frame position {position}"
                        )
                    if frames[source_index] is not None:
                        raise AssertionError(f"source {source_index} returned more than one frame")
                    frames[source_index] = frame

        if any(frame is None for frame in frames):
            raise AssertionError("group API did not return one frame for every input video")
        return frames

    def close(self) -> None:
        self.demuxer = None
        super().close()


class GopFileAdapter(DecoderTestAdapter):
    name = "gop_file"
    output_formats = ("rgb", "bgr", "yuv")

    def __init__(
        self,
        demuxer: object,
        decoder: object,
        save_gop_to_file: Callable[[object, str], object],
    ) -> None:
        super().__init__(decoder)
        self.demuxer = demuxer
        self.save_gop_to_file = save_gop_to_file

    def decode(
        self,
        videos: Sequence[str],
        frame_indices: Sequence[int],
        output_format: OutputFormat,
    ) -> List[object]:
        gop_list = self.demuxer.GetGOPList(
            list(videos),
            list(frame_indices),
            useGOPCache=False,
        )
        with TemporaryDirectory(prefix="decoder-common-gop-") as temp_dir:
            gop_paths = []
            for index, (gop_data, _, _) in enumerate(gop_list):
                gop_path = Path(temp_dir) / f"gop-{index}.bin"
                self.save_gop_to_file(gop_data, str(gop_path))
                gop_paths.append(str(gop_path))
            loaded_gop_data = self.decoder.LoadGopsToList(gop_paths)
            return _decode_gop_data(
                self.decoder,
                loaded_gop_data,
                videos,
                frame_indices,
                output_format,
            )

    def close(self) -> None:
        self.demuxer = None
        super().close()


@dataclass(frozen=True)
class AdapterFactory:
    name: str
    output_formats: Tuple[OutputFormat, ...]
    create: Callable[[], DecoderTestAdapter]


def adapter_factories(nvc: object) -> List[AdapterFactory]:
    """Create fresh decoders for each test without adding a registry system."""

    max_files = 8
    return [
        AdapterFactory(
            RandomAdapter.name,
            RandomAdapter.output_formats,
            lambda: RandomAdapter(nvc.CreateGopDecoder(maxfiles=max_files, iGpu=0)),
        ),
        AdapterFactory(
            RandomFastInitAdapter.name,
            RandomFastInitAdapter.output_formats,
            lambda: RandomFastInitAdapter(
                nvc.CreateGopDecoder(maxfiles=max_files, iGpu=0),
                nvc.GetFastInitInfo,
            ),
        ),
        AdapterFactory(
            StreamAdapter.name,
            StreamAdapter.output_formats,
            lambda: StreamAdapter(nvc.CreateSampleReader(num_of_set=1, num_of_file=max_files, iGpu=0)),
        ),
        AdapterFactory(
            StreamAsyncAdapter.name,
            StreamAsyncAdapter.output_formats,
            lambda: StreamAsyncAdapter(nvc.CreateSampleReader(num_of_set=1, num_of_file=max_files, iGpu=0)),
        ),
        AdapterFactory(
            BatchStreamAsyncAdapter.name,
            BatchStreamAsyncAdapter.output_formats,
            lambda: BatchStreamAsyncAdapter(
                nvc.CreateBatchAsyncStreamReader(
                    num_of_set=1,
                    num_of_file=max_files,
                    max_frames_per_decode_call=1,
                    iGpu=0,
                )
            ),
        ),
        AdapterFactory(
            BatchGopAsyncAdapter.name,
            BatchGopAsyncAdapter.output_formats,
            lambda: BatchGopAsyncAdapter(
                nvc.CreateGopDecoder(maxfiles=max_files, iGpu=0),
                nvc.CreateBatchAsyncGopDecoder(
                    maxfiles=max_files,
                    max_frames_per_decode_call=1,
                    iGpu=0,
                ),
            ),
        ),
        AdapterFactory(
            GopListAdapter.name,
            GopListAdapter.output_formats,
            lambda: GopListAdapter(
                nvc.CreateGopDecoder(maxfiles=max_files, iGpu=0),
                nvc.CreateGopDecoder(maxfiles=max_files, iGpu=0),
            ),
        ),
        AdapterFactory(
            GopListFastInitAdapter.name,
            GopListFastInitAdapter.output_formats,
            lambda: GopListFastInitAdapter(
                nvc.CreateGopDecoder(maxfiles=max_files, iGpu=0),
                nvc.CreateGopDecoder(maxfiles=max_files, iGpu=0),
                nvc.GetFastInitInfo,
            ),
        ),
        AdapterFactory(
            GroupAdapter.name,
            GroupAdapter.output_formats,
            lambda: GroupAdapter(
                nvc.CreateGopDecoder(maxfiles=max_files, iGpu=0),
                nvc.CreateGopDecoder(maxfiles=max_files, iGpu=0),
            ),
        ),
        AdapterFactory(
            GopFileAdapter.name,
            GopFileAdapter.output_formats,
            lambda: GopFileAdapter(
                nvc.CreateGopDecoder(maxfiles=max_files, iGpu=0),
                nvc.CreateGopDecoder(maxfiles=max_files, iGpu=0),
                nvc.SaveGopToFile,
            ),
        ),
    ]
