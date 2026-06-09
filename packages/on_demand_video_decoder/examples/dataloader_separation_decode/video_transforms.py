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

from typing import List
import typing

import numpy as np
import accvlab.on_demand_video_decoder as nvc_ondemand

import torch
from torch.utils.data._utils import collate
import torch.cuda.nvtx as nvtx


def is_diff_in_range(to_comp_1, to_comp_2, tolerance):
    """
    Check if the difference is in the range.

    Args:
        to_comp_1: First value to compare.
        to_comp_2: Second value to compare.
        tolerance: Tolerance for the difference.
    """
    if isinstance(to_comp_1, tuple) and isinstance(to_comp_2, tuple):
        results = [is_diff_in_range(c1, c2, tolerance) for c1, c2 in zip(to_comp_1, to_comp_2)]
        in_range = all(r[0] for r in results)
        max_diff = max(r[1] for r in results)
        count_in_range = sum(r[2] for r in results)
        return in_range, max_diff, count_in_range

    # single channel logic
    if torch.is_tensor(to_comp_1):
        to_comp_1 = to_comp_1.cpu()
    if torch.is_tensor(to_comp_2):
        to_comp_2 = to_comp_2.cpu()
    to_comp_1 = np.array(to_comp_1)
    to_comp_2 = np.array(to_comp_2)
    use_int = issubclass(to_comp_1.dtype.type, np.integer) and issubclass(to_comp_2.dtype.type, np.integer)
    type_to_use = np.int64 if use_int else np.float64
    diffs = np.abs(np.array(to_comp_1).astype(type_to_use) - np.array(to_comp_2).astype(type_to_use))
    max_diff = diffs.max()
    in_range = max_diff <= tolerance
    count_in_range = np.sum(diffs > tolerance)
    return in_range, max_diff, count_in_range


def diff(gop_decoded, opencv_decoded, file_names, frames, num_files, diff_tolerance=21):
    """
    Check if the difference is in the range.

    Args:
        gop_decoded: Decoded frames from accvlab.on_demand_video_decoder.
        opencv_decoded: Decoded frames from OpenCV.
        file_names: List of file names.
        frames: List of frames.
        num_files: Number of files.
        diff_tolerance: Tolerance for the difference.
    """
    for file_name, frame, i in zip(file_names, frames, range(num_files)):
        same_with_opencv_dec, max_diff, count_in_range = is_diff_in_range(
            gop_decoded[i], opencv_decoded[i], diff_tolerance
        )
        print(file_name, frame, same_with_opencv_dec, max_diff)
        if not same_with_opencv_dec:
            print(f"Error: {file_name} {frame} is not same with opencv_decoded")
            return max_diff
    return 0


def opencv_decode_bgr(enc_file_paths, frame_ids):
    """
    Decode the video frames using OpenCV.

    Args:
        enc_file_paths: List of encoded file paths.
        frame_ids: List of frame indices.
    """
    import cv2

    res = []
    for filename, idx in zip(enc_file_paths, frame_ids):
        cap = cv2.VideoCapture(filename)
        for i in range(idx + 1):
            ret, frame = cap.read()
            assert (
                ret
            ), f"Frame {i} could not be read. Required frame: {idx}; all frames before the required are also read."
        res.append(torch.unsqueeze(torch.as_tensor(frame), 0))
    return res


class PacketOndemandBuffers(typing.NamedTuple):
    """List of GOP (Group of Pictures) packets. This contains packets for the GOP structure."""

    gop_packets: List[np.ndarray]
    """List of target frame indices."""
    target_frame_list: List[int]
    """List of target file paths."""
    target_file_list: List[str]
    """Use cache."""
    use_cache: bool
    """Group index."""
    sample_idx: int

    @classmethod
    def collate(
        cls, samples: List["PacketOndemandBuffers"], *, collate_fn_map=None
    ) -> List["PacketOndemandBuffers"]:
        """
        Collate the samples.

        Args:
            samples: List of PacketOndemandBuffers.
            collate_fn_map: Collate function map.
        """
        return samples


collate.default_collate_fn_map[PacketOndemandBuffers] = PacketOndemandBuffers.collate


class DecodeVideoOnDemand:
    def __init__(self, device_id: int, num_cameras: int, **kwargs) -> None:
        """
        Initialize the DecodeVideoOnDemand.

        Args:
            batch_size: Number of clips.
            num_cameras: Number of cameras.
        """
        self._nv_gop_dec = None
        self._cached_packet_data = [None]
        self._batch_size = 0
        self._num_cameras = num_cameras
        self._device_id = device_id
        self._check_result = False

    def __lazy_init__(self, batch_size):
        if self._nv_gop_dec is not None:
            return

        self._batch_size = batch_size
        self._nv_gop_dec = nvc_ondemand.CreateGopDecoder(
            maxfiles=self._num_cameras * self._batch_size,
            iGpu=self._device_id,
        )
        self._cached_packet_data = [None] * self._batch_size

    def reshape_to_2d(self, src, batch_size, num_camera):
        if len(src) != batch_size * num_camera:
            raise ValueError("invalid length of lst", len(src), batch_size, num_camera)

        reshaped = []
        idx = 0
        for b in range(batch_size):
            sample = src[idx : idx + num_camera]
            reshaped.append(sample)
            idx += num_camera
        return reshaped

    def transform(self, batch: List[PacketOndemandBuffers]) -> List[List[torch.Tensor]]:
        """
        Transform the batch.

        Args:
            batch: Batch of PacketOndemandBuffers.
        """
        self.__lazy_init__(len(batch) * len(batch[0]))

        if len(batch) == 1 and len(batch[0]) == 1:
            decoded_batch = self.decode_sample(batch[0][0])
        else:
            decoded_batch = self.decode_batch(batch)
        decoded_batch_2d = self.reshape_to_2d(decoded_batch, self._batch_size, self._num_cameras)
        return decoded_batch_2d

    def decode_batch(self, batch: List[PacketOndemandBuffers]) -> List[torch.Tensor]:
        """
        Decode the clips by merging multiple samples and decoding them together.

        Args:
            clips: clips of PacketOndemandBuffers.
        """
        nvtx.range_push("decode_clips")

        packet_data_arrays = []
        all_target_file_list = []
        all_target_frame_list = []

        for packet_buffers in batch:
            for packet_buffer in packet_buffers:
                use_cache = packet_buffer.use_cache
                sample_idx = packet_buffer.sample_idx

                # print(use_cache, sample_idx)
                if use_cache:
                    gop_packets = self._cached_packet_data[sample_idx]
                else:
                    gop_packets = packet_buffer.gop_packets
                    self._cached_packet_data[sample_idx] = gop_packets

                packet_data_arrays.extend(gop_packets)
                all_target_file_list.extend(packet_buffer.target_file_list)
                all_target_frame_list.extend(packet_buffer.target_frame_list)

        # Decode the per-video packet data list directly.
        nvtx.range_push("decode_frames")
        try:
            decoded_frames = self._nv_gop_dec.DecodeFromGOPListRGB(
                packet_data_arrays, all_target_file_list, all_target_frame_list, True  # RGB
            )
            # print(all_target_file_list)
            # print(all_target_frame_list)
        except Exception as e:
            print(f"Error decoding packets: {e}")
            print(f"Number of packet arrays: {len(packet_data_arrays)}")
            print(f"all_target_file_list: {all_target_file_list}")
            print(f"all_target_frame_list: {all_target_frame_list}")
            raise

        nvtx.range_pop()  # decode_frames

        # Convert to tensors and split back into individual samples
        nvtx.range_push("convert")
        target_tensors = [
            torch.unsqueeze(torch.as_tensor(df, device=torch.device('cuda')), 0) for df in decoded_frames
        ]
        nvtx.range_pop()  # convert

        # open it to check the result with opencv
        if self._check_result:
            opencv_decoded = opencv_decode_bgr(all_target_file_list, all_target_frame_list)
            st = diff(
                target_tensors,
                opencv_decoded,
                all_target_file_list,
                all_target_frame_list,
                len(all_target_file_list),
                diff_tolerance=21,
            )
            if st != 0:
                exit(-1)

        nvtx.range_pop()  # decode_batch
        return target_tensors

    def decode_sample(self, episode_packet_buffer: PacketOndemandBuffers) -> torch.Tensor:
        """
        Decode the sample.

        Args:
            episode_packet_buffer: Episode packet buffer.
        """
        nvtx.range_push("decode_sample")

        nvtx.range_push("load_packets")
        use_cache = episode_packet_buffer.use_cache
        sample_idx = episode_packet_buffer.sample_idx
        target_file_list = episode_packet_buffer.target_file_list
        target_frame_list = episode_packet_buffer.target_frame_list

        if use_cache:
            gop_packets = self._cached_packet_data[sample_idx]
        else:

            gop_packets = episode_packet_buffer.gop_packets
            self._cached_packet_data[sample_idx] = gop_packets
        nvtx.range_pop()  # load_packets

        nvtx.range_push("decode_frames")
        try:
            decoded_frames = self._nv_gop_dec.DecodeFromGOPListRGB(
                gop_packets, target_file_list, target_frame_list, True  # RGB
            )
        except Exception as e:
            print(f"Error decoding packets: {e}")
            print(f"gop_packets: {gop_packets}")
            exit(1)
        nvtx.range_pop()  # decode_frames

        nvtx.range_push("convert_to_tensor")
        target_tensor = [
            torch.unsqueeze(torch.as_tensor(df, device=torch.device('cuda')), 0) for df in decoded_frames
        ]

        # open it to check the result with opencv
        if self._check_result:
            opencv_decoded = opencv_decode_bgr(target_file_list, target_frame_list)
            st = diff(
                target_tensor,
                opencv_decoded,
                target_file_list,
                target_frame_list,
                len(target_file_list),
                diff_tolerance=21,
            )
            if st != 0:
                exit(-1)

        # for tensor in target_tensor:
        #     print(tensor.shape)
        nvtx.range_pop()  # convert_to_tensor

        nvtx.range_pop()  # decode_sample
        return target_tensor
