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

"""
GOP (Group of Pictures) Storage Management Module

This module provides utilities for storing and loading video GOP data,
enabling efficient video frame decoding without repeated demuxing.

This is a sample implementation that demonstrates how to organize GOP data storage.
Users can modify this according to their specific requirements.
"""

import os
import json
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import torch

import accvlab.on_demand_video_decoder as nvc


@dataclass
class GOPInfo:
    """
    Information about a single GOP file.

    Attributes:
        file_path (str): Full path to the GOP binary file
        first_frame_id (int): Index of the first frame in this GOP
        gop_len (int): Number of frames in this GOP
    """

    file_path: str
    first_frame_id: int
    gop_len: int


class GOPStorageManager:
    """
    Manager for storing and loading GOP (Group of Pictures) data using binary format.

    This class provides a hierarchical storage system for video GOP data:

    Storage Structure:
        video_base_path/
        ├── clip0/
        │   ├── video0.mp4
        │   └── video1.mp4
        └── clip1/
            ├── video0.mp4
            └── video1.mp4

        gop_base_path/
        ├── clip0/
        │   ├── video0.mp4/
        │   │   ├── .gop_index.json          # Index file for fast lookup
        │   │   ├── gop.0.30.bin             # Binary GOP files
        │   │   ├── gop.30.27.bin
        │   │   └── gop.57.30.bin
        │   └── video1.mp4/
        │       ├── .gop_index.json
        │       ├── gop.0.30.bin
        │       ├── gop.30.27.bin
        │       └── gop.57.30.bin
        └── clip1/
            ├── video0.mp4/
            │   ├── .gop_index.json
            │   ├── gop.0.30.bin
            │   ├── gop.30.27.bin
            │   └── gop.57.30.bin
            └── video1.mp4/
                ├── .gop_index.json
                ├── gop.0.30.bin
                ├── gop.30.27.bin
                └── gop.57.30.bin

    Args:
        video_base_path (str): Base directory containing video files
        gop_base_path (str): Base directory for storing GOP data
        use_persistent_index (bool): Whether to use persistent index files
    """

    # .. doc-marker-begin: gop-storage-init
    def __init__(
        self, video_base_path: str, gop_base_path: str, clip_size: int, use_persistent_index: bool = True
    ):
        """
        Initialize the GOP Storage Manager.

        Args:
            video_base_path (str): Base directory containing video files
            gop_base_path (str): Base directory for storing GOP data
            use_persistent_index (bool): Whether to use persistent index files
        """
        self.video_base_path = video_base_path
        self.gop_base_path = gop_base_path
        self.use_persistent_index = use_persistent_index
        self.nv_gop_demuxer = nvc.CreateGopDecoder(
            maxfiles=clip_size,  # Maximum number of files to use
            iGpu=0,  # GPU ID
        )
        os.makedirs(self.gop_base_path, exist_ok=True)

    # .. doc-marker-end: gop-storage-init

    def _get_index_file_path(self, gop_dir: str) -> str:
        """Get the path to the index file for a GOP directory."""
        return os.path.join(gop_dir, ".gop_index.json")

    def _save_persistent_index(self, gop_dir: str, gop_infos: List[GOPInfo]):
        """Save GOP index to persistent index file."""
        index_file = self._get_index_file_path(gop_dir)
        data = {
            'gops': [
                {'file_path': gop.file_path, 'first_frame_id': gop.first_frame_id, 'gop_len': gop.gop_len}
                for gop in gop_infos
            ]
        }
        with open(index_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_persistent_index(self, video_path: str) -> List[GOPInfo]:
        """
        Load GOP index from persistent index file or scan directory for GOP files.

        Args:
            video_path (str): Path to the video file

        Returns:
            List[GOPInfo]: Sorted list of GOP information
        """
        video_relpath = os.path.relpath(video_path, self.video_base_path)
        gop_dir = os.path.join(self.gop_base_path, video_relpath)

        if not os.path.exists(gop_dir):
            return []

        gop_infos = []

        if self.use_persistent_index:
            torch.cuda.nvtx.range_push("load_gop_index")
            # Load from persistent index file
            index_file = self._get_index_file_path(gop_dir)
            if os.path.exists(index_file):
                try:
                    with open(index_file, 'r') as f:
                        data = json.load(f)
                        for item in data.get('gops', []):
                            gop_infos.append(
                                GOPInfo(
                                    file_path=item['file_path'],
                                    first_frame_id=item['first_frame_id'],
                                    gop_len=item['gop_len'],
                                )
                            )
                except (json.JSONDecodeError, KeyError):
                    # Index file corrupted, fall back to directory scan
                    gop_infos = []
            torch.cuda.nvtx.range_pop()  # load_gop_index

        # If not using persistent index or index file doesn't exist/is corrupted,
        # scan directory and parse filenames directly
        if not gop_infos:
            torch.cuda.nvtx.range_push("scan_gop_dir")
            try:
                files = os.listdir(gop_dir)
                for file in files:
                    if file.endswith(".bin") and file.startswith("gop."):
                        file_path = os.path.join(gop_dir, file)
                        # Parse filename: gop.{first_frame_id}.{gop_len}.bin
                        file_name_parts = file.split(".")
                        if len(file_name_parts) >= 4:  # gop, first_frame_id, gop_len, bin
                            try:
                                first_frame_id = int(file_name_parts[-3])
                                gop_len = int(file_name_parts[-2])
                                gop_infos.append(
                                    GOPInfo(
                                        file_path=file_path, first_frame_id=first_frame_id, gop_len=gop_len
                                    )
                                )
                            except ValueError:
                                continue  # Skip invalid filenames
            except OSError:
                pass  # Directory might not be readable
            torch.cuda.nvtx.range_pop()  # scan_gop_dir

        # Sort by first_frame_id for efficient searching
        gop_infos.sort(key=lambda x: x.first_frame_id)
        return gop_infos

    def _find_gop_for_frame(self, gop_infos: List[GOPInfo], frame_idx: int) -> Optional[GOPInfo]:
        """
        Find the GOP that contains the specified frame using binary search.

        Args:
            gop_infos (List[GOPInfo]): Sorted list of GOP information
            frame_idx (int): Target frame index

        Returns:
            Optional[GOPInfo]: GOP info if found, None otherwise
        """
        # Use binary search since gop_infos is sorted by first_frame_id
        left, right = 0, len(gop_infos) - 1

        while left <= right:
            mid = (left + right) // 2
            gop_info = gop_infos[mid]

            if gop_info.first_frame_id <= frame_idx < gop_info.first_frame_id + gop_info.gop_len:
                return gop_info
            elif frame_idx < gop_info.first_frame_id:
                right = mid - 1
            else:
                left = mid + 1

        return None

    def store_single_gop(self, clip_name: str, video_path: str, packets_tuple) -> bool:
        """
        Save raw packet data to disk using binary format.

        Args:
            clip_name (str): Name of the clip (subdirectory)
            video_path (str): Path to the video file
            packets_tuple: Tuple from one GetGOPList item containing (numpy_data, first_frame_ids, gop_lens)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            numpy_data, first_frame_ids, gop_lens = packets_tuple
            video_name = os.path.basename(video_path)

            clip_dir = os.path.join(self.gop_base_path, clip_name, video_name)
            os.makedirs(clip_dir, exist_ok=True)

            # Use binary format instead of pickle
            gop_name = f"gop.{first_frame_ids[0]}.{gop_lens[0]}.bin"
            gop_path = os.path.join(clip_dir, gop_name)

            # Save using the new SavePacketsToFile interface
            nvc.SavePacketsToFile(numpy_data, gop_path)
            print(f"Saved GOP to {gop_path} ({numpy_data.size} bytes)")
            return True

        except Exception as e:
            print(f"Error saving GOP: {e}")
            return False

    # .. doc-marker-begin: gop-storage-store
    def store_gops(self):
        """
        Extract and store GOP data for all video files in the video_base_path.

        This method recursively processes all .mp4 files in the video base directory,
        extracts GOP data, and stores them as binary files in the GOP base directory.
        """
        for root, dirs, files in os.walk(self.video_base_path):
            for file in files:
                if file.endswith('.mp4'):
                    video_path = os.path.join(root, file)
                    video_relpath = os.path.relpath(video_path, self.video_base_path)
                    clip_relpath = os.path.dirname(video_relpath)

                    print(f"Processing video: {video_path}")

                    frame_idx = 0
                    pre_first_frame_id = -1
                    gop_infos = []

                    while True:
                        try:
                            packets_tuple = self.nv_gop_demuxer.GetGOPList([video_path], [frame_idx])[0]
                            numpy_data, first_frame_ids, gop_lens = packets_tuple

                            # Check if we've processed this GOP already
                            if pre_first_frame_id == first_frame_ids[0]:
                                break
                            pre_first_frame_id = first_frame_ids[0]

                            # Store the GOP
                            if self.store_single_gop(clip_relpath, video_path, packets_tuple):
                                # Add to index info
                                video_name = os.path.basename(video_path)
                                clip_dir = os.path.join(self.gop_base_path, clip_relpath, video_name)
                                gop_name = f"gop.{first_frame_ids[0]}.{gop_lens[0]}.bin"
                                gop_path = os.path.join(clip_dir, gop_name)

                                gop_infos.append(
                                    GOPInfo(
                                        file_path=gop_path,
                                        first_frame_id=first_frame_ids[0],
                                        gop_len=gop_lens[0],
                                    )
                                )

                            frame_idx += gop_lens[0]

                        except Exception as e:
                            print(f"Error processing frame {frame_idx}: {e}")
                            break

                    # Create index file immediately after processing each video
                    if gop_infos:
                        video_name = os.path.basename(video_path)
                        clip_dir = os.path.join(self.gop_base_path, clip_relpath, video_name)
                        self._save_persistent_index(clip_dir, gop_infos)
                        print(f"Created index with {len(gop_infos)} GOPs for {video_path}")

    # .. doc-marker-end: gop-storage-store

    # .. doc-marker-begin: gop-storage-load
    def load_gops(self, frame_ids: List[int], video_paths: List[str]) -> Optional[List[np.ndarray]]:
        """
        Load multiple GOPs for the specified frames and video paths using LoadGopsToList.

        Args:
            frame_ids (List[int]): List of target frame indices
            video_paths (List[str]): List of video file paths

        Returns:
            Optional[List[np.ndarray]]: Per-video GOP arrays compatible with DecodeFromGOPListRGB,
                                       or None if any GOP is not found
        """
        torch.cuda.nvtx.range_push("load_gops_manager")

        if len(frame_ids) != len(video_paths):
            print(f"Error: frame_ids length ({len(frame_ids)}) != video_paths length ({len(video_paths)})")
            torch.cuda.nvtx.range_pop()
            return None

        # Find GOP files for each frame
        gop_file_paths = []

        for frame_id, video_path in zip(frame_ids, video_paths):
            # Get GOP index for this video
            gop_infos = self._load_persistent_index(video_path)
            if not gop_infos:
                print(f"Error: No GOP index found for video {video_path}")
                torch.cuda.nvtx.range_pop()
                return None

            # Find the GOP containing the target frame
            gop_info = self._find_gop_for_frame(gop_infos, frame_id)
            if gop_info is None:
                print(f"Error: Could not find GOP for video {video_path}, frame {frame_id}")
                torch.cuda.nvtx.range_pop()
                return None

            gop_file_paths.append(gop_info.file_path)

        # Load one GOP array per video file.
        try:
            gop_data_list = self.nv_gop_demuxer.LoadGopsToList(gop_file_paths)
            torch.cuda.nvtx.range_pop()
            return gop_data_list
        except Exception as e:
            print(f"Error loading GOPs: {e}")
            torch.cuda.nvtx.range_pop()
            return None

    # .. doc-marker-end: gop-storage-load

    # .. doc-marker-begin: gop-storage-load-fast
    def load_gops_fast(
        self, frame_ids: List[int], video_paths: List[str], fix_gop_size: int
    ) -> Optional[List[np.ndarray]]:
        """
        Fast path to load multiple GOPs assuming a fixed GOP size in filenames.

        This avoids scanning directories or loading indices by directly computing the
        GOP file path: gop.{first_frame_id}.{fix_gop_size}.bin where
        first_frame_id = (frame_id // fix_gop_size) * fix_gop_size.

        Args:
            frame_ids (List[int]): List of target frame indices.
            video_paths (List[str]): List of corresponding video file paths.
            fix_gop_size (int): Fixed GOP size used when storing, e.g., 30.

        Returns:
            Optional[List[np.ndarray]]: Per-video GOP arrays compatible with DecodeFromGOPListRGB,
                                       or None if any GOP file is not found.
        """
        torch.cuda.nvtx.range_push("load_gops_fast_manager")

        if fix_gop_size is None or fix_gop_size <= 0:
            print(f"Error: invalid fix_gop_size {fix_gop_size}")
            torch.cuda.nvtx.range_pop()
            return None

        if len(frame_ids) != len(video_paths):
            print(f"Error: frame_ids length ({len(frame_ids)}) != video_paths length ({len(video_paths)})")
            torch.cuda.nvtx.range_pop()
            return None

        gop_file_paths: List[str] = []

        for frame_id, video_path in zip(frame_ids, video_paths):
            # Compute GOP directory based on relative path
            video_relpath = os.path.relpath(video_path, self.video_base_path)
            gop_dir = os.path.join(self.gop_base_path, video_relpath)

            # Compute first_frame_id for this GOP
            if frame_id < 0:
                print(f"Error: negative frame_id {frame_id} for {video_path}")
                torch.cuda.nvtx.range_pop()
                return None

            first_frame_id = (frame_id // fix_gop_size) * fix_gop_size
            expected_name = f"gop.{first_frame_id}.{fix_gop_size}.bin"
            expected_path = os.path.join(gop_dir, expected_name)

            if os.path.exists(expected_path):
                gop_file_paths.append(expected_path)
                continue

            # Fallback: try to find any file that matches the first_frame_id with any GOP len (e.g., tail GOP)
            fallback_found = False
            try:
                for filename in os.listdir(gop_dir):
                    if filename.startswith(f"gop.{first_frame_id}.") and filename.endswith('.bin'):
                        gop_file_paths.append(os.path.join(gop_dir, filename))
                        fallback_found = True
                        break
            except Exception:
                pass

            if not fallback_found:
                print(
                    f"Error: GOP file not found for video {video_path}, frame {frame_id}. Expected {expected_path}"
                )
                torch.cuda.nvtx.range_pop()
                return None

        # Load GOPs as a list.
        try:
            gop_data_list = self.nv_gop_demuxer.LoadGopsToList(gop_file_paths)
            torch.cuda.nvtx.range_pop()
            return gop_data_list
        except Exception as e:
            print(f"Error loading GOPs (fast): {e}")
            torch.cuda.nvtx.range_pop()
            return None

    # .. doc-marker-end: gop-storage-load-fast

    def get_gop_stats(self, video_path: str) -> dict:
        """
        Get statistics about stored GOPs for a video.

        Args:
            video_path (str): Path to the video file

        Returns:
            dict: Dictionary containing GOP statistics
        """
        gop_infos = self._load_persistent_index(video_path)

        if not gop_infos:
            return {'total_gops': 0, 'total_frames': 0, 'avg_gop_size': 0, 'frame_ranges': []}

        total_frames = sum(gop.gop_len for gop in gop_infos)
        avg_gop_size = total_frames / len(gop_infos) if gop_infos else 0
        frame_ranges = [(gop.first_frame_id, gop.first_frame_id + gop.gop_len - 1) for gop in gop_infos]

        return {
            'total_gops': len(gop_infos),
            'total_frames': total_frames,
            'avg_gop_size': avg_gop_size,
            'frame_ranges': frame_ranges,
        }
