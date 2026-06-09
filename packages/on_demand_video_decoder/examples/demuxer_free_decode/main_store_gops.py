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

import os
import torch
import numpy as np
import random

random.seed(27)

import accvlab.on_demand_video_decoder as nvc

# Import the local gop_storage module
from gop_storage import GOPStorageManager

NUM_CAMERAS = 6
VIDEO_BASE_PATH = "/data/nuscenes/video_samples"
GOP_BASE_PATH = "/data/nuscenes/video_packats"


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


def group_shuffle_file_list_generator(test_cases_dir, list_num=1000):
    dirs = [d for d in os.listdir(test_cases_dir) if os.path.isdir(os.path.join(test_cases_dir, d))]

    if not dirs:
        print("Error: No test case directories found")
        return []

    file_lists = []
    for i in range(list_num):
        # Randomly select a directory
        path_base = os.path.join(test_cases_dir, random.choice(dirs))

        # Get all video files in the selected directory and sort them
        file_names = sorted([f for f in os.listdir(path_base) if f.endswith(('.mp4', '.MP4'))])

        if not file_names:
            print(f"Warning: No video files found in {path_base}, skipping...")
            continue

        files = [os.path.join(path_base, name) for name in file_names]
        file_lists.append(files)

    return file_lists


def shuffle_list_generator(start, end, step, element_num):
    res = []
    for i in range(start, end, step):
        tmp = [i] * element_num
        res.append(tmp)
    random.shuffle(res)
    return res


def demuxer_free_decoder_test_load(
    frame_idxs, file_lists, video_base_path, gop_base_path, use_persistent_index=True
):
    """
    Test loading GOP data using the GOPStorageManager interface.

    Args:
        frame_idxs: List of frame index lists
        file_lists: List of file path lists
        video_base_path: Base path for video files
        gop_base_path: Base path for GOP storage
        use_persistent_index: Whether to use persistent index files
    """
    storage = GOPStorageManager(
        video_base_path,
        gop_base_path,
        clip_size=len(frame_idxs[0]),
        use_persistent_index=use_persistent_index,
    )
    nv_gop_decoder = nvc.CreateGopDecoder(
        maxfiles=len(frame_idxs[0]),  # Maximum number of files to use
        iGpu=0,  # GPU ID
    )

    successful_loads = 0
    total_attempts = 0

    for frame_id_list, file_list in zip(frame_idxs, file_lists):
        total_attempts += 1
        print(f"\nProcessing batch {total_attempts}/{len(frame_idxs)}")
        print(f"  Frame IDs: {frame_id_list}")
        print(f"  Files: {[os.path.basename(f) for f in file_list]}")

        try:
            # Load GOP data using the GOPStorageManager interface - returns one array per video.
            gop_data_list = storage.load_gops(frame_id_list, file_list)

            if gop_data_list is None:
                print(
                    f"  Warning: Could not load GOP data for files {[os.path.basename(f) for f in file_list]} and frames {frame_id_list}"
                )
                continue

            total_bytes = sum(gop_data.size for gop_data in gop_data_list)
            print(f"  Loaded GOP data: {len(gop_data_list)} arrays, {total_bytes} bytes total")

            # Decode frames using DecodeFromGOPListRGB with the per-video data.
            decoded_frames = nv_gop_decoder.DecodeFromGOPListRGB(
                gop_data_list, file_list, frame_id_list, True  # as_bgr=True for RGB output
            )

            print(f"  Successfully decoded {len(decoded_frames)} frames")
            successful_loads += 1

            # Optional: Compare with OpenCV for validation
            if False:  # Set to True to enable validation
                print("  Validating against OpenCV...")
                target_tensor = [torch.unsqueeze(torch.as_tensor(df), 0) for df in decoded_frames]
                opencv_decoded = opencv_decode_bgr(file_list, frame_id_list)
                max_diff = diff(target_tensor, opencv_decoded, file_list, frame_id_list, len(frame_id_list))
                if max_diff == 0:
                    print("  ✅ Validation passed")
                else:
                    print(f"  ⚠️ Validation failed with max diff: {max_diff}")

        except Exception as e:
            print(f"  ❌ Error processing batch: {e}")
            import traceback

            traceback.print_exc()
            continue

    print(f"\n=== Summary ===")
    print(f"Successfully loaded: {successful_loads}/{total_attempts} batches")
    print(
        f"Success rate: {successful_loads/total_attempts*100:.1f}%"
        if total_attempts > 0
        else "No attempts made"
    )


# .. doc-marker-begin: preprocessing-main
def demuxer_free_decoder_test_store(video_base_path, gop_base_path):
    """
    Test storing GOP data using the GOPStorageManager interface.

    Args:
        video_base_path: Base path for video files
        gop_base_path: Base path for GOP storage
    """
    print(f"Cleaning existing GOP storage at: {gop_base_path}")
    cmd = f"rm -rf {gop_base_path}"
    os.system(cmd)

    print(f"Creating GOP storage manager...")
    storage = GOPStorageManager(video_base_path, gop_base_path, clip_size=1)

    print(f"Starting GOP extraction and storage...")
    storage.store_gops()

    print(f"GOP storage completed. Files saved to: {gop_base_path}")


# .. doc-marker-end: preprocessing-main


def main_test(video_base_path, gop_base_path, use_persistent_index=True):
    """
    Main test function for demuxer-free decoder.

    Args:
        video_base_path: Base path for video files
        gop_base_path: Base path for GOP storage
        use_persistent_index: Whether to use persistent index files
    """
    print("accvlab.on_demand_video_decoder GOP Storage Demo")
    print("=" * 50)
    print(f"Video base path: {video_base_path}")
    print(f"GOP base path: {gop_base_path}")
    print(f"Use persistent index: {use_persistent_index}")

    # Generate test cases
    frame_idx = shuffle_list_generator(0, 180, 7, NUM_CAMERAS)
    file_lists = group_shuffle_file_list_generator(video_base_path, len(frame_idx))

    if not file_lists:
        print("❌ No test files found. Please check video_base_path.")
        return

    print(f"Generated {len(file_lists)} test cases")

    # Run the load test
    demuxer_free_decoder_test_load(
        frame_idx, file_lists, video_base_path, gop_base_path, use_persistent_index=use_persistent_index
    )


if __name__ == "__main__":
    # Configuration
    video_base_path = VIDEO_BASE_PATH
    gop_base_path = GOP_BASE_PATH

    print("Starting GOP Storage Test...")
    print(f"Video path: {video_base_path}")
    print(f"GOP storage path: {gop_base_path}")

    # Step 1: Store GOP data
    print("\n" + "=" * 60)
    print("STEP 1: Storing GOP data")
    print("=" * 60)
    demuxer_free_decoder_test_store(video_base_path, gop_base_path)

    # Step 2: Load and test GOP data
    print("\n" + "=" * 60)
    print("STEP 2: Loading and testing GOP data")
    print("=" * 60)
    main_test(video_base_path, gop_base_path, use_persistent_index=False)

    print("\n" + "=" * 60)
    print("GOP Storage Test Completed!")
    print("=" * 60)
