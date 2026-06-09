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
DataLoader Demuxer-Free Example

This example mirrors example/dataloader_separation_decode/main.py but replaces
packet fetching via demuxer with loading pre-stored GOP packet data from disk
using GOPStorageManager, then decodes using ``accvlab.on_demand_video_decoder``.

Prerequisite: Generate GOP packet files and indices using
example/dataloader_decode_only/main_store_gops.py
"""

import os
import time
import argparse
import json
import logging
import sys
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.cuda.nvtx as nvtx
from torch.utils.data import Dataset, DataLoader, Sampler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'examples'))
import dataloader_separation_decode.video_transforms as video_transforms
import dataloader_separation_decode.video_clip_sampler as video_clip_sampler
from gop_storage import GOPStorageManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set multiprocessing start method
mp.set_start_method('fork', force=True)


NUM_CAMERAS = 6
DEFAULT_WARMUP_ITERATIONS = 10
DEFAULT_LOG_INTERVAL = 10


def load_index_frame(json_file: str) -> Dict[str, Dict[str, int]]:
    try:
        with open(json_file, 'r') as f:
            index_frame = json.load(f)
        logger.info(f"Successfully loaded index frame from {json_file}")
        return index_frame
    except FileNotFoundError:
        logger.error(f"Index file not found: {json_file}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in {json_file}: {e}")
        raise


def is_distributed() -> bool:
    return "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ


def setup_distributed() -> Tuple[int, int]:
    if is_distributed():
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        logger.info(f"Distributed mode - Local rank: {local_rank}, World size: {world_size}")
        return local_rank, world_size
    else:
        logger.info("Single process mode")
        return 0, 1


def cleanup_distributed():
    if is_distributed() and dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed process group destroyed")


# .. doc-marker-begin: dataset-decode-only
class VideoClipDatasetDecodeOnly(Dataset):
    """
    Dataset that loads pre-stored GOP packet data for the requested frames and clips
    using GOPStorageManager, returning buffers suitable for direct decoding.
    """

    def __init__(
        self,
        index_frame: Dict[str, Dict[str, int]],
        group_num: int,
        video_base_path: str,
        gop_base_path: str,
        num_cameras: int = NUM_CAMERAS,
        use_persistent_index: bool = True,
        fix_gop_size: int = 0,
    ):
        self.index_frame = index_frame
        self.group_num = group_num
        self.video_base_path = video_base_path
        self.gop_base_path = gop_base_path
        self.num_cameras = num_cameras
        self.use_persistent_index = use_persistent_index
        self.fix_gop_size = fix_gop_size

        self._is_initialized = False
        self._storage: GOPStorageManager = None  # type: ignore

        if group_num <= 0:
            raise ValueError(f"group_num must be positive, got {group_num}")
        if num_cameras <= 0:
            raise ValueError(f"num_cameras must be positive, got {num_cameras}")

        logger.info(f"Initialized demuxer-free dataset with {num_cameras} cameras, group_num={group_num}")

    def __len__(self) -> int:
        total_frames = 0
        for _, clip_info in self.index_frame.items():
            for _, frame_count in clip_info.items():
                total_frames += frame_count
        return total_frames // self.group_num

    def __lazy_init__(self):
        if self._is_initialized:
            return
        self._storage = GOPStorageManager(
            video_base_path=self.video_base_path,
            gop_base_path=self.gop_base_path,
            clip_size=self.num_cameras,
            use_persistent_index=self.use_persistent_index,
        )
        self._is_initialized = True

    def __getitem__(self, index: List[Tuple[str, int]]) -> List[Any]:
        if not isinstance(index, list):
            raise ValueError(f"Expected list of tuples, got {type(index)}")
        self.__lazy_init__()

        episode_buffers = []

        for i, (clip_path, frame_idx, _) in enumerate(index):
            # Find all MP4 files in the clip directory
            if not os.path.isdir(clip_path):
                logger.warning(f"Clip path is not a directory: {clip_path}")
                continue

            video_paths = sorted(
                [os.path.join(clip_path, f) for f in os.listdir(clip_path) if f.endswith('.mp4')],
                key=lambda x: os.path.basename(x),
            )

            if not video_paths:
                logger.warning(f"No MP4 files found in {clip_path}")
                continue

            frame_indices = [frame_idx] * len(video_paths)

            if self.fix_gop_size > 0:
                gop_data_list = self._storage.load_gops_fast(
                    frame_indices, video_paths, self.fix_gop_size
                )
            else:
                gop_data_list = self._storage.load_gops(frame_indices, video_paths)

            if gop_data_list is None:
                logger.warning(f"Failed to load GOP packets for {clip_path} at frame {frame_idx}")
                continue

            episode_buffers.append(
                video_transforms.PacketOndemandBuffers(
                    gop_packets=gop_data_list,
                    target_frame_list=frame_indices,
                    target_file_list=video_paths,
                    use_cache=False,
                    sample_idx=i,
                )
            )

        return episode_buffers


# .. doc-marker-end: dataset-decode-only


def run_warmup(
    dataloader: DataLoader,
    decoder: video_transforms.DecodeVideoOnDemand,
    warmup_iterations: int = DEFAULT_WARMUP_ITERATIONS,
) -> None:
    logger.info(f"Starting warmup with {warmup_iterations} iterations")
    loader_iter = iter(dataloader)
    for i in range(warmup_iterations):
        try:
            nvtx.range_push(f"warmup_batch_{i}")
            nvtx.range_push("warmup_next_batch")
            batch = next(loader_iter)
            nvtx.range_pop()

            nvtx.range_push("warmup_decode_batch")
            decoder.transform(batch)
            nvtx.range_pop()
            nvtx.range_pop()
        except StopIteration:
            logger.warning(f"DataLoader exhausted during warmup at iteration {i}")
            break
        except Exception as e:
            logger.error(f"Error during warmup iteration {i}: {e}")
            continue
    logger.info("Warmup completed")


def run_benchmark(
    dataloader: DataLoader,
    decoder: video_transforms.DecodeVideoOnDemand,
    group_num: int,
    log_interval: int = DEFAULT_LOG_INTERVAL,
) -> Dict[str, float]:
    logger.info("Starting benchmark")
    loader_iter = iter(dataloader)

    started_at = time.perf_counter()
    elapsed_time = 0
    frame_loaded = 0
    samples_loaded = 0
    batches_loaded = 0
    load_gaps: List[float] = []
    load_started_at = time.perf_counter()
    i = 0

    while True:
        nvtx.range_push(f"batch_{i}")
        nvtx.range_push("next_batch")
        try:
            batch = next(loader_iter)
        except StopIteration:
            logger.info("DataLoader exhausted, benchmark complete")
            break
        nvtx.range_pop()

        nvtx.range_push("decode_batch")
        try:
            _ = decoder.transform(batch)
        except Exception as e:
            logger.error(f"Error decoding batch {i}: {e}")
            continue
        nvtx.range_pop()

        nvtx.range_push("load_batch")
        load_ended_at = time.perf_counter()
        load_gaps.append(load_ended_at - load_started_at)
        load_started_at = time.perf_counter()

        batches_loaded += 1
        samples_loaded += group_num
        frame_loaded += NUM_CAMERAS * group_num

        current_time = time.perf_counter()
        if i % log_interval == 0:
            elapsed_time = current_time - started_at
            throughput_fps = frame_loaded / elapsed_time if elapsed_time > 0 else 0
            logger.info(f"Batch: {i} - {len(batch)} frames in {load_gaps[-1]:.4f} seconds")
            logger.info(f"Current Time taken: {elapsed_time:.4f}s")
            logger.info(f"Current Samples loaded: {samples_loaded}")
            logger.info(f"Current Throughput: {throughput_fps:.2f} frames per second")

        nvtx.range_pop()
        nvtx.range_pop()
        i += 1

    # ended_at = time.perf_counter()
    ended_at = current_time
    total_time = ended_at - started_at
    metrics = {
        'frames_loaded': frame_loaded,
        'samples_loaded': samples_loaded,
        'batches_loaded': batches_loaded,
        'total_time': total_time,
        'throughput_fps': frame_loaded / total_time if total_time > 0 else 0,
        'throughput_sps': samples_loaded / total_time if total_time > 0 else 0,
        'throughput_bps': batches_loaded / total_time if total_time > 0 else 0,
        'avg_batch_time': sum(load_gaps) / len(load_gaps) if load_gaps else 0,
    }
    return metrics


def print_performance_summary(metrics: Dict[str, float], num_workers: int) -> None:
    print("\n" + "=" * 50)
    print("PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"Frames loaded: {metrics['frames_loaded']:,}")
    print(f"Samples loaded: {metrics['samples_loaded']:,}")
    print(f"Batches loaded: {metrics['batches_loaded']:,}")
    print(f"Time taken: {metrics['total_time']:.2f} seconds")
    print(f"Throughput: {metrics['throughput_fps']:.2f} frames per second")
    print(f"Throughput: {metrics['throughput_sps']:.2f} samples per second")
    print(f"Throughput: {metrics['throughput_bps']:.2f} batches per second")
    print(f"Average batch load time: {metrics['avg_batch_time']:.4f} seconds")
    print(f"Number of workers: {num_workers}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="``accvlab.on_demand_video_decoder`` DataLoader Demuxer-Free Example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--index_file",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "index_frame.json"),
        help='Path to the index_frame JSON file',
    )
    parser.add_argument(
        "--group_num",
        type=int,
        default=4,
        help='Number of clips to process in each batch',
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help='Number of worker processes for data loading',
    )
    parser.add_argument(
        "--video_base_path",
        type=str,
        default='/data/nuscenes/video_samples/',
        help='Base directory containing video files used to build relative GOP storage paths',
    )
    parser.add_argument(
        "--gop_base_path",
        type=str,
        default='/data/nuscenes/video_packats/',
        help='Base directory containing stored GOP packet files (generated by main_store_gops.py)',
    )
    parser.add_argument(
        "--use_persistent_index",
        action='store_true',
        default=True,
        help='Use persistent .gop_index.json files for faster lookup',
    )
    parser.add_argument(
        "--warmup_iterations",
        type=int,
        default=DEFAULT_WARMUP_ITERATIONS,
        help='Number of warmup iterations',
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=DEFAULT_LOG_INTERVAL,
        help='Interval for logging progress',
    )
    parser.add_argument(
        "--fix_gop_size",
        type=int,
        default=0,
        help='Fixed GOP size used when storing packets (e.g., 30). If > 0, dataloader uses fast path to compute GOP filenames directly (gop.{first}.{fix}.bin) without scanning/index lookup. If 0, falls back to generic load_gops with index/scan.',
    )
    parser.add_argument(
        "--frame_read_type",
        type=str,
        choices=['stream', 'random'],
        default='stream',
        help='Frame reading type: stream (sequential) or random (random sampling)',
    )

    args = parser.parse_args()

    if not os.path.exists(args.index_file):
        logger.error(f"Index file not found: {args.index_file}")
        return 1
    if not os.path.isdir(args.video_base_path):
        logger.error(f"video_base_path does not exist: {args.video_base_path}")
        return 1
    if not os.path.isdir(args.gop_base_path):
        logger.error(f"gop_base_path does not exist: {args.gop_base_path}")
        return 1

    logger.info(f"Using index_file: {args.index_file}")
    logger.info(f"Using group_num: {args.group_num}")
    logger.info(f"Using num_workers: {args.num_workers}")
    logger.info(f"Using video_base_path: {args.video_base_path}")
    logger.info(f"Using gop_base_path: {args.gop_base_path}")
    logger.info(f"Use persistent index: {args.use_persistent_index}")

    try:
        # .. doc-marker-begin: training-setup
        local_rank, local_world_size = setup_distributed()

        index_frame = load_index_frame(args.index_file)

        dataset = VideoClipDatasetDecodeOnly(
            index_frame=index_frame,
            group_num=args.group_num,
            video_base_path=args.video_base_path,
            gop_base_path=args.gop_base_path,
            num_cameras=NUM_CAMERAS,
            use_persistent_index=args.use_persistent_index,
            fix_gop_size=args.fix_gop_size,
        )

        if args.frame_read_type == 'stream':
            sampler = video_clip_sampler.VideoClipSamplerStream(
                index_frame=index_frame,
                group_num=args.group_num,
                rank=local_rank,
                world_size=local_world_size,
            )
        elif args.frame_read_type == 'random':
            sampler = video_clip_sampler.VideoClipSamplerRandom(
                index_frame=index_frame,
                group_num=args.group_num,
                rank=local_rank,
                world_size=local_world_size,
            )
        else:
            raise ValueError(f"Unknown frame_read_type: {args.frame_read_type}")

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=False,
        )

        decoder = video_transforms.DecodeVideoOnDemand(device_id=local_rank, num_cameras=NUM_CAMERAS)

        run_warmup(dataloader, decoder, args.warmup_iterations)

        metrics = run_benchmark(dataloader, decoder, args.group_num, args.log_interval)

        print_performance_summary(metrics, args.num_workers)
        # .. doc-marker-end: training-setup
        return 0
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    exit(main())
