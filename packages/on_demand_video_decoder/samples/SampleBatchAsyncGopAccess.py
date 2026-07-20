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
``accvlab.on_demand_video_decoder`` - 2D Batch Async GOP Access Sample

Demonstrates :class:`~accvlab.on_demand_video_decoder.PyNvBatchAsyncGopDecoder`:
an async 2D GOP-based decoder that submits V videos × F frames per call and
retrieves results with a separate blocking call, enabling prefetch overlap.

Key differences from ``SampleBatchAsyncStreamAccess.py`` (the stream-based sample):

- Caller supplies serialized GOP bundles (``numpy_datas``) obtained from
  :func:`GetGOPList`.  These bundles are indexable structures that can be
  pre-fetched, cached, or served from a dataloader worker — the decode GPU
  work is fully decoupled from I/O.
- ``numpy_datas[v]`` is a deduplicated list: frames in the same GOP share one
  bundle, so multiple frames from the same GOP do not duplicate data.
- Submit via ``DecodeFromGOPListRGB`` / collect via ``DecodeFromGOPListRGBGetBuffer``
  (the split submit/collect API mirrors ``PyNvBatchAsyncStreamReader.Decode`` /
  ``GetBuffer`` but operates on pre-fetched GOP bundles).
- YUV output is also available via ``DecodeFromGOPList`` /
  ``DecodeFromGOPListGetBuffer``.

Typical production topology
---------------------------
::

    Dataloader worker (CPU):
        GetGOPList(filepaths, frame_ids)  →  numpy_datas  (serializable, cacheable)

    GPU decode thread (this class):
        DecodeFromGOPListRGB(numpy_datas_N, ...)    ← submit batch N
        DecodeFromGOPListRGBGetBuffer(...)           ← block on batch N-1
"""

import os
import numpy as np
import torch
import accvlab.on_demand_video_decoder as nvc

# ---------------------------------------------------------------------------
# Helper: build numpy_datas for one batch
# ---------------------------------------------------------------------------


def _get_numpy_datas_for_video(gop_dec, filepath, frame_ids):
    """Return deduplicated GOP bundles covering every frame_id for one video.

    Calls GetGOPList once per frame_id and deduplicates by GOP start frame so
    that frames sharing a GOP contribute only one bundle.

    Returns:
        List[np.ndarray]: one uint8 array per unique GOP touched, sorted by
        GOP start frame.
    """
    seen = {}  # gop_start_frame -> numpy_data
    for fid in frame_ids:
        numpy_data, first_frame_ids, _gop_lens = gop_dec.GetGOPList([filepath], [fid])[0]
        key = int(first_frame_ids[0])
        if key not in seen:
            seen[key] = numpy_data
    return [seen[k] for k in sorted(seen.keys())]


def build_numpy_datas(gop_dec, filepaths, frame_ids_2d):
    """Build numpy_datas[v] for a full V-video batch.

    In production this step runs in a dataloader worker (CPU) and its output
    is passed to the GPU decoder.  The bundles are plain numpy arrays and are
    fully serializable / cacheable.

    Args:
        gop_dec: PyNvGopDecoder instance used only for GOP extraction.
        filepaths: List[str] of length V.
        frame_ids_2d: List[List[int]] shaped [V][F].

    Returns:
        List[List[np.ndarray]] shaped [V][num_gops_v].
    """
    return [_get_numpy_datas_for_video(gop_dec, filepaths[v], frame_ids_2d[v]) for v in range(len(filepaths))]


# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------


def SampleBatchAsyncGopAccess():
    """
    Show the canonical prefetch pattern with the 2D async GOP decoder.

    Per iteration:
        iter 0:  build numpy_datas(0) -> DecodeFromGOPListRGB(0)
                 -> DecodeFromGOPListRGBGetBuffer(0) -> clone -> process
                 + build numpy_datas(1) -> DecodeFromGOPListRGB(1)   [prefetch]
        iter i:  DecodeFromGOPListRGBGetBuffer(i) -> clone -> process
                 + build numpy_datas(i+1) -> DecodeFromGOPListRGB(i+1)

    Each "batch" is V videos × F frames (a 2D request).
    """

    # ── Configuration ─────────────────────────────────────────────────────
    max_num_files_to_use = 5
    max_frames_per_decode_call = 4

    # ── Resolve sample clip paths ────────────────────────────────────────
    base_dir = os.path.dirname(__file__)
    sample_clip_dir = os.path.join(base_dir, "..", "data", "sample_clip")
    file_path_list = [
        os.path.join(sample_clip_dir, "moving_shape_circle_h265.mp4"),
        os.path.join(sample_clip_dir, "moving_shape_ellipse_h265.mp4"),
        os.path.join(sample_clip_dir, "moving_shape_hexagon_h265.mp4"),
        os.path.join(sample_clip_dir, "moving_shape_rect_h265.mp4"),
        os.path.join(sample_clip_dir, "moving_shape_triangle_h265.mp4"),
    ]
    V = len(file_path_list)
    F = max_frames_per_decode_call

    print("Initializing decoders...")

    # GOP extractor — used only for GetGOPList (CPU-side bundle extraction).
    # In production this lives in a dataloader worker and is separate from the
    # GPU decode process.
    gop_extractor = nvc.CreateGopDecoder(maxfiles=V, iGpu=0)

    # 2D async GOP decoder — GPU decode path.
    dec = nvc.CreateBatchAsyncGopDecoder(
        maxfiles=max_num_files_to_use,
        max_frames_per_decode_call=max_frames_per_decode_call,
        iGpu=0,
    )
    print(
        f"GOP decoder initialized on GPU 0 — V <= {max_num_files_to_use} videos, "
        f"F <= {max_frames_per_decode_call} frames per video per call"
    )

    # ── Build the 2D frame_id schedule ───────────────────────────────────
    # In production this comes from the dataset sampler. The fixed stride here
    # keeps adjacent frames within the same GOP so numpy_datas[v] has 1 entry;
    # see the multi-GOP section below for the cross-GOP case.
    num_iterations = 4
    step = 1  # consecutive frames → same GOP → 1 bundle per video

    def make_batch(iter_idx):
        start = iter_idx * F
        return [[start + f * step for f in range(F)] for _ in range(V)]

    print(f"\nStarting {num_iterations} prefetched 2D-batch iterations")
    print("Pattern: GetGOPList(N) + DecodeFromGOPListRGB(N) overlap with GetBuffer(N-1)")

    # ── Main loop ────────────────────────────────────────────────────────
    for idx in range(num_iterations):
        frame_ids_2d = make_batch(idx)
        print(f"\n--- Iteration {idx + 1}/{num_iterations} ---")
        print(f"Frame ids (V×F = {V}×{F}): {frame_ids_2d[0]}")

        try:
            # Build GOP bundles for the current batch.
            # In production: dataloader worker does this and passes numpy_datas
            # to the collate function. The bundles are plain numpy arrays.
            numpy_datas = build_numpy_datas(gop_extractor, file_path_list, frame_ids_2d)
            print(f"GOP bundles per video: {[len(nd) for nd in numpy_datas]}")

            if idx == 0:
                # First iteration: submit and immediately collect.
                print("[Async] Submitting initial decode for batch 0")
                dec.DecodeFromGOPListRGB(numpy_datas, file_path_list, frame_ids_2d, False)
                print("[Async] Retrieving batch 0 from buffer")
                decoded = dec.DecodeFromGOPListRGBGetBuffer(file_path_list, frame_ids_2d, False)
            else:
                # Subsequent iterations: result was prefetched at end of the
                # previous iteration; GetBuffer blocks only if the worker hasn't
                # finished yet.
                print(f"[Async] Retrieving prefetched batch {idx} from buffer")
                decoded = dec.DecodeFromGOPListRGBGetBuffer(file_path_list, frame_ids_2d, False)

            assert len(decoded) == V
            for v in range(V):
                assert len(decoded[v]) == F, f"unexpected F at v={v}: {len(decoded[v])}"

            # ── CRITICAL: clone before the next DecodeFromGOPListRGB() ──
            #
            # decoded[v][f] is a zero-copy RGBFrame backed by the decoder's
            # internal aggregator pool.  That pool is overwritten on the next
            # DecodeFromGOPListRGB() call.  Clone every frame you need to keep
            # *before* submitting the next batch.
            print("Cloning V×F frames to PyTorch tensors")
            tensor_grid = [
                [torch.as_tensor(decoded[v][f], device="cuda").clone() for f in range(F)] for v in range(V)
            ]

            # Stack into [V, F, H, W, 3] for model consumption.
            batch = torch.stack([torch.stack(row, dim=0) for row in tensor_grid], dim=0)
            print(f"Batch shape: {tuple(batch.shape)}, dtype: {batch.dtype}, device: {batch.device}")
            print(f"Value range: [{batch.min().item()}, {batch.max().item()}]")

            # ── Prefetch the next batch ─────────────────────────────────
            # DecodeFromGOPListRGB returns immediately; the worker decodes
            # batch N+1 in the background while we "process" batch N.
            if idx < num_iterations - 1:
                next_frame_ids_2d = make_batch(idx + 1)
                next_numpy_datas = build_numpy_datas(gop_extractor, file_path_list, next_frame_ids_2d)
                print(f"[Async] Prefetching batch {idx + 1} (frames {next_frame_ids_2d[0]})")
                dec.DecodeFromGOPListRGB(next_numpy_datas, file_path_list, next_frame_ids_2d, False)

            # ── Simulated "process" stage (model forward, etc.) ─────────
            print("[Processing] (simulated) — prefetched decode is running in parallel")

        except Exception as e:
            print(f"Decode failed in iteration {idx + 1}: {type(e).__name__}: {e}")
            print("Common causes:")
            print("  - frame_id not covered by any supplied GOP bundle")
            print("  - filepaths / frame_ids mismatch between submit and GetBuffer")
            print("  - frame count exceeds max_frames_per_decode_call")
            print("  - insufficient GPU memory")
            continue

    # ── Multi-GOP example ────────────────────────────────────────────────
    # When requested frames span more than one GOP, numpy_datas[v] must contain
    # one bundle per GOP touched.  build_numpy_datas() deduplicates by GOP start
    # frame automatically, so the caller only needs to supply all relevant frames
    # and let the helper figure out how many bundles are needed.
    #
    # We probe the first GOP's length from the first video to find two frames
    # that are guaranteed to be in different GOPs, regardless of GOP size.
    print("\n--- Multi-GOP example: two frames from different GOPs ---")
    try:
        _, first_frame_ids_probe, gop_lens_probe = gop_extractor.GetGOPList([file_path_list[0]], [0])[0]
        gop0_start = int(first_frame_ids_probe[0])
        gop0_len = int(gop_lens_probe[0])
        frame_in_gop0 = gop0_start
        frame_in_gop1 = gop0_start + gop0_len  # first frame of the next GOP
        print(
            f"GOP 0: frames [{gop0_start}, {gop0_start + gop0_len - 1}]  "
            f"-> using frames {frame_in_gop0} and {frame_in_gop1}"
        )

        multi_gop_frame_ids = [[frame_in_gop0, frame_in_gop1]] * V
        multi_gop_dec = nvc.CreateBatchAsyncGopDecoder(maxfiles=V, max_frames_per_decode_call=2, iGpu=0)
        nd = build_numpy_datas(gop_extractor, file_path_list, multi_gop_frame_ids)
        assert all(len(x) == 2 for x in nd), f"expected 2 GOP bundles per video, got {[len(x) for x in nd]}"
        print(f"GOP bundles per video (expect 2): {[len(x) for x in nd]}")
        multi_gop_dec.DecodeFromGOPListRGB(nd, file_path_list, multi_gop_frame_ids, False)
        out = multi_gop_dec.DecodeFromGOPListRGBGetBuffer(file_path_list, multi_gop_frame_ids, False)
        t0 = torch.as_tensor(out[0][0], device="cuda")
        t1 = torch.as_tensor(out[0][1], device="cuda")
        print(
            f"Frame {frame_in_gop0} shape: {tuple(t0.shape)}, Frame {frame_in_gop1} shape: {tuple(t1.shape)}"
        )
        print("Multi-GOP decode OK")
    except Exception as e:
        print(f"Multi-GOP example failed: {type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    print("2D async batch GOP decoding completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    print("NVIDIA accvlab.on_demand_video_decoder — 2D Batch Async GOP Sample")
    print("=" * 70)
    print()
    SampleBatchAsyncGopAccess()
