"""Regression coverage for grouped decoder resolution-aware slot reuse."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

av = pytest.importorskip("av")

from accvlab import on_demand_video_decoder as nvc  # noqa: E402


def _write_h264_clip(path: Path, *, width: int, height: int) -> None:
    try:
        output = av.open(str(path), mode="w")
        stream = output.add_stream("libx264", rate=10)
    except (av.AVError, ValueError) as exc:
        pytest.skip(f"PyAV libx264 encoder is unavailable: {exc}")
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.gop_size = 4
    stream.codec_context.max_b_frames = 0
    for frame_id in range(8):
        pixels = np.full((height, width, 3), frame_id * 17, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
        for packet in stream.encode(frame):
            output.mux(packet)
    for packet in stream.encode():
        output.mux(packet)
    output.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="grouped decode requires CUDA")
def test_grouped_decoder_replaces_single_slot_when_resolution_changes(tmp_path: Path) -> None:
    small_path = tmp_path / "small.mp4"
    large_path = tmp_path / "large.mp4"
    _write_h264_clip(small_path, width=64, height=64)
    _write_h264_clip(large_path, width=192, height=128)

    demuxer = nvc.CreateGopDecoder(maxfiles=1, iGpu=0)
    decoder = nvc.CreateGopDecoder(maxfiles=1, iGpu=0)
    requests = [
        (small_path, (64, 64, 3)),
        (large_path, (128, 192, 3)),
    ] * 10
    for path, expected_shape in requests:
        groups = demuxer.GetGOPGroups([{"filepath": str(path), "frame_ids": [0]}])
        decoded = decoder.DecodeFromGOPGroupsRGB(groups)
        assert decoded[0]["frames"][0].shape == expected_shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="grouped decode requires CUDA")
def test_grouped_decoder_pool_keeps_inactive_shape_slots(tmp_path: Path) -> None:
    small_path = tmp_path / "small.mp4"
    large_path = tmp_path / "large.mp4"
    _write_h264_clip(small_path, width=64, height=64)
    _write_h264_clip(large_path, width=192, height=128)

    demuxer = nvc.CreateGopDecoder(maxfiles=2, iGpu=0)
    decoder = nvc.CreateGopDecoder(maxfiles=2, iGpu=0)
    requests = [
        (small_path, (64, 64, 3)),
        (large_path, (128, 192, 3)),
    ] * 10
    for path, expected_shape in requests:
        groups = demuxer.GetGOPGroups([{"filepath": str(path), "frame_ids": [0]}])
        decoded = decoder.DecodeFromGOPGroupsRGB(groups)
        assert decoded[0]["frames"][0].shape == expected_shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="grouped decode requires CUDA")
def test_grouped_decoder_pool_matches_shape_when_group_order_changes(tmp_path: Path) -> None:
    small_a_path = tmp_path / "small_a.mp4"
    small_b_path = tmp_path / "small_b.mp4"
    large_path = tmp_path / "large.mp4"
    _write_h264_clip(small_a_path, width=64, height=64)
    _write_h264_clip(small_b_path, width=64, height=64)
    _write_h264_clip(large_path, width=192, height=128)

    demuxer = nvc.CreateGopDecoder(maxfiles=3, iGpu=0)
    decoder = nvc.CreateGopDecoder(maxfiles=3, iGpu=0)
    requests = [
        (
            [small_a_path, large_path, small_b_path],
            [(64, 64, 3), (128, 192, 3), (64, 64, 3)],
        ),
        (
            [large_path, small_b_path, small_a_path],
            [(128, 192, 3), (64, 64, 3), (64, 64, 3)],
        ),
    ] * 10
    for paths, expected_shapes in requests:
        groups = demuxer.GetGOPGroups(
            [{"filepath": str(path), "frame_ids": [0]} for path in paths]
        )
        decoded = decoder.DecodeFromGOPGroupsRGB(groups)
        actual_shapes = [group["frames"][0].shape for group in decoded]
        assert actual_shapes == expected_shapes
