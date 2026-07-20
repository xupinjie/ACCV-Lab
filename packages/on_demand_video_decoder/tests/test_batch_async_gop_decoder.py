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
Tests for ``PyNvBatchAsyncGopDecoder`` (2D async GOP-based decoder).

Layout:
    Section A — construction / module exports
    Section B — input validation
    Section C — maintenance methods
    Section D — functional RGB decode (shape, dtype, device)
    Section E — functional YUV decode (shape, views, format)
    Section F — precision: 2D RGB output must bit-match 1D GOP decode reference
    Section G — frame order preservation (unsorted frame_ids)
    Section H — multi-GOP bundles (frames spanning multiple GOPs)
    Section I — async behavior: error propagation, mismatch, prefetch loop
"""

import pytest
import numpy as np
import threading
import torch

import utils
import accvlab.on_demand_video_decoder as nvc

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

# GOP interval large enough to ensure adjacent frames stay in the same GOP.
_SAME_GOP_FRAMES = [0, 1, 2, 3]
# Frames intentionally placed far apart to span two GOPs (typical GOP = ~30 frames).
_MULTI_GOP_FRAMES = [0, 60]


def _sample_files():
    files = utils.select_random_clip(utils.get_data_dir())
    assert files is not None and len(files) > 0, "test data missing"
    return files


def _make_gop_dec():
    """Shared synchronous 1D GOP decoder used for GOP bundle extraction and reference."""
    return nvc.CreateGopDecoder(maxfiles=8, iGpu=0)


def _make_async_dec(V=4, F=8):
    return nvc.CreateBatchAsyncGopDecoder(maxfiles=V, max_frames_per_decode_call=F, iGpu=0)


def _get_numpy_data(gop_dec, filepath, frame_id):
    """Return the serialized GOP bundle (numpy uint8 array) covering *frame_id* in *filepath*."""
    gop_data, _first_frame_ids, _gop_lens = gop_dec.GetGOPList([filepath], [frame_id])[0]
    return gop_data


def _get_numpy_datas_for_video(gop_dec, filepath, frame_ids):
    """Return deduplicated serialized GOP bundles covering every frame_id in frame_ids for one video.

    Calls GetGOPList once per frame_id and deduplicates by GOP start frame so
    that frames in the same GOP share one bundle (avoiding redundant copies).

    Returns:
        list of unique numpy uint8 arrays, one per GOP touched.
    """
    seen = {}  # gop_start_frame -> gop_data
    for fid in frame_ids:
        gop_data, first_frame_ids, _gop_lens = gop_dec.GetGOPList([filepath], [fid])[0]
        key = int(first_frame_ids[0])
        if key not in seen:
            seen[key] = gop_data
    return [seen[k] for k in sorted(seen.keys())]


def _build_numpy_datas(gop_dec, files, frame_ids_2d):
    """Build numpy_datas[v] = List[serialized GOP bundle] covering all frames for each video."""
    return [_get_numpy_datas_for_video(gop_dec, files[v], frame_ids_2d[v]) for v in range(len(files))]


def _ref_rgb(gop_dec_ref, filepath, gop_data, frame_id, as_bgr):
    """Synchronous 1D GOP RGB decode for one (video, frame) pair.

    Returns a cloned CUDA tensor so the result survives subsequent gop_dec calls.
    """
    frames = gop_dec_ref.DecodeFromGOPListRGB([gop_data], [filepath], [frame_id], as_bgr=as_bgr)
    return torch.as_tensor(frames[0], device="cuda").clone()


# ===========================================================================
# Section A — construction / module exports
# ===========================================================================


def test_module_exports():
    """Factory and class are re-exported at the package top level."""
    assert hasattr(nvc, "CreateBatchAsyncGopDecoder")
    assert hasattr(nvc, "PyNvBatchAsyncGopDecoder")


def test_construct_valid():
    """Construction with valid args succeeds and exposes expected methods."""
    dec = _make_async_dec()
    methods = {m for m in dir(dec) if not m.startswith("_")}
    expected = {
        "DecodeFromGOPListRGB",
        "DecodeFromGOPListRGBGetBuffer",
        "DecodeFromGOPList",
        "DecodeFromGOPListGetBuffer",
        "release_device_memory",
        "release_decoder",
    }
    assert expected.issubset(methods), f"missing methods: {expected - methods}"


def test_destructor_clean():
    """Decoder destructs cleanly with no pending task."""
    dec = _make_async_dec()
    del dec


def test_rejects_direct_construction():
    """PyNvBatchAsyncGopDecoder must be created via the factory function."""
    with pytest.raises(TypeError):
        nvc.PyNvBatchAsyncGopDecoder()


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(maxfiles=0, max_frames_per_decode_call=1),
        dict(maxfiles=-1, max_frames_per_decode_call=1),
        dict(maxfiles=1, max_frames_per_decode_call=0),
        dict(maxfiles=1, max_frames_per_decode_call=-1),
    ],
)
def test_construct_rejects_invalid_args(kwargs):
    """Non-positive sizing arguments are rejected at construction."""
    with pytest.raises((ValueError, RuntimeError)):
        nvc.CreateBatchAsyncGopDecoder(**kwargs)


# ===========================================================================
# Section B — input validation
# ===========================================================================


def test_validate_empty_filepaths():
    """Empty filepaths list is rejected before any decode work."""
    dec = _make_async_dec()
    with pytest.raises(RuntimeError, match="filepaths must not be empty"):
        dec.DecodeFromGOPListRGB([], [], [], False)


def test_validate_too_many_files():
    """More filepaths than num_of_file is rejected."""
    files = _sample_files()
    dec = _make_async_dec(V=1)  # only room for 1 video
    if len(files) <= 1:
        pytest.skip("need at least 2 sample videos")
    gop_dec = _make_gop_dec()
    numpy_datas = _build_numpy_datas(gop_dec, files, [[0]] * len(files))
    with pytest.raises(RuntimeError, match="exceeds maxfiles"):
        dec.DecodeFromGOPListRGB(numpy_datas, files, [[0]] * len(files), False)


def test_validate_too_many_frames():
    """More frames than max_frames_per_decode_call is rejected."""
    files = _sample_files()
    V = len(files)
    dec = _make_async_dec(V=V, F=4)
    gop_dec = _make_gop_dec()
    gop_data = _get_numpy_data(gop_dec, files[0], 0)
    numpy_datas = [[gop_data]] * V
    with pytest.raises(RuntimeError, match="exceeds max_frames_per_decode_call"):
        dec.DecodeFromGOPListRGB(numpy_datas, files, [list(range(100))] * V, False)


def test_validate_jagged_inner_lengths():
    """Inner frame_id lists of different lengths are rejected."""
    files = _sample_files()
    if len(files) < 2:
        pytest.skip("need at least 2 sample videos")
    V = len(files)
    dec = _make_async_dec(V=V)
    gop_dec = _make_gop_dec()
    numpy_datas = _build_numpy_datas(gop_dec, files, [[0, 1]] * V)
    jagged = [[0, 1]] + [[0]] * (V - 1)  # first video has 2 frames, rest have 1
    with pytest.raises(RuntimeError, match="jagged input not supported"):
        dec.DecodeFromGOPListRGB(numpy_datas, files, jagged, False)


def test_validate_empty_inner_frame_ids():
    """Empty inner frame_id list is rejected."""
    files = _sample_files()
    V = len(files)
    dec = _make_async_dec(V=V)
    gop_dec = _make_gop_dec()
    gop_data = _get_numpy_data(gop_dec, files[0], 0)
    numpy_datas = [[gop_data]] * V
    with pytest.raises(RuntimeError, match="must not be empty"):
        dec.DecodeFromGOPListRGB(numpy_datas, files, [[]] * V, False)


def test_validate_empty_blob_list_for_video():
    """Passing an empty serialized GOP bundle list for any video is rejected."""
    files = _sample_files()
    V = len(files)
    dec = _make_async_dec(V=V)
    gop_dec = _make_gop_dec()
    gop_data = _get_numpy_data(gop_dec, files[0], 0)
    # First video has a bundle, second video gets empty list
    numpy_datas = [[gop_data]] + [[]] * (V - 1)
    frame_ids_2d = [[0]] * V
    with pytest.raises(RuntimeError, match="GOP bundle is required"):
        dec.DecodeFromGOPListRGB(numpy_datas, files, frame_ids_2d, False)


def test_validate_outer_size_mismatch_blobs():
    """numpy_datas outer length != filepaths length is rejected."""
    files = _sample_files()
    V = len(files)
    dec = _make_async_dec(V=V)
    gop_dec = _make_gop_dec()
    gop_data = _get_numpy_data(gop_dec, files[0], 0)
    # One fewer bundle list than files
    numpy_datas = [[gop_data]] * (V - 1)
    with pytest.raises(RuntimeError):
        dec.DecodeFromGOPListRGB(numpy_datas, files, [[0]] * V, False)


def test_validate_frame_ids_outer_too_short():
    """frame_ids_2d outer length < len(filepaths) is rejected.

    filepaths supplies V video paths; frame_ids_2d supplies V-1 inner lists.
    The mismatch is caught at check 3 in validate_decode_input, before any
    frame-count or jagged checks run.
    """
    files = _sample_files()
    V = len(files)
    if V < 2:
        pytest.skip("need at least 2 sample videos")
    gop_dec = _make_gop_dec()
    gop_data = _get_numpy_data(gop_dec, files[0], 0)
    numpy_datas = [[gop_data]] * V
    # frame_ids_2d has one fewer entry than filepaths
    frame_ids_2d = [[0]] * (V - 1)
    dec = _make_async_dec(V=V)
    with pytest.raises(RuntimeError, match="frame_ids_2d outer length"):
        dec.DecodeFromGOPListRGB(numpy_datas, files, frame_ids_2d, False)


def test_validate_frame_ids_outer_too_long():
    """frame_ids_2d outer length > len(filepaths) is rejected.

    filepaths supplies V paths; frame_ids_2d supplies V+1 inner lists.
    The mismatch is caught at check 3 in validate_decode_input, before any
    frame-count or jagged checks run.
    """
    files = _sample_files()
    V = len(files)
    gop_dec = _make_gop_dec()
    gop_data = _get_numpy_data(gop_dec, files[0], 0)
    numpy_datas = [[gop_data]] * V
    # frame_ids_2d has one extra entry beyond filepaths
    frame_ids_2d = [[0]] * (V + 1)
    dec = _make_async_dec(V=V)
    with pytest.raises(RuntimeError, match="frame_ids_2d outer length"):
        dec.DecodeFromGOPListRGB(numpy_datas, files, frame_ids_2d, False)


def test_validate_numpy_datas_outer_too_large():
    """numpy_datas outer length > len(filepaths) is rejected.

    The existing mismatch test covers the too-small direction (V-1).
    This covers the too-large direction (V+1).
    """
    files = _sample_files()
    V = len(files)
    gop_dec = _make_gop_dec()
    gop_data = _get_numpy_data(gop_dec, files[0], 0)
    # One extra bundle list beyond what filepaths and frame_ids_2d expect
    numpy_datas = [[gop_data]] * (V + 1)
    dec = _make_async_dec(V=V)
    with pytest.raises(RuntimeError, match="numpy_datas outer length"):
        dec.DecodeFromGOPListRGB(numpy_datas, files, [[0]] * V, False)


def test_validate_zero_byte_bundle_propagates_async_error():
    """A zero-byte numpy array inside numpy_datas[v] slips past validate_decode_input.

    numpy_datas[v].empty() checks whether the per-video list is empty (len==0),
    NOT whether any individual bundle array is zero bytes.  A zero-size array
    makes the list non-empty so check 8 passes.  The error surfaces later, inside
    the async worker, when parseSerializedPacketData tries to read an empty buffer.
    It must be re-thrown at GetBuffer time, not silently swallowed.
    """
    files = _sample_files()
    V = len(files)
    empty_bundle = np.array([], dtype=np.uint8)
    # Each video gets a list with one zero-byte bundle — passes the "not empty" check.
    numpy_datas = [[empty_bundle]] * V
    dec = _make_async_dec(V=V, F=1)
    dec.DecodeFromGOPListRGB(numpy_datas, files, [[0]] * V, False)
    with pytest.raises(RuntimeError):
        dec.DecodeFromGOPListRGBGetBuffer(files, [[0]] * V, False)


# ===========================================================================
# Section C — maintenance methods
# ===========================================================================


def test_maintenance_idle_callable():
    """Maintenance methods are safe no-ops when no decode task is pending."""
    dec = _make_async_dec()
    dec.release_device_memory()
    dec.release_decoder()
    # Idempotent
    dec.release_device_memory()
    dec.release_decoder()


def test_maintenance_after_decode():
    """release_device_memory / release_decoder work after a completed decode."""
    files = _sample_files()
    V = len(files)
    gop_dec = _make_gop_dec()
    numpy_datas = _build_numpy_datas(gop_dec, files, [[0]] * V)
    dec = _make_async_dec(V=V)
    dec.DecodeFromGOPListRGB(numpy_datas, files, [[0]] * V, False)
    _ = dec.DecodeFromGOPListRGBGetBuffer(files, [[0]] * V, False)
    dec.release_device_memory()
    dec.release_decoder()


# ===========================================================================
# Section D — functional RGB decode (shape, dtype, device)
# ===========================================================================


def test_rgb_output_shape():
    """DecodeFromGOPListRGBGetBuffer returns List[List[RGBFrame]] [V][F]."""
    files = _sample_files()
    V = len(files)
    F = len(_SAME_GOP_FRAMES)
    frame_ids_2d = [_SAME_GOP_FRAMES] * V

    gop_dec = _make_gop_dec()
    numpy_datas = _build_numpy_datas(gop_dec, files, frame_ids_2d)

    dec = _make_async_dec(V=V, F=F)
    dec.DecodeFromGOPListRGB(numpy_datas, files, frame_ids_2d, False)
    out = dec.DecodeFromGOPListRGBGetBuffer(files, frame_ids_2d, False)

    assert len(out) == V
    for v in range(V):
        assert len(out[v]) == F, f"out[{v}] has {len(out[v])} frames, expected {F}"


def test_rgb_output_dtype_and_device():
    """Each RGBFrame converts to a uint8 (H, W, 3) CUDA tensor."""
    files = _sample_files()
    V = len(files)
    frame_ids_2d = [[0]] * V

    gop_dec = _make_gop_dec()
    numpy_datas = _build_numpy_datas(gop_dec, files, frame_ids_2d)

    dec = _make_async_dec(V=V, F=1)
    dec.DecodeFromGOPListRGB(numpy_datas, files, frame_ids_2d, False)
    out = dec.DecodeFromGOPListRGBGetBuffer(files, frame_ids_2d, False)

    for v in range(V):
        t = torch.as_tensor(out[v][0], device="cuda")
        assert t.dtype == torch.uint8, f"v={v}: dtype={t.dtype}"
        assert t.ndim == 3, f"v={v}: ndim={t.ndim}"
        assert t.shape[-1] == 3, f"v={v}: shape={tuple(t.shape)}"
        assert t.device.type == "cuda"


def test_rgb_single_video_multi_frame():
    """V=1, F=4 is supported."""
    files = _sample_files()
    single = [files[0]]
    frame_ids_2d = [_SAME_GOP_FRAMES]

    gop_dec = _make_gop_dec()
    numpy_datas = _build_numpy_datas(gop_dec, single, frame_ids_2d)

    dec = _make_async_dec(V=1, F=len(_SAME_GOP_FRAMES))
    dec.DecodeFromGOPListRGB(numpy_datas, single, frame_ids_2d, False)
    out = dec.DecodeFromGOPListRGBGetBuffer(single, frame_ids_2d, False)
    assert len(out) == 1 and len(out[0]) == len(_SAME_GOP_FRAMES)


def test_rgb_single_frame_per_video():
    """F=1 is supported (one frame per video)."""
    files = _sample_files()
    V = len(files)
    frame_ids_2d = [[0]] * V

    gop_dec = _make_gop_dec()
    numpy_datas = _build_numpy_datas(gop_dec, files, frame_ids_2d)

    dec = _make_async_dec(V=V, F=1)
    dec.DecodeFromGOPListRGB(numpy_datas, files, frame_ids_2d, False)
    out = dec.DecodeFromGOPListRGBGetBuffer(files, frame_ids_2d, False)
    assert len(out) == V
    for v in range(V):
        assert len(out[v]) == 1


@pytest.mark.parametrize("as_bgr", [False, True])
def test_rgb_as_bgr_flag(as_bgr):
    """as_bgr=True and as_bgr=False both complete without error."""
    files = _sample_files()
    V = len(files)
    frame_ids_2d = [[0]] * V

    gop_dec = _make_gop_dec()
    numpy_datas = _build_numpy_datas(gop_dec, files, frame_ids_2d)

    dec = _make_async_dec(V=V, F=1)
    dec.DecodeFromGOPListRGB(numpy_datas, files, frame_ids_2d, as_bgr)
    out = dec.DecodeFromGOPListRGBGetBuffer(files, frame_ids_2d, as_bgr)
    assert out[0][0] is not None


# ===========================================================================
# Section E — functional YUV decode (shape, views, format)
# ===========================================================================


def test_yuv_output_shape():
    """DecodeFromGOPListGetBuffer returns List[List[DecodedFrameExt]] [V][F]."""
    files = _sample_files()
    V = len(files)
    F = len(_SAME_GOP_FRAMES)
    frame_ids_2d = [_SAME_GOP_FRAMES] * V

    gop_dec = _make_gop_dec()
    numpy_datas = _build_numpy_datas(gop_dec, files, frame_ids_2d)

    dec = _make_async_dec(V=V, F=F)
    dec.DecodeFromGOPList(numpy_datas, files, frame_ids_2d)
    out = dec.DecodeFromGOPListGetBuffer(files, frame_ids_2d)

    assert len(out) == V
    for v in range(V):
        assert len(out[v]) == F


def test_yuv_frame_has_views():
    """Each DecodedFrameExt has the correct number of CAIMemoryView plane views."""
    files = _sample_files()
    V = len(files)
    frame_ids_2d = [[0]] * V

    gop_dec = _make_gop_dec()
    numpy_datas = _build_numpy_datas(gop_dec, files, frame_ids_2d)

    dec = _make_async_dec(V=V, F=1)
    dec.DecodeFromGOPList(numpy_datas, files, frame_ids_2d)
    out = dec.DecodeFromGOPListGetBuffer(files, frame_ids_2d)

    # Raw Pixel_Format enum values (from PyCAIMemoryView.hpp):
    #   NV12=3 → 2 planes (Y, UV interleaved)
    #   YUV444=4 → 3 planes (Y, U, V separate)
    #   P016=5 → 2 planes (Y, UV interleaved 16-bit)
    #   YUV444_16Bit=6 → 3 planes (Y, U, V separate 16-bit)
    EXPECTED_VIEWS = {3: 2, 4: 3, 5: 2, 6: 3}

    for v in range(V):
        frame = out[v][0]
        views = frame.cuda()
        fmt = frame.format
        expected = EXPECTED_VIEWS.get(fmt, 2)
        assert (
            len(views) == expected
        ), f"v={v}: format={fmt}, expected {expected} plane views, got {len(views)}"
        # Y-plane shape: (H, W, 1)
        assert len(views[0].shape) == 3
        assert views[0].shape[2] == 1


def test_yuv_frame_format_set():
    """DecodedFrameExt.format is not UNDEFINED."""
    files = _sample_files()
    V = len(files)
    frame_ids_2d = [[0]] * V

    gop_dec = _make_gop_dec()
    numpy_datas = _build_numpy_datas(gop_dec, files, frame_ids_2d)

    dec = _make_async_dec(V=V, F=1)
    dec.DecodeFromGOPList(numpy_datas, files, frame_ids_2d)
    out = dec.DecodeFromGOPListGetBuffer(files, frame_ids_2d)

    for v in range(V):
        assert (
            out[v][0].format != nvc.Pixel_Format_UNDEFINED if hasattr(nvc, "Pixel_Format_UNDEFINED") else True
        )


# ===========================================================================
# Section F — precision: 2D RGB output must bit-match 1D GOP reference
# ===========================================================================


@pytest.mark.parametrize("as_bgr", [False, True])
def test_precision_rgb_matches_1d_gop_reference(as_bgr):
    """2D output pixels are bit-identical to the 1D DecodeFromGOPListRGB reference.

    Both use the same internal gop decode path so atol=0 / rtol=0 must hold.
    """
    files = _sample_files()
    V = len(files)
    frame_ids_2d = [_SAME_GOP_FRAMES] * V
    F = len(_SAME_GOP_FRAMES)

    gop_dec = _make_gop_dec()
    numpy_datas = _build_numpy_datas(gop_dec, files, frame_ids_2d)

    # Ground-truth: 1D GOP reference per (v, f)
    ref = [[None] * F for _ in range(V)]
    for v in range(V):
        for fi, fid in enumerate(frame_ids_2d[v]):
            # Use the serialized GOP bundle that covers this frame
            gop_data, first_frame_ids, _ = gop_dec.GetGOPList([files[v]], [fid])[0]
            ref[v][fi] = _ref_rgb(gop_dec, files[v], gop_data, fid, as_bgr)

    # Under test: 2D async GOP decode
    dec = _make_async_dec(V=V, F=F)
    dec.DecodeFromGOPListRGB(numpy_datas, files, frame_ids_2d, as_bgr)
    out = dec.DecodeFromGOPListRGBGetBuffer(files, frame_ids_2d, as_bgr)

    for v in range(V):
        for fi in range(F):
            actual = torch.as_tensor(out[v][fi], device="cuda")
            torch.testing.assert_close(
                actual,
                ref[v][fi],
                atol=0,
                rtol=0,
                msg=lambda m, vv=v, ff=fi: f"pixel mismatch v={vv} f={ff} as_bgr={as_bgr}: {m}",
            )


def test_precision_rgb_different_frames_per_video():
    """Different frame ids per video; result still matches per-video 1D reference."""
    files = _sample_files()
    V = len(files)
    F = 3
    # Stagger frames: video v gets frames [v*2, v*2+1, v*2+2]
    frame_ids_2d = [[v * 2 + i for i in range(F)] for v in range(V)]

    gop_dec = _make_gop_dec()
    numpy_datas = _build_numpy_datas(gop_dec, files, frame_ids_2d)

    ref = [[None] * F for _ in range(V)]
    for v in range(V):
        for fi, fid in enumerate(frame_ids_2d[v]):
            gop_data, _, _ = gop_dec.GetGOPList([files[v]], [fid])[0]
            ref[v][fi] = _ref_rgb(gop_dec, files[v], gop_data, fid, as_bgr=False)

    dec = _make_async_dec(V=V, F=F)
    dec.DecodeFromGOPListRGB(numpy_datas, files, frame_ids_2d, False)
    out = dec.DecodeFromGOPListRGBGetBuffer(files, frame_ids_2d, False)

    for v in range(V):
        for fi in range(F):
            actual = torch.as_tensor(out[v][fi], device="cuda")
            torch.testing.assert_close(actual, ref[v][fi], atol=0, rtol=0)


# ===========================================================================
# Section G — frame order preservation (unsorted frame_ids)
# ===========================================================================


def test_output_order_matches_unsorted_input():
    """out[v][f] corresponds to frame_ids_2d[v][f] even when IDs are unsorted.

    Submit frames in descending order.  The aggregated output must match the
    per-frame 1D reference at the SAME position (not the sorted position).
    """
    files = _sample_files()
    V = len(files)
    # Descending order — decoder must sort internally but return in original order
    unsorted_ids = [3, 2, 1, 0]
    F = len(unsorted_ids)
    frame_ids_2d = [unsorted_ids] * V

    gop_dec = _make_gop_dec()
    numpy_datas = _build_numpy_datas(gop_dec, files, frame_ids_2d)

    # Reference: 1D decode at each position in ORIGINAL (descending) order
    ref = [[None] * F for _ in range(V)]
    for v in range(V):
        for fi, fid in enumerate(frame_ids_2d[v]):
            gop_data, _, _ = gop_dec.GetGOPList([files[v]], [fid])[0]
            ref[v][fi] = _ref_rgb(gop_dec, files[v], gop_data, fid, as_bgr=False)

    dec = _make_async_dec(V=V, F=F)
    dec.DecodeFromGOPListRGB(numpy_datas, files, frame_ids_2d, False)
    out = dec.DecodeFromGOPListRGBGetBuffer(files, frame_ids_2d, False)

    for v in range(V):
        for fi in range(F):
            fid = frame_ids_2d[v][fi]
            actual = torch.as_tensor(out[v][fi], device="cuda")
            torch.testing.assert_close(
                actual,
                ref[v][fi],
                atol=0,
                rtol=0,
                msg=lambda m, vv=v, ff=fi, fid_=fid: (
                    f"order mismatch at v={vv} fi={ff} (frame_id={fid_}): {m}"
                ),
            )


# ===========================================================================
# Section H — multi-GOP bundles (frames spanning different GOPs)
# ===========================================================================


def test_multi_gop_rgb_decode_shape():
    """Frames spanning two GOPs (e.g. [0, 60]) decode correctly.

    The test only checks shape/device — pixel correctness is covered by the
    precision section above which also feeds multi-frame inputs.
    """
    files = _sample_files()
    V = len(files)
    frame_ids_2d = [_MULTI_GOP_FRAMES] * V
    F = len(_MULTI_GOP_FRAMES)

    gop_dec = _make_gop_dec()
    # _build_numpy_datas deduplicates by GOP: frames 0 and 60 are in different
    # GOPs, so numpy_datas[v] will have 2 entries.
    numpy_datas = _build_numpy_datas(gop_dec, files, frame_ids_2d)
    for v in range(V):
        assert len(numpy_datas[v]) >= 1, f"expected at least 1 GOP bundle for video {v}"

    dec = _make_async_dec(V=V, F=F)
    dec.DecodeFromGOPListRGB(numpy_datas, files, frame_ids_2d, False)
    out = dec.DecodeFromGOPListRGBGetBuffer(files, frame_ids_2d, False)

    assert len(out) == V
    for v in range(V):
        assert len(out[v]) == F
        for fi in range(F):
            t = torch.as_tensor(out[v][fi], device="cuda")
            assert t.shape[-1] == 3


def test_multi_gop_rgb_precision():
    """Multi-GOP decode pixel values match 1D reference."""
    files = _sample_files()
    V = len(files)
    frame_ids_2d = [_MULTI_GOP_FRAMES] * V
    F = len(_MULTI_GOP_FRAMES)

    gop_dec = _make_gop_dec()
    numpy_datas = _build_numpy_datas(gop_dec, files, frame_ids_2d)

    ref = [[None] * F for _ in range(V)]
    for v in range(V):
        for fi, fid in enumerate(frame_ids_2d[v]):
            gop_data, _, _ = gop_dec.GetGOPList([files[v]], [fid])[0]
            ref[v][fi] = _ref_rgb(gop_dec, files[v], gop_data, fid, as_bgr=False)

    dec = _make_async_dec(V=V, F=F)
    dec.DecodeFromGOPListRGB(numpy_datas, files, frame_ids_2d, False)
    out = dec.DecodeFromGOPListRGBGetBuffer(files, frame_ids_2d, False)

    for v in range(V):
        for fi in range(F):
            actual = torch.as_tensor(out[v][fi], device="cuda")
            torch.testing.assert_close(actual, ref[v][fi], atol=0, rtol=0)


# ===========================================================================
# Section I — async behavior: error propagation, mismatch, prefetch loop
# ===========================================================================


def test_getbuffer_rgb_without_decode_raises():
    """GetBuffer with no pending task raises RuntimeError."""
    files = _sample_files()
    dec = _make_async_dec(V=len(files))
    with pytest.raises(RuntimeError):
        dec.DecodeFromGOPListRGBGetBuffer(files, [[0]] * len(files), False)


def test_getbuffer_yuv_without_decode_raises():
    """YUV GetBuffer with no pending task raises RuntimeError."""
    files = _sample_files()
    dec = _make_async_dec(V=len(files))
    with pytest.raises(RuntimeError):
        dec.DecodeFromGOPListGetBuffer(files, [[0]] * len(files))


def test_rgb_getbuffer_after_yuv_decode_raises():
    """Calling RGB GetBuffer after a YUV Decode raises a type-mismatch error."""
    files = _sample_files()
    V = len(files)
    frame_ids_2d = [[0]] * V
    gop_dec = _make_gop_dec()
    numpy_datas = _build_numpy_datas(gop_dec, files, frame_ids_2d)

    dec = _make_async_dec(V=V, F=1)
    dec.DecodeFromGOPList(numpy_datas, files, frame_ids_2d)
    with pytest.raises(RuntimeError):
        dec.DecodeFromGOPListRGBGetBuffer(files, frame_ids_2d, False)


def test_yuv_getbuffer_after_rgb_decode_raises():
    """Calling YUV GetBuffer after an RGB Decode raises a type-mismatch error."""
    files = _sample_files()
    V = len(files)
    frame_ids_2d = [[0]] * V
    gop_dec = _make_gop_dec()
    numpy_datas = _build_numpy_datas(gop_dec, files, frame_ids_2d)

    dec = _make_async_dec(V=V, F=1)
    dec.DecodeFromGOPListRGB(numpy_datas, files, frame_ids_2d, False)
    with pytest.raises(RuntimeError):
        dec.DecodeFromGOPListGetBuffer(files, frame_ids_2d)


def test_getbuffer_request_mismatch_files_raises():
    """GetBuffer with different filepaths than Decode raises RuntimeError."""
    files = _sample_files()
    V = len(files)
    if V < 2:
        pytest.skip("need at least 2 sample videos")
    frame_ids_2d = [[0]] * V
    gop_dec = _make_gop_dec()
    numpy_datas = _build_numpy_datas(gop_dec, files, frame_ids_2d)

    dec = _make_async_dec(V=V, F=1)
    dec.DecodeFromGOPListRGB(numpy_datas, files, frame_ids_2d, False)

    swapped = list(files)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(RuntimeError):
        dec.DecodeFromGOPListRGBGetBuffer(swapped, frame_ids_2d, False)


def test_getbuffer_request_mismatch_frame_ids_raises():
    """GetBuffer with different frame_ids than Decode raises RuntimeError."""
    files = _sample_files()
    V = len(files)
    frame_ids_2d = [[0]] * V
    gop_dec = _make_gop_dec()
    numpy_datas = _build_numpy_datas(gop_dec, files, frame_ids_2d)

    dec = _make_async_dec(V=V, F=1)
    dec.DecodeFromGOPListRGB(numpy_datas, files, frame_ids_2d, False)
    with pytest.raises(RuntimeError):
        dec.DecodeFromGOPListRGBGetBuffer(files, [[7]] * V, False)


def test_resubmit_only_latest_result_retrievable():
    """Two consecutive Decode calls: only the second result is retrievable.

    After re-submitting, GetBuffer with the OLD params must fail (result consumed);
    with the NEW params must succeed.
    """
    files = _sample_files()
    V = len(files)
    frame_ids_a = [[0]] * V
    frame_ids_b = [[7]] * V
    gop_dec = _make_gop_dec()
    nd_a = _build_numpy_datas(gop_dec, files, frame_ids_a)
    nd_b = _build_numpy_datas(gop_dec, files, frame_ids_b)

    # Variant 1: Get with new params — should succeed.
    dec = _make_async_dec(V=V, F=1)
    dec.DecodeFromGOPListRGB(nd_a, files, frame_ids_a, False)
    dec.DecodeFromGOPListRGB(nd_b, files, frame_ids_b, False)
    out = dec.DecodeFromGOPListRGBGetBuffer(files, frame_ids_b, False)
    assert len(out) == V and len(out[0]) == 1

    # Variant 2: Get with old params — should fail (result was overwritten).
    dec2 = _make_async_dec(V=V, F=1)
    dec2.DecodeFromGOPListRGB(nd_a, files, frame_ids_a, False)
    dec2.DecodeFromGOPListRGB(nd_b, files, frame_ids_b, False)
    with pytest.raises(RuntimeError):
        dec2.DecodeFromGOPListRGBGetBuffer(files, frame_ids_a, False)


def test_invalid_frame_id_propagates_exception():
    """Out-of-range frame_id (beyond video length) is rethrown at GetBuffer.

    The bundle for frame 0 does not cover frame 999999, so the worker throws
    "no serialized GOP bundle covers frame 999999".  This tests that async
    worker errors are stored and re-raised at GetBuffer time.
    """
    files = _sample_files()
    V = len(files)
    # Frame id 999999 is almost certainly out of range for any test video.
    frame_ids_2d = [[999999]] * V
    gop_dec = _make_gop_dec()
    # Bundle obtained for a valid frame; the invalid frame_id causes a decode error.
    gop_data = _get_numpy_data(gop_dec, files[0], 0)
    numpy_datas = [[gop_data]] * V

    dec = _make_async_dec(V=V, F=1)
    dec.DecodeFromGOPListRGB(numpy_datas, files, frame_ids_2d, False)
    with pytest.raises(RuntimeError):
        dec.DecodeFromGOPListRGBGetBuffer(files, frame_ids_2d, False)


def test_wrong_gop_bundle_propagates_async_error():
    """Valid frame_id but bundle from the wrong GOP is rethrown at GetBuffer.

    This is distinct from an out-of-range frame_id: the frame exists in the
    video, but the caller supplied the GOP bundle for a *different* GOP — the
    kind of bug that occurs when GetGOPList and frame_ids get out of sync in a
    dataloader pipeline.

    Concretely: we request frame 0 (in GOP-0) but supply only the bundle that
    covers the *second* GOP.  The worker's bundle-to-frame matching loop finds
    no bundle that covers frame 0 and throws "no serialized GOP bundle covers
    frame 0".  That error must be re-raised at GetBuffer, not silently swallowed.
    """
    files = _sample_files()
    V = len(files)
    gop_dec = _make_gop_dec()

    # Probe GOP-0 length to find the start of GOP-1 (guaranteed different GOP).
    _, first_frame_ids_probe, gop_lens_probe = gop_dec.GetGOPList([files[0]], [0])[0]
    gop1_start = int(first_frame_ids_probe[0]) + int(gop_lens_probe[0])

    # Bundle covers GOP-1 (frames starting at gop1_start).
    wrong_bundle = _get_numpy_data(gop_dec, files[0], gop1_start)
    numpy_datas = [[wrong_bundle]] * V  # wrong GOP for every video

    # But frame_ids request frame 0, which is in GOP-0, not GOP-1.
    frame_ids_2d = [[0]] * V

    dec = _make_async_dec(V=V, F=1)
    dec.DecodeFromGOPListRGB(numpy_datas, files, frame_ids_2d, False)
    with pytest.raises(RuntimeError):
        dec.DecodeFromGOPListRGBGetBuffer(files, frame_ids_2d, False)


def test_partial_frame_coverage_propagates_async_error():
    """F=2: first frame covered by the supplied bundle, second frame not covered.

    Partial coverage is NOT valid.  The worker iterates all F frames for each
    video; when it reaches the second frame and finds no bundle covering it, it
    throws "no GOP bundle covers frame", aborting the entire batch.  The error
    must surface at GetBuffer time.

    This differs from the wrong-GOP-bundle test (F=1) in that here the first
    frame would decode successfully if the worker weren't aborting mid-loop —
    it tests that a partial-match within a multi-frame request is still rejected.
    """
    files = _sample_files()
    V = len(files)
    gop_dec = _make_gop_dec()

    # Probe GOP-0 to find the start of GOP-1.
    _, first_frame_ids_probe, gop_lens_probe = gop_dec.GetGOPList([files[0]], [0])[0]
    gop1_start = int(first_frame_ids_probe[0]) + int(gop_lens_probe[0])

    # Provide only the GOP-0 bundle.
    gop0_bundle = _get_numpy_data(gop_dec, files[0], 0)
    numpy_datas = [[gop0_bundle]] * V

    # Request two frames: frame 0 (covered by GOP-0 bundle) and gop1_start (NOT covered).
    frame_ids_2d = [[0, gop1_start]] * V

    dec = _make_async_dec(V=V, F=2)
    dec.DecodeFromGOPListRGB(numpy_datas, files, frame_ids_2d, False)
    with pytest.raises(RuntimeError):
        dec.DecodeFromGOPListRGBGetBuffer(files, frame_ids_2d, False)


def test_single_video_error_kills_entire_batch():
    """One video with a wrong GOP bundle causes ALL V videos' results to be discarded.

    V=3 videos: video[0] and video[2] have correct bundles for frame 0;
    video[1] has the bundle for GOP-1 but requests frame 0 (which is in GOP-0).

    The worker iterates Phase 1 for each video in order.  When it reaches
    video[1] and finds no bundle covering frame 0, it throws immediately —
    video[2] is never even attempted.  The entire batch (all three results) is
    cleared.  This test asserts that the error still surfaces at GetBuffer even
    though two of the three videos would have decoded successfully on their own.
    """
    files = _sample_files()
    if len(files) < 3:
        pytest.skip("need at least 3 sample videos")
    gop_dec = _make_gop_dec()

    _, first_frame_ids_probe, gop_lens_probe = gop_dec.GetGOPList([files[0]], [0])[0]
    gop1_start = int(first_frame_ids_probe[0]) + int(gop_lens_probe[0])

    correct_bundle = _get_numpy_data(gop_dec, files[0], 0)  # covers frame 0
    wrong_bundle = _get_numpy_data(gop_dec, files[1], gop1_start)  # covers GOP-1, NOT frame 0

    # video[0] and video[2] get the correct bundle; video[1] gets the wrong one.
    numpy_datas = [
        [correct_bundle],  # video 0 — would succeed alone
        [wrong_bundle],  # video 1 — wrong GOP → kills the batch
        [correct_bundle],  # video 2 — would succeed alone but never runs
    ]
    three_files = [files[0], files[1], files[2]]
    frame_ids_2d = [[0]] * 3

    dec = _make_async_dec(V=3, F=1)
    dec.DecodeFromGOPListRGB(numpy_datas, three_files, frame_ids_2d, False)
    with pytest.raises(RuntimeError):
        dec.DecodeFromGOPListRGBGetBuffer(three_files, frame_ids_2d, False)


def test_malformed_bundle_propagates_async_error():
    """A non-empty bundle with invalid (garbage) bytes is rethrown at GetBuffer.

    This is distinct from the zero-byte bundle test: here numpy_datas[v] is a
    list with one element (so the "not empty" check passes), but the bytes are
    random garbage that parseSerializedPacketData cannot interpret.  The result
    is either a parse exception or an empty gop_ranges list (causing the
    "no bundle covers frame" error).  Either way the error must reach GetBuffer.
    """
    files = _sample_files()
    V = len(files)
    garbage_bundle = np.ones(128, dtype=np.uint8)  # non-empty, non-zero, invalid format
    numpy_datas = [[garbage_bundle]] * V

    dec = _make_async_dec(V=V, F=1)
    dec.DecodeFromGOPListRGB(numpy_datas, files, [[0]] * V, False)
    with pytest.raises(RuntimeError):
        dec.DecodeFromGOPListRGBGetBuffer(files, [[0]] * V, False)


def test_prefetch_loop_pattern():
    """Classic prefetch: submit B immediately after collecting A so B's decode overlaps
    with CPU-side processing of A.

    Correct pattern:
      Submit A → GetBuffer A → clone A to CPU → Submit B → [process A] → GetBuffer B

    A frames are cloned to CPU before Submit B because the internal aggregator pool
    is reused by B's background worker (lifetime contract: frames are invalidated on
    the next Decode call).
    """
    files = _sample_files()
    V = len(files)
    F = 2
    frame_ids_a = [[0, 1]] * V
    frame_ids_b = [[2, 3]] * V

    gop_dec = _make_gop_dec()
    nd_a = _build_numpy_datas(gop_dec, files, frame_ids_a)
    nd_b = _build_numpy_datas(gop_dec, files, frame_ids_b)

    dec = _make_async_dec(V=V, F=F)

    # Batch A: submit and collect.
    dec.DecodeFromGOPListRGB(nd_a, files, frame_ids_a, False)
    out_a = dec.DecodeFromGOPListRGBGetBuffer(files, frame_ids_a, False)

    # Clone A to CPU before submitting B — pool memory will be reused by B's worker.
    cpu_a = [[torch.as_tensor(out_a[v][fi], device="cuda").cpu() for fi in range(F)] for v in range(V)]

    # Submit B — background decode starts immediately.
    dec.DecodeFromGOPListRGB(nd_b, files, frame_ids_b, False)

    # Process A on CPU while B decodes in the background (true overlap).
    for v in range(V):
        for fi in range(F):
            assert cpu_a[v][fi].shape[-1] == 3
            assert cpu_a[v][fi].dtype == torch.uint8

    # Collect B.
    out_b = dec.DecodeFromGOPListRGBGetBuffer(files, frame_ids_b, False)

    assert len(out_b) == V
    for v in range(V):
        assert len(out_b[v]) == F
        for fi in range(F):
            t = torch.as_tensor(out_b[v][fi], device="cuda")
            assert t.shape[-1] == 3 and t.dtype == torch.uint8


# ===========================================================================
# Section J — boundary: maxfiles and max_frames_per_decode_call step-up
# ===========================================================================


def test_maxfiles_step_up_within_limit_succeeds():
    """maxfiles=5: V=2 and V=4 both decode successfully with correct output shape.

    Uses the same video file repeated V times so the test is independent of
    how many distinct sample clips are available.
    """
    files = _sample_files()
    base_file = files[0]
    gop_dec = _make_gop_dec()
    dec = nvc.CreateBatchAsyncGopDecoder(maxfiles=5, max_frames_per_decode_call=4, iGpu=0)

    for V in (2, 4):
        vfiles = [base_file] * V
        frame_ids_2d = [[0]] * V
        numpy_datas = _build_numpy_datas(gop_dec, vfiles, frame_ids_2d)
        dec.DecodeFromGOPListRGB(numpy_datas, vfiles, frame_ids_2d, False)
        out = dec.DecodeFromGOPListRGBGetBuffer(vfiles, frame_ids_2d, False)
        assert len(out) == V, f"V={V}: expected {V} output videos, got {len(out)}"
        for v in range(V):
            assert len(out[v]) == 1, f"V={V} v={v}: expected 1 frame per video"


def test_maxfiles_step_up_exceeds_limit_raises():
    """maxfiles=5: V=6 is rejected synchronously at validate_decode_input time.

    The error must be raised from DecodeFromGOPListRGB itself (not from
    GetBuffer), because validate_decode_input runs before submit_work.
    """
    files = _sample_files()
    base_file = files[0]
    gop_dec = _make_gop_dec()
    dec = nvc.CreateBatchAsyncGopDecoder(maxfiles=5, max_frames_per_decode_call=4, iGpu=0)

    V = 6
    vfiles = [base_file] * V
    frame_ids_2d = [[0]] * V
    numpy_datas = _build_numpy_datas(gop_dec, vfiles, frame_ids_2d)

    with pytest.raises(RuntimeError, match="exceeds"):
        dec.DecodeFromGOPListRGB(numpy_datas, vfiles, frame_ids_2d, False)


def test_maxfiles_step_up_full_sequence():
    """maxfiles=5: V=2 → V=4 (both succeed) → V=6 (raises) on the same decoder.

    Verifies that a successful decode after a previous one doesn't corrupt
    decoder state, and that the subsequent over-limit call still raises cleanly.
    """
    files = _sample_files()
    base_file = files[0]
    gop_dec = _make_gop_dec()
    dec = nvc.CreateBatchAsyncGopDecoder(maxfiles=5, max_frames_per_decode_call=4, iGpu=0)

    for V in (2, 4):
        vfiles = [base_file] * V
        frame_ids_2d = [[0]] * V
        numpy_datas = _build_numpy_datas(gop_dec, vfiles, frame_ids_2d)
        dec.DecodeFromGOPListRGB(numpy_datas, vfiles, frame_ids_2d, False)
        out = dec.DecodeFromGOPListRGBGetBuffer(vfiles, frame_ids_2d, False)
        assert len(out) == V

    V = 6
    vfiles = [base_file] * V
    frame_ids_2d = [[0]] * V
    numpy_datas = _build_numpy_datas(gop_dec, vfiles, frame_ids_2d)
    with pytest.raises(RuntimeError, match="exceeds"):
        dec.DecodeFromGOPListRGB(numpy_datas, vfiles, frame_ids_2d, False)

    # Decoder must still be usable after a rejected call.
    V_ok = 3
    vfiles = [base_file] * V_ok
    frame_ids_2d = [[0]] * V_ok
    numpy_datas = _build_numpy_datas(gop_dec, vfiles, frame_ids_2d)
    dec.DecodeFromGOPListRGB(numpy_datas, vfiles, frame_ids_2d, False)
    out = dec.DecodeFromGOPListRGBGetBuffer(vfiles, frame_ids_2d, False)
    assert len(out) == V_ok


def test_max_frames_step_up_within_limit_succeeds():
    """max_frames_per_decode_call=5: F=2 and F=4 both decode with correct output shape.

    All frames are within the same GOP (frames 0–3) to avoid multi-GOP setup.
    """
    files = _sample_files()
    base_file = files[0]
    gop_dec = _make_gop_dec()
    dec = nvc.CreateBatchAsyncGopDecoder(maxfiles=2, max_frames_per_decode_call=5, iGpu=0)
    vfiles = [base_file, base_file]

    for F in (2, 4):
        frame_ids = list(range(F))
        frame_ids_2d = [frame_ids] * 2
        numpy_datas = _build_numpy_datas(gop_dec, vfiles, frame_ids_2d)
        dec.DecodeFromGOPListRGB(numpy_datas, vfiles, frame_ids_2d, False)
        out = dec.DecodeFromGOPListRGBGetBuffer(vfiles, frame_ids_2d, False)
        assert len(out) == 2
        for v in range(2):
            assert len(out[v]) == F, f"F={F} v={v}: expected {F} frames, got {len(out[v])}"


def test_max_frames_step_up_exceeds_limit_raises():
    """max_frames_per_decode_call=5: F=6 is rejected synchronously at validate_decode_input.

    validate_decode_input checks the frame count before starting any GPU work,
    so the error comes from DecodeFromGOPListRGB, not from GetBuffer.
    """
    files = _sample_files()
    base_file = files[0]
    gop_dec = _make_gop_dec()
    dec = nvc.CreateBatchAsyncGopDecoder(maxfiles=2, max_frames_per_decode_call=5, iGpu=0)
    vfiles = [base_file, base_file]

    frame_ids = list(range(6))
    frame_ids_2d = [frame_ids] * 2
    numpy_datas = _build_numpy_datas(gop_dec, vfiles, frame_ids_2d)

    with pytest.raises(RuntimeError, match="exceeds"):
        dec.DecodeFromGOPListRGB(numpy_datas, vfiles, frame_ids_2d, False)


def test_max_frames_step_up_full_sequence():
    """max_frames_per_decode_call=5: F=2 → F=4 (both succeed) → F=6 (raises) on one decoder.

    Also checks the decoder remains usable after the rejected call.
    """
    files = _sample_files()
    base_file = files[0]
    gop_dec = _make_gop_dec()
    dec = nvc.CreateBatchAsyncGopDecoder(maxfiles=2, max_frames_per_decode_call=5, iGpu=0)
    vfiles = [base_file, base_file]

    for F in (2, 4):
        frame_ids = list(range(F))
        frame_ids_2d = [frame_ids] * 2
        numpy_datas = _build_numpy_datas(gop_dec, vfiles, frame_ids_2d)
        dec.DecodeFromGOPListRGB(numpy_datas, vfiles, frame_ids_2d, False)
        out = dec.DecodeFromGOPListRGBGetBuffer(vfiles, frame_ids_2d, False)
        assert len(out) == 2
        for v in range(2):
            assert len(out[v]) == F

    frame_ids = list(range(6))
    frame_ids_2d = [frame_ids] * 2
    numpy_datas = _build_numpy_datas(gop_dec, vfiles, frame_ids_2d)
    with pytest.raises(RuntimeError, match="exceeds"):
        dec.DecodeFromGOPListRGB(numpy_datas, vfiles, frame_ids_2d, False)

    # Decoder must still be usable.
    F_ok = 3
    frame_ids = list(range(F_ok))
    frame_ids_2d = [frame_ids] * 2
    numpy_datas = _build_numpy_datas(gop_dec, vfiles, frame_ids_2d)
    dec.DecodeFromGOPListRGB(numpy_datas, vfiles, frame_ids_2d, False)
    out = dec.DecodeFromGOPListRGBGetBuffer(vfiles, frame_ids_2d, False)
    assert len(out) == 2
    for v in range(2):
        assert len(out[v]) == F_ok


def test_jagged_frame_ids_varying_lengths_raises():
    """frame_ids_2d with unequal inner lengths is rejected synchronously.

    Scenario: V=3 videos with frame counts [10, 3, 7].  The transposed decode
    loop requires all V videos to share the same frame-slot index F, so jagged
    input is fundamentally unsupported.  The error fires from DecodeFromGOPListRGB
    (not GetBuffer) because validate_decode_input runs before submit_work.

    max_frames_per_decode_call is set to 10 (the largest inner length) so the
    frame-count limit is not the trigger — only the jagged shape check is.
    """
    files = _sample_files()
    base_file = files[0]
    gop_dec = _make_gop_dec()
    V = 3
    vfiles = [base_file] * V
    # numpy_datas outer size must match V; inner content is irrelevant because
    # validate_decode_input throws at the jagged check before inspecting bundles.
    gop_data = _get_numpy_data(gop_dec, base_file, 0)
    numpy_datas = [[gop_data]] * V

    frame_lengths = [10, 3, 7]
    frame_ids_2d = [list(range(n)) for n in frame_lengths]

    dec = nvc.CreateBatchAsyncGopDecoder(maxfiles=V, max_frames_per_decode_call=max(frame_lengths), iGpu=0)
    with pytest.raises(RuntimeError, match="jagged"):
        dec.DecodeFromGOPListRGB(numpy_datas, vfiles, frame_ids_2d, False)


def test_concurrent_decode_from_two_threads_no_deadlock():
    """Two threads call DecodeFromGOPListRGB concurrently on the same decoder.

    Both threads submit the same batch.  The second caller enters submit_work,
    acquires async_mutex_, sees has_pending_task_=True, releases the lock, joins
    the first worker, re-acquires the lock, clears the queue, and re-submits.
    This exercises the join-and-supersede path under real concurrency.

    Assertions:
    - Both threads terminate within the timeout (no deadlock).
    - GetBuffer with the common batch params succeeds after both submits complete.
    """
    files = _sample_files()
    V = len(files)
    frame_ids_2d = [[0]] * V
    gop_dec = _make_gop_dec()
    nd = _build_numpy_datas(gop_dec, files, frame_ids_2d)
    dec = _make_async_dec(V=V, F=1)

    submit_errors = {}
    barrier = threading.Barrier(2)

    def submit(tid):
        barrier.wait()  # both threads race into submit_work simultaneously
        try:
            dec.DecodeFromGOPListRGB(nd, files, frame_ids_2d, False)
        except Exception as e:
            submit_errors[tid] = e

    t1 = threading.Thread(target=submit, args=(1,), daemon=True)
    t2 = threading.Thread(target=submit, args=(2,), daemon=True)
    t1.start()
    t2.start()

    TIMEOUT = 15.0
    t1.join(TIMEOUT)
    t2.join(TIMEOUT)
    assert not t1.is_alive(), "Thread-1 deadlocked in DecodeFromGOPListRGB"
    assert not t2.is_alive(), "Thread-2 deadlocked in DecodeFromGOPListRGB"
    assert not submit_errors, f"unexpected exception during concurrent submit: {submit_errors}"

    # Both submitted the same batch — exactly one result is in the queue; GetBuffer must succeed.
    out = dec.DecodeFromGOPListRGBGetBuffer(files, frame_ids_2d, False)
    assert len(out) == V


def test_resubmit_while_getbuffer_blocking_no_deadlock():
    """Main thread resubmits a new decode while a worker thread is blocking in GetBuffer.

    Timeline:
      1. Main: DecodeFromGOPListRGB(batch_A) — worker A starts.
      2. Worker: starts GetBuffer(batch_A) — blocks in result_cv_.wait().
      3. Main: DecodeFromGOPListRGB(batch_B) — enters submit_work, joins worker A,
               clears queue, starts worker B.

    After step 3, worker A's result is either:
      (a) Already popped by Worker thread before the clear → Worker thread returns
          batch_A result successfully; main thread's subsequent GetBuffer(B) works.
      (b) Cleared before Worker thread pops → Worker thread wakes up, queue is empty,
          then waits for worker B's result; validate_request(B, frame_ids_A) fails →
          Worker thread gets RuntimeError; main thread's GetBuffer(B) may or may not
          find the result depending on whether Worker thread consumed it.

    In all cases: Worker thread must terminate (no deadlock).  The test only
    asserts on the no-deadlock property, not on the exact result routing.
    """
    files = _sample_files()
    V = len(files)
    frame_ids_a = [[0]] * V
    frame_ids_b = [[1]] * V
    gop_dec = _make_gop_dec()
    nd_a = _build_numpy_datas(gop_dec, files, frame_ids_a)
    nd_b = _build_numpy_datas(gop_dec, files, frame_ids_b)
    dec = _make_async_dec(V=V, F=1)

    worker_done = {}
    ready = threading.Event()

    def collect_a():
        ready.set()  # signal main thread we're about to call GetBuffer
        try:
            out = dec.DecodeFromGOPListRGBGetBuffer(files, frame_ids_a, False)
            worker_done['result'] = out
        except RuntimeError as e:
            worker_done['error'] = e

    dec.DecodeFromGOPListRGB(nd_a, files, frame_ids_a, False)
    t = threading.Thread(target=collect_a, daemon=True)
    t.start()
    ready.wait()  # give the worker thread a moment to enter GetBuffer

    # Resubmit: joins worker A (which may still be running), clears queue, starts worker B.
    dec.DecodeFromGOPListRGB(nd_b, files, frame_ids_b, False)

    TIMEOUT = 15.0
    t.join(TIMEOUT)
    assert not t.is_alive(), "Worker thread deadlocked during resubmit"
    # Worker thread must have produced exactly one outcome.
    assert len(worker_done) == 1, f"expected 1 outcome, got {worker_done}"


def test_concurrent_getbuffer_no_deadlock():
    """Two threads calling GetBuffer on the same pending result must not deadlock.

    The check, wait, and pop are atomic under async_mutex_, so the second thread
    wakes up, finds the queue empty, and raises RuntimeError instead of hanging.
    """
    files = _sample_files()
    files = files[:1]
    frame_ids_2d = [[0]]

    gop_dec = _make_gop_dec()
    nd = _build_numpy_datas(gop_dec, files, frame_ids_2d)
    dec = _make_async_dec(V=1, F=1)

    results = {}
    errors = {}

    # Barrier synchronises submit + both getters so they race on the same result slot.
    barrier = threading.Barrier(3)

    def getter(tid):
        barrier.wait()
        try:
            out = dec.DecodeFromGOPListRGBGetBuffer(files, frame_ids_2d, False)
            results[tid] = out
        except RuntimeError as e:
            errors[tid] = e

    t1 = threading.Thread(target=getter, args=(1,), daemon=True)
    t2 = threading.Thread(target=getter, args=(2,), daemon=True)
    t1.start()
    t2.start()

    dec.DecodeFromGOPListRGB(nd, files, frame_ids_2d, False)
    barrier.wait()  # release both threads while the worker is still (or just finished) running

    TIMEOUT = 10.0
    t1.join(TIMEOUT)
    t2.join(TIMEOUT)

    assert not t1.is_alive(), "Thread-1 deadlocked in GetBuffer"
    assert not t2.is_alive(), "Thread-2 deadlocked in GetBuffer"

    # Exactly one thread must have succeeded; the other must have raised RuntimeError.
    assert len(results) == 1, f"expected 1 success, got {len(results)}"
    assert len(errors) == 1, f"expected 1 error, got {len(errors)}"
    assert (
        "concurrent" in str(list(errors.values())[0]).lower()
        or "already consumed" in str(list(errors.values())[0]).lower()
    )


# ---------------------------------------------------------------------------
# Section K — release_device_memory / release_decoder with pending task
# ---------------------------------------------------------------------------


def test_release_device_memory_while_pending_no_crash():
    """release_device_memory() joins the pending worker before freeing pools.

    Verified behaviors:
    - No crash / hang (data race would manifest as SIGSEGV or deadlock).
    - GetBuffer raises RuntimeError after release (result was cleared).
    - A subsequent decode succeeds (decoder is still valid; pools are re-allocable).
    """
    files = _sample_files()
    files = files[:1]
    frame_ids_2d = [[0]]

    gop_dec = _make_gop_dec()
    nd = _build_numpy_datas(gop_dec, files, frame_ids_2d)
    dec = _make_async_dec(V=1, F=1)

    # Submit — worker starts running immediately.
    dec.DecodeFromGOPListRGB(nd, files, frame_ids_2d, False)

    # Release while the worker may still be running. Must not crash.
    dec.release_device_memory()

    # Result queue was cleared; GetBuffer must raise.
    with pytest.raises(RuntimeError):
        dec.DecodeFromGOPListRGBGetBuffer(files, frame_ids_2d, False)

    # Decoder is still operational: a fresh decode must succeed.
    nd2 = _build_numpy_datas(gop_dec, files, frame_ids_2d)
    dec.DecodeFromGOPListRGB(nd2, files, frame_ids_2d, False)
    out = dec.DecodeFromGOPListRGBGetBuffer(files, frame_ids_2d, False)
    assert len(out) == 1
    assert len(out[0]) == 1


def test_release_decoder_while_pending_no_crash():
    """release_decoder() joins the pending worker before releasing gop_dec_.

    Verified behaviors:
    - No crash / hang.
    - GetBuffer still returns the completed result (result_queue_ is NOT cleared;
      the result lives in rgb_agg_pools_, not in gop_dec_).
    - A subsequent decode succeeds (decoder is lazily re-created).
    """
    files = _sample_files()
    files = files[:1]
    frame_ids_2d = [[0]]

    gop_dec = _make_gop_dec()
    nd = _build_numpy_datas(gop_dec, files, frame_ids_2d)
    dec = _make_async_dec(V=1, F=1)

    # Submit — worker starts running immediately.
    dec.DecodeFromGOPListRGB(nd, files, frame_ids_2d, False)

    # Release while the worker may still be running. Must not crash.
    dec.release_decoder()

    # The result was placed in rgb_agg_pools_ (still valid); GetBuffer must succeed.
    out = dec.DecodeFromGOPListRGBGetBuffer(files, frame_ids_2d, False)
    assert len(out) == 1
    assert len(out[0]) == 1

    # A subsequent decode must succeed (decoder is re-created lazily).
    nd2 = _build_numpy_datas(gop_dec, files, frame_ids_2d)
    dec.DecodeFromGOPListRGB(nd2, files, frame_ids_2d, False)
    out2 = dec.DecodeFromGOPListRGBGetBuffer(files, frame_ids_2d, False)
    assert len(out2) == 1
    assert len(out2[0]) == 1
