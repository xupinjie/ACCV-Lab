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

"""Tests for ``GetGOPList`` partition behavior on open-GOP HEVC video.

Open-GOP HEVC associates each IRAP (CRA) with zero or more leading
RASL/RADL pictures whose display index is less than the IRAP's. The
invariant verified here is that ``(first_frame_id, gop_len)`` defines a
non-overlapping partition of the display-index space: every display frame
belongs to exactly one GOP, and a leading picture is assigned to the
previous GOP in display order, not to the GOP introduced by its associated
IRAP. The same invariant is what lets ``SharedGopStore.lookup``,
``CachedGopDecoder._is_cache_hit``, and ``DecodeFromGOP``'s range check
resolve a frame to a unique GOP.

Fixture: a 100-frame, 256x256, 5-GOP open-GOP HEVC clip
(``moving_shape_open_gop_h265.mp4``) with CRA pictures at display
indices ``[0, 20, 40, 60, 80]`` and RASL leading pictures straddling
several CRAs. Stored under ``data/open_gop_variant/`` rather than
``data/sample_clip/`` so the general random-clip sweep in
``utils.select_random_clip`` does not pick it up (the 100-frame length
is below the ``frame_max=200`` those tests assume).
"""

import os

import pytest
import torch

import accvlab.on_demand_video_decoder as nvc
from accvlab.on_demand_video_decoder import SharedGopStore

import utils

OPEN_GOP_SAMPLE = os.path.join(utils.get_data_dir(), "open_gop_variant", "moving_shape_open_gop_h265.mp4")

# Ground truth — the structure of the committed fixture. If
# ``moving_shape_open_gop_h265.mp4`` is ever replaced, this table and
# ``TOTAL_FRAMES`` must be updated to match the new clip.
EXPECTED_PARTITION = [
    (0, 20),  # IDR
    (20, 40),  # CRA with 2 leading pictures (display 18, 19)
    (40, 60),  # CRA with 1 leading picture  (display 39)
    (60, 80),  # CRA with 1 leading picture  (display 59)
    (80, 100),  # CRA with 3 leading pictures (display 77, 78, 79)
]
TOTAL_FRAMES = 100


@pytest.fixture(scope="module")
def decoder():
    return nvc.CreateGopDecoder(maxfiles=1, iGpu=0)


def _get_gop(decoder, fid):
    """``GetGOPList`` for a single frame, unwrapped to ``(data, first, gop_len)``."""
    data, first_ids, gop_lens = decoder.GetGOPList([OPEN_GOP_SAMPLE], [fid], useGOPCache=False)[0]
    return data, int(first_ids[0]), int(gop_lens[0])


class TestGetGOPListPartition:
    """``GetGOPList`` must produce a non-overlapping partition of display indices."""

    def test_fixture_exists(self):
        assert os.path.exists(OPEN_GOP_SAMPLE), f"missing sample fixture: {OPEN_GOP_SAMPLE}"

    def test_partition_matches_expected(self, decoder):
        """Sweep every frame index; (first, first+len) must equal the expected boundaries."""
        seen = set()
        for fid in range(TOTAL_FRAMES):
            _, first, glen = _get_gop(decoder, fid)
            seen.add((first, first + glen))
        assert sorted(seen) == EXPECTED_PARTITION

    def test_no_overlap_between_adjacent_gops(self, decoder):
        """For every pair of adjacent GOPs, the higher's start must equal the lower's end."""
        distinct = sorted({_get_gop(decoder, fid)[1:] for fid in range(TOTAL_FRAMES)})
        ends = [f + g for f, g in distinct]
        starts = [f for f, _ in distinct]
        for prev_end, next_start in zip(ends, starts[1:]):
            assert (
                prev_end == next_start
            ), f"GOPs not contiguous: previous ended at {prev_end}, next starts at {next_start}"

    def test_partition_covers_all_frames(self, decoder):
        """Every display index in [0, TOTAL_FRAMES) is covered by exactly one GOP."""
        coverage_count = [0] * TOTAL_FRAMES
        distinct = {_get_gop(decoder, fid)[1:] for fid in range(TOTAL_FRAMES)}
        for first, glen in distinct:
            for fid in range(first, first + glen):
                if 0 <= fid < TOTAL_FRAMES:
                    coverage_count[fid] += 1
        assert all(c == 1 for c in coverage_count), f"coverage breakdown: {coverage_count}"

    @pytest.mark.parametrize(
        "boundary_fid,expected_gop",
        [
            # Leading-picture indices map to the previous GOP in display order,
            # not to the IRAP that introduces them.
            (18, (0, 20)),
            (19, (0, 20)),
            (39, (20, 40)),
            (59, (40, 60)),
            (77, (60, 80)),
            (78, (60, 80)),
            (79, (60, 80)),
            # CRA indices themselves map to the GOP they start.
            (20, (20, 40)),
            (40, (40, 60)),
            (80, (80, 100)),
        ],
    )
    def test_boundary_fid_maps_to_correct_gop(self, decoder, boundary_fid, expected_gop):
        _, first, glen = _get_gop(decoder, boundary_fid)
        assert (first, first + glen) == expected_gop

    def test_decodefromgop_rejects_cross_gop_frame(self, decoder):
        """``DecodeFromGOP`` raises when the requested frame lies outside the
        GOP's declared ``[first_frame_id, first_frame_id + gop_len)`` range,
        producing a clear API-boundary error instead of a downstream decode
        failure.
        """
        # GOP-A covers [0, 20). Frame 25 lives in GOP-B and must be rejected.
        gop_a_data, _, _ = _get_gop(decoder, 10)
        with pytest.raises(Exception) as exc_info:
            decoder.DecodeFromGOP(gop_a_data, [OPEN_GOP_SAMPLE], [25])
        assert "GOP range" in str(exc_info.value) or "frame_id" in str(exc_info.value)


class TestSharedGopStoreOpenGop:
    """End-to-end check that ``GetGOPList`` + ``SharedGopStore.put`` produce
    independent entries for adjacent open-GOP GOPs."""

    @pytest.fixture
    def store(self):
        store_id = 9100 + (os.getpid() % 100)
        s = SharedGopStore.create(capacity=8, store_id=store_id)
        yield s
        s.cleanup()

    def test_adjacent_gops_stored_independently(self, decoder, store):
        """Putting GOP-A ([0, 20)) and GOP-B ([20, 40)) creates two distinct
        shm slots, and ``lookup`` resolves each frame index to the GOP whose
        range contains it.
        """
        gop_a, first_a, len_a = _get_gop(decoder, 10)
        gop_b, first_b, len_b = _get_gop(decoder, 25)
        # Sanity-check the fixture: GOP-A and GOP-B must be adjacent, distinct.
        assert (first_a, len_a) == (0, 20)
        assert (first_b, len_b) == (20, 20)

        ref_a = store.put(OPEN_GOP_SAMPLE, first_a, len_a, gop_a)
        ref_b = store.put(OPEN_GOP_SAMPLE, first_b, len_b, gop_b)
        assert ref_a.first_frame_id == 0 and ref_a.gop_len == 20
        assert ref_b.first_frame_id == 20 and ref_b.gop_len == 20
        assert ref_a.shm_name != ref_b.shm_name

        # Lookup at frame 25 must return GOP-B, not GOP-A.
        hit = store.lookup(OPEN_GOP_SAMPLE, 25)
        assert hit is not None
        assert hit.first_frame_id == 20


# Display indices that two adjacent GOPs would have both claimed under the
# pre-fix algorithm. Concretely, on the synthetic fixture the GOP starting at
# CRA_X would over-count leading pictures of CRA_(X+20), inflating its declared
# range past the next CRA. The frames listed below are the ones inside that
# over-counted tail, and are the canonical regression points: every decoding
# entry point must produce a real frame for each of them.
OVERLAP_FIDS = [40, 41, 60, 80]


class TestAllDecodingApisOnOverlapFrames:
    """Every public decoding entry point produces a non-empty 256x256 frame
    for each display index that previously sat inside two GOPs' claimed
    ranges simultaneously."""

    def _assert_yuv_nv12(self, frames):
        """One NV12 YUV frame: tensor shape (256 * 3 / 2, 256), non-zero."""
        assert len(frames) == 1
        t = torch.as_tensor(frames[0])
        assert t.shape == (384, 256), f"unexpected NV12 shape {tuple(t.shape)}"
        assert torch.any(t != 0)

    def _assert_rgb(self, frames):
        """One RGB frame: tensor shape (256, 256, 3), non-zero."""
        assert len(frames) == 1
        t = torch.as_tensor(frames[0])
        assert t.shape == (256, 256, 3), f"unexpected RGB shape {tuple(t.shape)}"
        assert torch.any(t != 0)

    @pytest.mark.parametrize("fid", OVERLAP_FIDS)
    def test_decode(self, decoder, fid):
        """``Decode`` — random-access YUV path (no GOP intermediate)."""
        frames = decoder.Decode([OPEN_GOP_SAMPLE], [fid])
        self._assert_yuv_nv12(frames)

    @pytest.mark.parametrize("fid", OVERLAP_FIDS)
    def test_decode_n12_to_rgb(self, decoder, fid):
        """``DecodeN12ToRGB`` — random-access RGB path."""
        frames = decoder.DecodeN12ToRGB([OPEN_GOP_SAMPLE], [fid])
        self._assert_rgb(frames)

    @pytest.mark.parametrize("fid", OVERLAP_FIDS)
    def test_getgop_decode_from_gop(self, decoder, fid):
        """``GetGOPList`` + ``DecodeFromGOP`` — single-bundle YUV path."""
        gop_data, _, _ = decoder.GetGOPList([OPEN_GOP_SAMPLE], [fid])[0]
        frames = decoder.DecodeFromGOP(gop_data, [OPEN_GOP_SAMPLE], [fid])
        self._assert_yuv_nv12(frames)

    @pytest.mark.parametrize("fid", OVERLAP_FIDS)
    def test_getgoplist_decode_from_gop_list_rgb(self, decoder, fid):
        """``GetGOPList`` + ``DecodeFromGOPListRGB`` — per-file-bundle RGB path."""
        gop_list = decoder.GetGOPList([OPEN_GOP_SAMPLE], [fid])
        gop_data_list = [bundle for bundle, _, _ in gop_list]
        frames = decoder.DecodeFromGOPListRGB(gop_data_list, [OPEN_GOP_SAMPLE], [fid], False)
        self._assert_rgb(frames)
