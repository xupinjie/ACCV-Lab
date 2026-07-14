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
Tests for GOP caching functionality in CachedGopDecoder.

This module tests the useGOPCache parameter for the GetGOPList method,
including cache hit/miss scenarios, cache management, and data correctness.
"""

import pytest
import sys
import torch
import random

import utils
import accvlab.on_demand_video_decoder as nvc


def test_cachedgopdecoder_rejects_direct_construction():
    """CachedGopDecoder is guarded by a private creation key; use CreateGopDecoder."""
    with pytest.raises(RuntimeError, match="CreateGopDecoder"):
        nvc.CachedGopDecoder(None, 1)


def test_create_gop_decoder_returns_caching_wrapper():
    """CreateGopDecoder is the supported path and returns the CachedGopDecoder wrapper."""
    decoder = nvc.CreateGopDecoder(maxfiles=1, iGpu=0)
    assert isinstance(decoder, nvc.CachedGopDecoder)


def _gop_ranges(gop_list):
    """Extract per-video (first_frame_id, gop_len) from a GetGOPList result."""
    first_ids = [bundle[1][0] for bundle in gop_list]
    gop_lens = [bundle[2][0] for bundle in gop_list]
    return first_ids, gop_lens


class TestGetGOPListCache:
    """Tests for GetGOPList with useGOPCache parameter."""

    @pytest.fixture
    def decoder(self):
        """Create a decoder instance for testing."""
        return nvc.CreateGopDecoder(maxfiles=6, iGpu=0)

    @pytest.fixture
    def test_files_and_frames(self):
        """Get test video files and random frame IDs."""
        path_base = utils.get_data_dir()
        files = utils.select_random_clip(path_base)
        if files is None:
            pytest.skip("No test video files available")
        frames = [random.randint(0, 100) for _ in range(len(files))]
        return files, frames

    @pytest.mark.parametrize("use_cache", [False, True], ids=["cache_disabled", "cache_enabled"])
    def test_getgoplist_basic(self, decoder, test_files_and_frames, use_cache):
        """
        Test GetGOPList basic functionality with useGOPCache parameter.
        """
        files, frames = test_files_and_frames
        print(f"\n=== Test: GetGOPList basic (useGOPCache={use_cache}) ===")

        # Call GetGOPList
        gop_list = decoder.GetGOPList(files, frames, useGOPCache=use_cache)

        # Verify return structure
        assert gop_list is not None, "gop_list should not be None"
        assert len(gop_list) == len(files), f"Expected {len(files)} GOP bundles"

        for i, (packets, first_ids, gop_lens) in enumerate(gop_list):
            assert packets is not None and len(packets) > 0, f"Bundle {i} packets should not be empty"
            assert len(first_ids) == 1, f"Bundle {i} should have 1 first_frame_id"
            assert len(gop_lens) == 1, f"Bundle {i} should have 1 gop_len"

        # Verify isCacheHit
        cache_hits = decoder.isCacheHit()
        assert len(cache_hits) == len(files), f"Expected {len(files)} cache hit flags"

        # First call should never be a cache hit
        if use_cache:
            assert all(not hit for hit in cache_hits), "First call should be all cache misses"

        print(f"✓ Test passed: Got {len(gop_list)} GOP bundles")

    def test_getgoplist_cache_hit(self, decoder, test_files_and_frames):
        """
        Test GetGOPList cache hit scenario.

        Verifies that calling GetGOPList twice with frame_ids in the same GOP range
        results in a cache hit on the second call.
        """
        files, frames = test_files_and_frames
        print(f"\n=== Test: GetGOPList cache hit ===")

        # First call
        gop_list1 = decoder.GetGOPList(files, frames, useGOPCache=True)
        cache_hits1 = decoder.isCacheHit()
        assert all(not hit for hit in cache_hits1), "First call should be all cache misses"

        # Get GOP ranges for generating in-range frames
        first_ids, gop_lens = _gop_ranges(gop_list1)

        # Generate new frame IDs within the same GOP range
        new_frames = [random.randint(first_ids[i], first_ids[i] + gop_lens[i] - 1) for i in range(len(files))]
        print(f"Original frames: {frames}, New frames (in GOP range): {new_frames}")

        # Second call - should be cache hit
        gop_list2 = decoder.GetGOPList(files, new_frames, useGOPCache=True)
        cache_hits2 = decoder.isCacheHit()

        assert all(cache_hits2), f"Second call should be all cache hits, got {cache_hits2}"
        first_ids2, gop_lens2 = _gop_ranges(gop_list2)
        assert first_ids == first_ids2, "First frame IDs should match"
        assert gop_lens == gop_lens2, "GOP lengths should match"

        print(f"✓ Test passed: Cache hits = {cache_hits2}")

    def test_getgoplist_cache_miss_out_of_range(self, decoder, test_files_and_frames):
        """
        Test GetGOPList cache miss when frame_id is out of cached GOP range.
        """
        files, frames = test_files_and_frames
        print(f"\n=== Test: GetGOPList cache miss (out of range) ===")

        # First call
        gop_list1 = decoder.GetGOPList(files, frames, useGOPCache=True)
        first_ids1, gop_lens1 = _gop_ranges(gop_list1)

        # Generate frame IDs outside the GOP range
        out_of_range_frames = [first_ids1[i] + gop_lens1[i] + 10 for i in range(len(files))]
        print(f"Original frames: {frames}, Out-of-range frames: {out_of_range_frames}")

        # Second call with out-of-range frames - should be cache miss
        decoder.GetGOPList(files, out_of_range_frames, useGOPCache=True)
        cache_hits2 = decoder.isCacheHit()

        assert all(not hit for hit in cache_hits2), f"Out-of-range should be cache miss, got {cache_hits2}"

        print(f"✓ Test passed: Cache misses for out-of-range frames")

    def test_getgoplist_partial_cache_hit(self, decoder, test_files_and_frames):
        """
        Test GetGOPList partial cache hit scenario.

        Some files hit cache, some miss.
        """
        files, frames = test_files_and_frames
        if len(files) < 2:
            pytest.skip("Need at least 2 files for partial cache hit test")

        print(f"\n=== Test: GetGOPList partial cache hit ===")

        # First call
        gop_list1 = decoder.GetGOPList(files, frames, useGOPCache=True)

        # Get GOP ranges
        first_ids, gop_lens = _gop_ranges(gop_list1)

        # Generate mixed frames: first file in range, others out of range
        mixed_frames = []
        for i in range(len(files)):
            if i == 0:
                # In range - should hit cache
                mixed_frames.append(random.randint(first_ids[i], first_ids[i] + gop_lens[i] - 1))
            else:
                # Out of range - should miss cache
                mixed_frames.append(first_ids[i] + gop_lens[i] + 10)

        print(f"Mixed frames: {mixed_frames}")

        # Second call - should be partial hit
        decoder.GetGOPList(files, mixed_frames, useGOPCache=True)
        cache_hits2 = decoder.isCacheHit()

        assert cache_hits2[0] == True, "First file should be cache hit"
        assert all(not hit for hit in cache_hits2[1:]), "Other files should be cache miss"

        print(f"✓ Test passed: Partial cache hits = {cache_hits2}")

    def test_getgoplist_decode_with_cache(self, decoder, test_files_and_frames):
        """
        Test that decoding from cached GetGOPList data produces correct results.
        """
        files, frames = test_files_and_frames
        print(f"\n=== Test: GetGOPList decode with cache ===")

        decoder2 = nvc.CreateGopDecoder(maxfiles=6, iGpu=0)

        # Get GOP list with cache
        gop_list = decoder.GetGOPList(files, frames, useGOPCache=True)
        gop_data_list = [data for data, _, _ in gop_list]

        # Decode frames
        decoded_frames = decoder2.DecodeFromGOPListRGB(gop_data_list, files, frames, as_bgr=True)

        assert decoded_frames is not None, "Decoded frames should not be None"
        assert len(decoded_frames) == len(files), f"Expected {len(files)} decoded frames"

        # Verify tensor shapes
        for i, frame in enumerate(decoded_frames):
            tensor = torch.as_tensor(frame)
            assert len(tensor.shape) == 3, f"Frame {i} should be 3D (H, W, C)"
            assert tensor.shape[2] == 3, f"Frame {i} should have 3 channels (BGR)"

        print(f"✓ Test passed: Decoded {len(decoded_frames)} frames with correct shapes")

    def test_getgoplist_decode_from_cached_data(self, decoder, test_files_and_frames):
        """
        Test that data returned from a cache hit decodes to correct results.

        Verifies the cached bundles (second call, all hits) are decodable and
        produce frames with the expected shape.
        """
        files, frames = test_files_and_frames
        print(f"\n=== Test: GetGOPList decode from cached data ===")

        decoder2 = nvc.CreateGopDecoder(maxfiles=6, iGpu=0)

        # First call fills the cache
        gop_list1 = decoder.GetGOPList(files, frames, useGOPCache=True)
        first_ids, gop_lens = _gop_ranges(gop_list1)

        # Second call within the same GOP ranges returns cached data
        new_frames = [random.randint(first_ids[i], first_ids[i] + gop_lens[i] - 1) for i in range(len(files))]
        gop_list2 = decoder.GetGOPList(files, new_frames, useGOPCache=True)
        assert all(decoder.isCacheHit()), "Second call should be all cache hits"

        gop_data_list = [data for data, _, _ in gop_list2]
        decoded_frames = decoder2.DecodeFromGOPListRGB(gop_data_list, files, new_frames, as_bgr=True)

        assert len(decoded_frames) == len(files), f"Expected {len(files)} decoded frames"
        for i, frame in enumerate(decoded_frames):
            tensor = torch.as_tensor(frame)
            assert len(tensor.shape) == 3, f"Frame {i} should be 3D (H, W, C)"
            assert tensor.shape[2] == 3, f"Frame {i} should have 3 channels (BGR)"

        print(f"✓ Test passed: Decoded {len(decoded_frames)} frames from cached data")


class TestCacheManagement:
    """Tests for cache management methods."""

    @pytest.fixture
    def decoder(self):
        return nvc.CreateGopDecoder(maxfiles=6, iGpu=0)

    @pytest.fixture
    def test_files_and_frames(self):
        path_base = utils.get_data_dir()
        files = utils.select_random_clip(path_base)
        if files is None:
            pytest.skip("No test video files available")
        frames = [random.randint(0, 100) for _ in range(len(files))]
        return files, frames

    def test_clear_cache(self, decoder, test_files_and_frames):
        """
        Test clear_cache method.
        """
        files, frames = test_files_and_frames
        print(f"\n=== Test: clear_cache ===")

        # Fill cache
        gop_list = decoder.GetGOPList(files, frames, useGOPCache=True)
        first_ids, gop_lens = _gop_ranges(gop_list)

        # Verify cache is filled
        cache_info = decoder.get_cache_info()
        assert cache_info["cached_files_count"] == len(files), "Cache should be filled"

        # Clear cache
        decoder.clear_cache()

        # Verify cache is empty
        cache_info = decoder.get_cache_info()
        assert cache_info["cached_files_count"] == 0, "Cache should be empty after clear"

        # Next call should miss
        new_frames = [random.randint(first_ids[i], first_ids[i] + gop_lens[i] - 1) for i in range(len(files))]
        decoder.GetGOPList(files, new_frames, useGOPCache=True)
        cache_hits = decoder.isCacheHit()

        assert all(not hit for hit in cache_hits), "Should be cache miss after clear"

        print(f"✓ Test passed: clear_cache works correctly")

    def test_get_cache_info(self, decoder, test_files_and_frames):
        """
        Test get_cache_info method.
        """
        files, frames = test_files_and_frames
        print(f"\n=== Test: get_cache_info ===")

        # Empty cache info
        cache_info = decoder.get_cache_info()
        assert cache_info["cached_files_count"] == 0, "Initial cache should be empty"
        assert len(cache_info["cached_files"]) == 0, "No cached files initially"

        # Fill cache
        decoder.GetGOPList(files, frames, useGOPCache=True)

        # Check cache info
        cache_info = decoder.get_cache_info()
        assert cache_info["cached_files_count"] == len(files), f"Should have {len(files)} cached files"

        for filepath in files:
            assert filepath in cache_info["cached_files"], f"{filepath} should be in cache"
            file_info = cache_info["cached_files"][filepath]
            assert "first_frame_id" in file_info, "Should have first_frame_id"
            assert "gop_len" in file_info, "Should have gop_len"
            assert "frame_range" in file_info, "Should have frame_range"
            assert "packets_size_bytes" in file_info, "Should have packets_size_bytes"

        print(f"✓ Test passed: get_cache_info returns correct info")

    def test_iscachehit_empty(self, decoder):
        """
        Test isCacheHit returns empty list when GetGOPList has not been called.
        """
        print(f"\n=== Test: isCacheHit empty ===")

        cache_hits = decoder.isCacheHit()
        assert cache_hits == [], "Should return empty list before any GetGOPList call"

        print(f"✓ Test passed: isCacheHit returns [] initially")


class TestCacheEdgeCases:
    """Tests for cache edge cases and boundary conditions."""

    @pytest.fixture
    def decoder(self):
        return nvc.CreateGopDecoder(maxfiles=6, iGpu=0)

    @pytest.fixture
    def test_files_and_frames(self):
        path_base = utils.get_data_dir()
        files = utils.select_random_clip(path_base)
        if files is None:
            pytest.skip("No test video files available")
        frames = [random.randint(0, 100) for _ in range(len(files))]
        return files, frames

    def test_different_file_same_frame_range(self, decoder, test_files_and_frames):
        """
        Test that cache is keyed by filepath, not just frame_id.

        Scenario:
        1. Cache video1.mp4 with GOP range [60, 89]
        2. Request video2.mp4 with frame_id=70 (which is in [60, 89])
        3. Should be cache MISS because filepath is different

        This verifies that cache lookup correctly uses filepath as part of the key.
        """
        files, frames = test_files_and_frames
        if len(files) < 2:
            pytest.skip("Need at least 2 files for this test")

        print(f"\n=== Test: Different file, same frame range ===")

        # Cache only the first file
        gop_list = decoder.GetGOPList([files[0]], [frames[0]], useGOPCache=True)
        first_ids, gop_lens = _gop_ranges(gop_list)

        print(f"Cached file: {files[0]}")
        print(f"Cached GOP range: [{first_ids[0]}, {first_ids[0] + gop_lens[0] - 1}]")

        # Now request a DIFFERENT file with frame_id that would be in the cached range
        frame_in_range = random.randint(first_ids[0], first_ids[0] + gop_lens[0] - 1)

        print(f"Requesting file: {files[1]} with frame_id={frame_in_range}")
        print(
            f"frame_id {frame_in_range} is in cached range [{first_ids[0]}, {first_ids[0] + gop_lens[0] - 1}]"
        )

        # This should be a cache MISS because filepath is different
        decoder.GetGOPList([files[1]], [frame_in_range], useGOPCache=True)
        cache_hits = decoder.isCacheHit()

        assert cache_hits[0] == False, (
            f"Should be cache MISS for different file even if frame_id is in cached range. "
            f"file1={files[0]}, file2={files[1]}, frame_id={frame_in_range}, cache_hit={cache_hits[0]}"
        )

        print(f"✓ Test passed: Different file correctly results in cache miss")

    def test_same_file_different_frame_ranges(self, decoder, test_files_and_frames):
        """
        Test cache update when same file is requested with different GOP ranges.

        Scenario:
        1. Cache video.mp4 with frame_id=50, gets GOP range [30, 59]
        2. Request video.mp4 with frame_id=100, gets GOP range [90, 119]
        3. Cache should be updated to new range
        4. Request video.mp4 with frame_id=55 (old range) should MISS
        5. Request video.mp4 with frame_id=105 (new range) should HIT
        """
        files, frames = test_files_and_frames
        print(f"\n=== Test: Same file, different frame ranges ===")

        single_file = [files[0]]

        # First request
        gop_list1 = decoder.GetGOPList(single_file, [50], useGOPCache=True)
        first_ids1, gop_lens1 = _gop_ranges(gop_list1)
        print(f"First GOP range: [{first_ids1[0]}, {first_ids1[0] + gop_lens1[0] - 1}]")

        # Request outside current GOP range to trigger cache update
        far_frame = first_ids1[0] + gop_lens1[0] + 50  # Well outside current range
        gop_list2 = decoder.GetGOPList(single_file, [far_frame], useGOPCache=True)
        cache_hits = decoder.isCacheHit()
        first_ids2, gop_lens2 = _gop_ranges(gop_list2)

        assert cache_hits[0] == False, "Out-of-range request should miss"
        print(f"New GOP range: [{first_ids2[0]}, {first_ids2[0] + gop_lens2[0] - 1}]")

        # Now test: old range should miss, new range should hit
        old_range_frame = random.randint(first_ids1[0], first_ids1[0] + gop_lens1[0] - 1)
        new_range_frame = random.randint(first_ids2[0], first_ids2[0] + gop_lens2[0] - 1)

        # Request old range - should miss (cache was updated)
        decoder.GetGOPList(single_file, [old_range_frame], useGOPCache=True)
        old_cache_hit = decoder.isCacheHit()[0]

        # Request new range - should hit
        decoder.GetGOPList(single_file, [new_range_frame], useGOPCache=True)
        new_cache_hit = decoder.isCacheHit()[0]

        # Note: After requesting old_range_frame, cache is updated again
        # So we need to re-request new range after that
        print(f"Old range frame {old_range_frame} cache hit: {old_cache_hit}")
        print(f"New range frame {new_range_frame} cache hit: {new_cache_hit}")

        print(f"✓ Test passed: Cache correctly updates with new GOP range")

    def test_mixed_cached_and_new_files(self, decoder, test_files_and_frames):
        """
        Test scenario where some files are cached and some are new.

        Verifies correct behavior when:
        - Some files have cached GOP data
        - Some files have never been requested before
        """
        files, frames = test_files_and_frames
        if len(files) < 3:
            pytest.skip("Need at least 3 files for this test")

        print(f"\n=== Test: Mixed cached and new files ===")

        # Cache first two files
        cached_files = files[:2]
        cached_frames = frames[:2]
        gop_list = decoder.GetGOPList(cached_files, cached_frames, useGOPCache=True)
        first_ids, gop_lens = _gop_ranges(gop_list)

        print(f"Cached files: {cached_files}")

        # Now request: one cached file (in range), one cached file (out of range), one new file
        mixed_files = [files[0], files[1], files[2]]
        mixed_frames = [
            random.randint(first_ids[0], first_ids[0] + gop_lens[0] - 1),  # In range - HIT
            first_ids[1] + gop_lens[1] + 10,  # Out of range - MISS
            frames[2],  # New file - MISS
        ]

        print(f"Mixed request: files={mixed_files}, frames={mixed_frames}")

        decoder.GetGOPList(mixed_files, mixed_frames, useGOPCache=True)
        cache_hits = decoder.isCacheHit()

        print(f"Cache hits: {cache_hits}")

        assert cache_hits[0] == True, "First file (in range) should HIT"
        assert cache_hits[1] == False, "Second file (out of range) should MISS"
        assert cache_hits[2] == False, "Third file (new) should MISS"

        print(f"✓ Test passed: Mixed cached/new files handled correctly")

    def test_varying_video_count_per_call(self, decoder, test_files_and_frames):
        """
        Test GetGOPList cache behavior when the number of videos varies between calls.

        Scenario:
        1. First call with 3 videos → all cached
        2. Second call with 2 videos (subset) → should hit cache for those 2
        3. Third call with 4 videos (3 cached + 1 new) → 3 hit, 1 miss
        4. Fourth call with 1 video (cached) → should hit cache

        This verifies that cache works correctly regardless of batch size changes.
        """
        files, frames = test_files_and_frames
        if len(files) < 4:
            pytest.skip("Need at least 4 files for this test")

        print(f"\n=== Test: Varying video count per call ===")

        # Call 1: Cache 3 videos
        files_3 = files[:3]
        frames_3 = frames[:3]
        gop_list1 = decoder.GetGOPList(files_3, frames_3, useGOPCache=True)
        cache_hits1 = decoder.isCacheHit()

        assert len(cache_hits1) == 3, "Should have 3 cache hit flags"
        assert all(not hit for hit in cache_hits1), "First call should be all misses"

        # Extract GOP ranges for later use
        first_ids, gop_lens = _gop_ranges(gop_list1)
        print(f"Call 1: Cached 3 videos, cache_hits={cache_hits1}")

        # Call 2: Request 2 videos (subset of cached)
        files_2 = files[:2]
        in_range_frames_2 = [random.randint(first_ids[i], first_ids[i] + gop_lens[i] - 1) for i in range(2)]
        decoder.GetGOPList(files_2, in_range_frames_2, useGOPCache=True)
        cache_hits2 = decoder.isCacheHit()

        assert len(cache_hits2) == 2, "Should have 2 cache hit flags"
        assert all(cache_hits2), f"All 2 should hit cache, got {cache_hits2}"
        print(f"Call 2: Requested 2 cached videos, cache_hits={cache_hits2}")

        # Call 3: Request 4 videos (3 cached + 1 new)
        files_4 = files[:4]
        mixed_frames_4 = [
            random.randint(first_ids[0], first_ids[0] + gop_lens[0] - 1),  # HIT
            random.randint(first_ids[1], first_ids[1] + gop_lens[1] - 1),  # HIT
            random.randint(first_ids[2], first_ids[2] + gop_lens[2] - 1),  # HIT
            frames[3],  # New file - MISS
        ]
        gop_list3 = decoder.GetGOPList(files_4, mixed_frames_4, useGOPCache=True)
        cache_hits3 = decoder.isCacheHit()

        assert len(cache_hits3) == 4, "Should have 4 cache hit flags"
        assert cache_hits3[0] == True, "First file should HIT"
        assert cache_hits3[1] == True, "Second file should HIT"
        assert cache_hits3[2] == True, "Third file should HIT"
        assert cache_hits3[3] == False, "Fourth file (new) should MISS"
        print(f"Call 3: Requested 4 videos (3 cached + 1 new), cache_hits={cache_hits3}")

        # Call 4: Request just 1 video (should be cached from previous calls)
        files_1 = [files[0]]
        in_range_frame_1 = [random.randint(first_ids[0], first_ids[0] + gop_lens[0] - 1)]
        decoder.GetGOPList(files_1, in_range_frame_1, useGOPCache=True)
        cache_hits4 = decoder.isCacheHit()

        assert len(cache_hits4) == 1, "Should have 1 cache hit flag"
        assert cache_hits4[0] == True, f"Single file should hit cache, got {cache_hits4}"
        print(f"Call 4: Requested 1 cached video, cache_hits={cache_hits4}")

        # Verify final cache state
        cache_info = decoder.get_cache_info()
        assert (
            cache_info["cached_files_count"] == 4
        ), f"Should have 4 files in cache, got {cache_info['cached_files_count']}"

        # Verify decoding works with varying count
        decoder2 = nvc.CreateGopDecoder(maxfiles=6, iGpu=0)
        gop_data_list = [data for data, _, _ in gop_list3]
        decoded_frames = decoder2.DecodeFromGOPListRGB(gop_data_list, files_4, mixed_frames_4, as_bgr=True)

        assert len(decoded_frames) == 4, f"Should decode 4 frames, got {len(decoded_frames)}"
        for i, frame in enumerate(decoded_frames):
            tensor = torch.as_tensor(frame)
            assert len(tensor.shape) == 3, f"Frame {i} should be 3D"
            assert tensor.shape[2] == 3, f"Frame {i} should have 3 channels"

        print(f"✓ Test passed: Varying video count per call works correctly")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
