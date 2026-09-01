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

import os

import utils
import accvlab.on_demand_video_decoder as nvc


def _video_without_color_range(filename="moving_shape_circle_h265.mp4"):
    path = os.path.join(utils.get_data_dir(), "sample_clip", filename)
    assert os.path.exists(path), f"test data missing: {path}"
    return path


def _get_gop_data(filepath, frame_id=0):
    decoder = nvc.CreateGopDecoder(maxfiles=1, iGpu=0)
    gop_data, _first_frame_ids, _gop_lens = decoder.GetGOPList([filepath], [frame_id])[0]
    return gop_data


def test_sample_reader_warning_emitted_once_per_file(capfd):
    path = _video_without_color_range()
    reader = nvc.CreateSampleReader(num_of_set=1, num_of_file=1, iGpu=0)

    reader.DecodeN12ToRGB([path], [0], False)
    reader.DecodeN12ToRGB([path], [1], False)

    output = capfd.readouterr().out
    warning = "PyNvVideoReader could not obtain color range"
    assert output.count(warning) == 1
    assert path in output


def test_sample_reader_warning_can_be_suppressed(capfd):
    path = _video_without_color_range()
    reader = nvc.CreateSampleReader(
        num_of_set=1,
        num_of_file=1,
        iGpu=0,
        suppressNoColorRangeWarning=True,
    )

    reader.DecodeN12ToRGB([path], [0], False)

    assert "PyNvVideoReader could not obtain color range" not in capfd.readouterr().out


def test_sample_reader_warning_resets_when_cache_slot_changes_file(capfd):
    first_path = _video_without_color_range("moving_shape_circle_h265.mp4")
    second_path = _video_without_color_range("moving_shape_ellipse_h265.mp4")
    reader = nvc.CreateSampleReader(num_of_set=1, num_of_file=1, iGpu=0)

    reader.DecodeN12ToRGB([first_path], [0], False)
    reader.DecodeN12ToRGB([second_path], [0], False)

    output = capfd.readouterr().out
    assert output.count("PyNvVideoReader could not obtain color range") == 2
    assert first_path in output
    assert second_path in output


def test_batch_async_stream_reader_warning_can_be_suppressed(capfd):
    path = _video_without_color_range()
    reader = nvc.CreateBatchAsyncStreamReader(
        num_of_set=1,
        num_of_file=1,
        max_frames_per_decode_call=1,
        iGpu=0,
        suppressNoColorRangeWarning=True,
    )

    reader.Decode([path], [[0]], False)
    reader.GetBuffer([path], [[0]], False)

    assert "PyNvVideoReader could not obtain color range" not in capfd.readouterr().out


def test_gop_decoder_warning_emitted_once_per_file(capfd):
    path = _video_without_color_range()
    decoder = nvc.CreateGopDecoder(maxfiles=1, iGpu=0)

    decoder.DecodeN12ToRGB([path], [0])
    decoder.DecodeN12ToRGB([path], [1])

    output = capfd.readouterr().out
    warning = "PyNvGopDecoder could not obtain color range"
    assert output.count(warning) == 1
    assert path in output


def test_gop_decoder_warning_can_be_suppressed(capfd):
    path = _video_without_color_range()
    decoder = nvc.CreateGopDecoder(maxfiles=1, iGpu=0, suppressNoColorRangeWarning=True)

    decoder.DecodeN12ToRGB([path], [0])

    assert "PyNvGopDecoder could not obtain color range" not in capfd.readouterr().out


def test_batch_async_gop_decoder_warning_emitted_once_per_file(capfd):
    path = _video_without_color_range()
    gop_data = _get_gop_data(path)
    frame_ids = [[0, 1]]
    decoder = nvc.CreateBatchAsyncGopDecoder(maxfiles=1, max_frames_per_decode_call=2, iGpu=0)

    decoder.DecodeFromGOPListRGB([[gop_data]], [path], frame_ids, False)
    decoder.DecodeFromGOPListRGBGetBuffer([path], frame_ids, False)

    output = capfd.readouterr().out
    warning = "PyNvGopDecoder could not obtain color range"
    assert output.count(warning) == 1
    assert path in output


def test_batch_async_gop_decoder_warning_can_be_suppressed(capfd):
    path = _video_without_color_range()
    gop_data = _get_gop_data(path)
    frame_ids = [[0]]
    decoder = nvc.CreateBatchAsyncGopDecoder(
        maxfiles=1,
        max_frames_per_decode_call=1,
        iGpu=0,
        suppressNoColorRangeWarning=True,
    )

    decoder.DecodeFromGOPListRGB([[gop_data]], [path], frame_ids, False)
    decoder.DecodeFromGOPListRGBGetBuffer([path], frame_ids, False)

    assert "PyNvGopDecoder could not obtain color range" not in capfd.readouterr().out
