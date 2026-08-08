/*
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "PyDecodedFrameExt.hpp"
#include "FrameOutput.hpp"
#include <stdexcept>

namespace frame_output = accvlab::on_demand_video_decoder::internal::frame_output;

DecodedFrameExt::VideoSurfaceFormat DecodedFrameExt::GetVideoSurfaceFormat() const {
    const VideoSurfaceFormat res = ConvertPixelFormatToVideoSurfaceFormatOut(this->format);
    return res;
}

void DecodedFrameExt::SetVideoSurfaceFormat(cudaVideoSurfaceFormat video_format_in) {
    this->format = ConvertVideoSurfaceFormatInToPixelFormat(video_format_in);
}

void DecodedFrameExt::SetColorRange(AVColorRange color_range_in) {
    this->color_range = ConvertColorRange(color_range_in);
}

DecodedFrameExt::VideoSurfaceFormat DecodedFrameExt::ConvertPixelFormatToVideoSurfaceFormatOut(
    Pixel_Format video_format_in) {
    VideoSurfaceFormat res;
    switch (video_format_in) {
        case Pixel_Format::Pixel_Format_NV12:
            res = VideoSurfaceFormat::NV12;
            break;
        case Pixel_Format::Pixel_Format_P016:
            res = VideoSurfaceFormat::P016;
            break;
        case Pixel_Format::Pixel_Format_YUV444:
            res = VideoSurfaceFormat::YUV444;
            break;
        case Pixel_Format::Pixel_Format_YUV444_16Bit:
            res = VideoSurfaceFormat::YUV444_16Bit;
            break;
        default:
            throw std::invalid_argument("Got unexpected value " + std::to_string(video_format_in) +
                                        " for input argument `video_format_in`.");
    }
    return res;
}

Pixel_Format DecodedFrameExt::ConvertVideoSurfaceFormatInToPixelFormat(
    cudaVideoSurfaceFormat video_format_in) {
    const Pixel_Format result = frame_output::pixel_format_from_surface(video_format_in);
    if (result == Pixel_Format_UNDEFINED) {
        throw std::invalid_argument("Got unexpected value " + std::to_string(video_format_in) +
                                    " for input argument `video_format_in`.");
    }
    return result;
}

DecodedFrameExt::ColorRange DecodedFrameExt::ConvertColorRange(AVColorRange color_range_in) {
    ColorRange res;
    switch (color_range_in) {
        case AVColorRange::AVCOL_RANGE_JPEG:
            res = ColorRange::ColorRange_FULL;
            break;
        case AVColorRange::AVCOL_RANGE_MPEG:
            res = ColorRange::ColorRange_LIMITED;
            break;
        case AVColorRange::AVCOL_RANGE_UNSPECIFIED:
        case AVColorRange::AVCOL_RANGE_NB:
            res = ColorRange::ColorRange_UNSPECIFIED;
            break;
        default:
            throw std::invalid_argument("Got unexpected value " + std::to_string(color_range_in) +
                                        " for input argument `color_range_in`.");
    }
    return res;
}
