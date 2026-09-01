/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "FrameOutput.hpp"

#include "ColorConvertKernels.cuh"
#include "NvCodecUtils.h"
#include "NvDecoder/NvDecoder.h"
#include "PyDecodedFrameExt.hpp"
#include "PyRGBFrame.hpp"
#include "nvtx3/nvtx3.hpp"

#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace accvlab::on_demand_video_decoder::internal::frame_output {

namespace {

constexpr size_t chroma_height_420(size_t height) { return (height + 1) / 2; }

}  // namespace

FrameOutputFormat output_format_from_pixel_format(Pixel_Format format) {
    switch (format) {
        case Pixel_Format_NV12:
            return FrameOutputFormat::NV12;
        case Pixel_Format_P016:
            return FrameOutputFormat::P016;
        case Pixel_Format_YUV444:
            return FrameOutputFormat::YUV444;
        case Pixel_Format_YUV444_16Bit:
            return FrameOutputFormat::YUV444_16Bit;
        default:
            throw std::invalid_argument("Unsupported pixel format for frame output: " +
                                        std::to_string(static_cast<int>(format)));
    }
}

Pixel_Format pixel_format_from_surface(cudaVideoSurfaceFormat surface_format) {
    switch (surface_format) {
        case cudaVideoSurfaceFormat_NV12:
            return Pixel_Format_NV12;
        case cudaVideoSurfaceFormat_P016:
            return Pixel_Format_P016;
        case cudaVideoSurfaceFormat_YUV444:
            return Pixel_Format_YUV444;
        case cudaVideoSurfaceFormat_YUV444_16Bit:
            return Pixel_Format_YUV444_16Bit;
        default:
            return Pixel_Format_UNDEFINED;
    }
}

FrameOutputFormat output_format_from_surface(cudaVideoSurfaceFormat surface_format) {
    return output_format_from_pixel_format(pixel_format_from_surface(surface_format));
}

FrameOutputFormat output_format_from_av_pixel_format(AVPixelFormat pixel_format) {
    switch (pixel_format) {
        case AV_PIX_FMT_YUV420P:
        case AV_PIX_FMT_YUVJ420P:
        case AV_PIX_FMT_YUVJ422P:
        case AV_PIX_FMT_YUVJ444P:
        case AV_PIX_FMT_GRAY8:
            return FrameOutputFormat::NV12;
        case AV_PIX_FMT_YUV420P10LE:
        case AV_PIX_FMT_YUV420P12LE:
        case AV_PIX_FMT_GRAY10LE:
            return FrameOutputFormat::P016;
        case AV_PIX_FMT_YUV444P:
            return FrameOutputFormat::YUV444;
        case AV_PIX_FMT_YUV444P10LE:
        case AV_PIX_FMT_YUV444P12LE:
            return FrameOutputFormat::YUV444_16Bit;
        default:
            throw std::invalid_argument("Unsupported FFmpeg pixel format for frame output: " +
                                        std::to_string(static_cast<int>(pixel_format)));
    }
}

size_t frame_bytes(FrameOutputFormat format, size_t height, size_t width) {
    switch (format) {
        case FrameOutputFormat::RGB8:
            return height * width * 3;
        case FrameOutputFormat::NV12:
            return width * (height + chroma_height_420(height));
        case FrameOutputFormat::P016:
            return 2 * width * (height + chroma_height_420(height));
        case FrameOutputFormat::YUV444:
            return height * width * 3;
        case FrameOutputFormat::YUV444_16Bit:
            return height * width * 6;
        default:
            throw std::invalid_argument("Unsupported frame output format: " +
                                        std::to_string(static_cast<int>(format)));
    }
}

namespace {

RGBFrame make_rgb_frame_view(size_t height, size_t width, CUdeviceptr data, CUstream stream) {
    const std::vector<size_t> shape{height, width, 3};
    const std::vector<size_t> stride{width * 3, 3, 1};
    return RGBFrame(shape, stride, "|u1", reinterpret_cast<size_t>(stream), data, false, false);
}

DecodedFrameExt make_yuv_frame_view(Pixel_Format format, size_t height, size_t width, int64_t timestamp,
                                    DecodedFrameExt::ColorRange color_range, CUdeviceptr data,
                                    CUstream stream) {
    DecodedFrameExt frame;
    frame.format = format;
    frame.timestamp = timestamp;
    frame.color_range = color_range;
    const size_t stream_id = reinterpret_cast<size_t>(stream);

    switch (format) {
        case Pixel_Format_NV12: {
            const size_t chroma_height = chroma_height_420(height);
            frame.views.push_back(
                CAIMemoryView{{height, width, 1}, {width, 1, 1}, "|u1", stream_id, data, false});
            frame.views.push_back(CAIMemoryView{{chroma_height, width / 2, 2},
                                                {width / 2 * 2, 2, 1},
                                                "|u1",
                                                stream_id,
                                                data + height * width,
                                                false});
            frame.extBuf->LoadDLPack({height + chroma_height, width}, {width, 1}, "|u1", stream_id, data,
                                     false);
            break;
        }
        case Pixel_Format_P016: {
            const size_t chroma_height = chroma_height_420(height);
            frame.views.push_back(
                CAIMemoryView{{height, width, 1}, {width * 2, 2, 2}, "|u2", stream_id, data, false});
            frame.views.push_back(CAIMemoryView{{chroma_height, width / 2, 2},
                                                {width * 2, 4, 2},
                                                "|u2",
                                                stream_id,
                                                data + 2 * height * width,
                                                false});
            break;
        }
        case Pixel_Format_YUV444:
            frame.views.push_back(
                CAIMemoryView{{height, width, 1}, {width, 1, 1}, "|u1", stream_id, data, false});
            frame.views.push_back(CAIMemoryView{
                {height, width, 1}, {width, 1, 1}, "|u1", stream_id, data + height * width, false});
            frame.views.push_back(CAIMemoryView{
                {height, width, 1}, {width, 1, 1}, "|u1", stream_id, data + 2 * height * width, false});
            break;
        case Pixel_Format_YUV444_16Bit:
            frame.views.push_back(
                CAIMemoryView{{height, width, 1}, {width * 2, 2, 2}, "|u2", stream_id, data, false});
            frame.views.push_back(CAIMemoryView{
                {height, width, 1}, {width * 2, 2, 2}, "|u2", stream_id, data + 2 * height * width, false});
            frame.views.push_back(CAIMemoryView{
                {height, width, 1}, {width * 2, 2, 2}, "|u2", stream_id, data + 4 * height * width, false});
            break;
        default:
            throw std::invalid_argument("Unsupported pixel format for YUV output: " +
                                        std::to_string(static_cast<int>(format)));
    }

    return frame;
}

}  // namespace

RGBFrame convert_decoded_frame_to_rgb(NvDecoder& decoder, const uint8_t* decoded_surface,
                                      CUdeviceptr output_buffer, AVColorRange color_range, bool as_bgr,
                                      bool is_async) {
    const Pixel_Format format = pixel_format_from_surface(decoder.GetOutputFormat());
    const size_t width = static_cast<size_t>(decoder.GetWidth());
    const size_t height = static_cast<size_t>(decoder.GetHeight());
    RGBFrame output = make_rgb_frame_view(height, width, output_buffer, decoder.GetStream());

    if (format != Pixel_Format_NV12) {
        throw std::invalid_argument("[ERROR] Conversion to RGB/BGR only supported for videos in NV12-format");
    }

    const CAIMemoryView y_view{{height, width, 1},
                               {width, 1, 1},
                               "|u1",
                               reinterpret_cast<size_t>(decoder.GetStream()),
                               reinterpret_cast<CUdeviceptr>(decoded_surface),
                               false};
    const CAIMemoryView uv_view{{chroma_height_420(height), width / 2, 2},
                                {width / 2 * 2, 2, 1},
                                "|u1",
                                reinterpret_cast<size_t>(decoder.GetStream()),
                                reinterpret_cast<CUdeviceptr>(decoded_surface + width * height),
                                false};

    {
        nvtx3::scoped_range range{"Color convert"};
        const bool is_full_range = color_range == AVColorRange::AVCOL_RANGE_JPEG;
        convert_nv12_to_rgb(y_view, uv_view, output, is_full_range, as_bgr);
    }

    if (!is_async) {
        CUDA_DRVAPI_CALL(cuStreamSynchronize(decoder.GetStream()));
    }
    return output;
}

DecodedFrameExt copy_decoded_frame_to_yuv(NvDecoder& decoder, const uint8_t* decoded_surface,
                                          CUdeviceptr output_buffer, AVColorRange color_range,
                                          int64_t timestamp, bool is_async) {
    const Pixel_Format format = pixel_format_from_surface(decoder.GetOutputFormat());
    if (format == Pixel_Format_UNDEFINED) {
        throw std::runtime_error("[ERROR] Unsupported pixel format for YUV output");
    }

    const size_t height = static_cast<size_t>(decoder.GetHeight());
    const size_t width = static_cast<size_t>(decoder.GetWidth());
    const size_t output_bytes = frame_bytes(output_format_from_pixel_format(format), height, width);
    CUDA_DRVAPI_CALL(cuMemcpyDtoDAsync(output_buffer, reinterpret_cast<CUdeviceptr>(decoded_surface),
                                       output_bytes, decoder.GetStream()));

    DecodedFrameExt output =
        make_yuv_frame_view(format, height, width, timestamp, DecodedFrameExt::ConvertColorRange(color_range),
                            output_buffer, decoder.GetStream());

    if (!is_async) {
        CUDA_DRVAPI_CALL(cuStreamSynchronize(decoder.GetStream()));
    }
    return output;
}

RGBFrame copy_rgb_frame(const RGBFrame& source, CUdeviceptr destination, CUstream destination_stream,
                        bool as_bgr, bool is_async) {
    const size_t height = std::get<0>(source.shape);
    const size_t width = std::get<1>(source.shape);

    const size_t output_bytes = frame_bytes(FrameOutputFormat::RGB8, height, width);
    CUDA_DRVAPI_CALL(cuMemcpyDtoDAsync(destination, source.data, output_bytes, destination_stream));

    const std::vector<size_t> shape{height, width, 3};
    const std::vector<size_t> stride{std::get<0>(source.stride), std::get<1>(source.stride),
                                     std::get<2>(source.stride)};
    RGBFrame output(shape, stride, source.typestr, reinterpret_cast<size_t>(destination_stream), destination,
                    false, as_bgr);

    if (!is_async) {
        CUDA_DRVAPI_CALL(cuStreamSynchronize(destination_stream));
    }
    return output;
}

DecodedFrameExt copy_yuv_frame(const DecodedFrameExt& source, CUdeviceptr destination,
                               CUstream destination_stream, bool is_async) {
    const size_t height = source.views[0].shape[0];
    const size_t width = source.views[0].shape[1];
    const size_t output_bytes = frame_bytes(output_format_from_pixel_format(source.format), height, width);
    CUDA_DRVAPI_CALL(cuMemcpyDtoDAsync(destination, source.views[0].data, output_bytes, destination_stream));

    DecodedFrameExt output = make_yuv_frame_view(source.format, height, width, source.timestamp,
                                                 source.color_range, destination, destination_stream);

    if (!is_async) {
        CUDA_DRVAPI_CALL(cuStreamSynchronize(destination_stream));
    }
    return output;
}

}  // namespace accvlab::on_demand_video_decoder::internal::frame_output
