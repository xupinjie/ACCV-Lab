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

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <cuviddec.h>
#include <libavutil/pixfmt.h>

class NvDecoder;
struct DecodedFrameExt;
class RGBFrame;

enum Pixel_Format {
    Pixel_Format_UNDEFINED = 0,
    Pixel_Format_NV12 = 3,
    Pixel_Format_YUV444 = 4,
    Pixel_Format_P016 = 5,
    Pixel_Format_YUV444_16Bit = 6

};

namespace accvlab::on_demand_video_decoder::internal::frame_output {

// Tightly packed layouts exposed by the public frame-output APIs. RGB8 also
// covers BGR8 because channel order does not change the allocation size.
enum class FrameOutputFormat : uint8_t {
    RGB8,
    NV12,
    P016,
    YUV444,
    YUV444_16Bit,
};

FrameOutputFormat output_format_from_pixel_format(Pixel_Format format);
Pixel_Format pixel_format_from_surface(cudaVideoSurfaceFormat surface_format);
FrameOutputFormat output_format_from_surface(cudaVideoSurfaceFormat surface_format);
FrameOutputFormat output_format_from_av_pixel_format(AVPixelFormat pixel_format);

// The single size-calculation entry point for all tightly packed RGB/YUV
// frames exposed by this package.
size_t frame_bytes(FrameOutputFormat format, size_t height, size_t width);

// Convert one NVDEC surface into an RGB/BGR frame backed by output_buffer.
// When is_async is true, work is only enqueued on decoder.GetStream(); the caller
// owns the terminal synchronization and must keep both buffers alive until then.
RGBFrame convert_decoded_frame_to_rgb(NvDecoder& decoder, const uint8_t* decoded_surface,
                                      CUdeviceptr output_buffer, AVColorRange color_range, bool as_bgr,
                                      bool is_async);

// Copy one NVDEC surface into output_buffer and expose its native YUV planes.
DecodedFrameExt copy_decoded_frame_to_yuv(NvDecoder& decoder, const uint8_t* decoded_surface,
                                          CUdeviceptr output_buffer, AVColorRange color_range,
                                          int64_t timestamp, bool is_async);

// Copy an existing RGB/BGR frame into aggregator-owned storage.
RGBFrame copy_rgb_frame(const RGBFrame& source, CUdeviceptr destination, CUstream destination_stream,
                        bool as_bgr, bool is_async);

// Copy an existing tightly packed YUV frame into aggregator-owned storage.
DecodedFrameExt copy_yuv_frame(const DecodedFrameExt& source, CUdeviceptr destination,
                               CUstream destination_stream, bool is_async);

}  // namespace accvlab::on_demand_video_decoder::internal::frame_output
