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

#pragma once

#include <string>

#include "PyCAIMemoryView.hpp"

#include "cuviddec.h"
#include <libavutil/pixfmt.h>

struct DecodedFrameExt : public DecodedFrame {
    enum class VideoSurfaceFormat {
        UNSPECIFIED = 0,
        NV12 = 1,
        P016 = 2,
        YUV444 = 3,
        YUV444_16Bit = 4,
    };

    enum VideoChromaFormat {
        VideoChromaFormat_UNSPECIFIED = 0,
        VideoChromaFormat_MONOCHROME = 1,
        VideoChromaFormat_YUV420 = 2,
        VideoChromaFormat_YUV422 = 3,
        VideoChromaFormat_YUV444 = 4
    };

    enum ColorRange {
        ColorRange_UNSPECIFIED = 0,
        ColorRange_LIMITED = 1,
        ColorRange_FULL = 2,
    };

    ColorRange color_range = ColorRange::ColorRange_UNSPECIFIED;
    DecodedFrameExt() = default;
    VideoSurfaceFormat GetVideoSurfaceFormat() const;

    void SetVideoSurfaceFormat(cudaVideoSurfaceFormat video_format_in);
    void SetColorRange(AVColorRange color_range_in);

    static VideoSurfaceFormat ConvertPixelFormatToVideoSurfaceFormatOut(Pixel_Format video_format_in);
    static Pixel_Format ConvertVideoSurfaceFormatInToPixelFormat(cudaVideoSurfaceFormat video_format_in);
    static ColorRange ConvertColorRange(AVColorRange color_range_in);
    static void Export(py::module& m) {
        py::enum_<DecodedFrameExt::ColorRange>(m, "VideoColorRange", py::module_local())
            .ENUM_VALUE(DecodedFrameExt::ColorRange, UNSPECIFIED)
            .ENUM_VALUE(DecodedFrameExt::ColorRange, FULL)
            .ENUM_VALUE(DecodedFrameExt::ColorRange, LIMITED);

        py::enum_<DecodedFrameExt::VideoChromaFormat>(m, "VideoChromaFormat", py::module_local())
            .ENUM_VALUE(DecodedFrameExt::VideoChromaFormat, UNSPECIFIED)
            .ENUM_VALUE(DecodedFrameExt::VideoChromaFormat, MONOCHROME)
            .ENUM_VALUE(DecodedFrameExt::VideoChromaFormat, YUV420)
            .ENUM_VALUE(DecodedFrameExt::VideoChromaFormat, YUV422)
            .ENUM_VALUE(DecodedFrameExt::VideoChromaFormat, YUV444);
        py::class_<DecodedFrameExt, std::shared_ptr<DecodedFrameExt>>(m, "DecodedFrameExt")
            .def_readonly("timestamp", &DecodedFrameExt::timestamp)
            //.def_readonly("chroma_format", &DecodedFrameExt::chroma_format)
            .def_property_readonly(
                "format",
                [](std::shared_ptr<DecodedFrameExt>& self) { return static_cast<int>(self->format); })
            .def_readonly("color_range", &DecodedFrameExt::color_range)
            .def("__repr__",
                 [](std::shared_ptr<DecodedFrameExt>& self) {
                     std::stringstream ss;
                     ss << "<DecodedFrameExt [";
                     ss << "timestamp=" << self->timestamp;
                     ss << ", format=" << static_cast<int>(self->format);
                     // ss << ", chroma_format=" <<
                     // py::str(py::cast(self->chroma_format));
                     ss << ", color_range=" << py::str(py::cast(self->color_range));
                     ss << ", " << py::str(py::cast(self->views));
                     ss << "]>";
                     return ss.str();
                 })
            .def(
                "framesize",
                [](std::shared_ptr<DecodedFrameExt>& self) {
                    int height = self->views.at(0).shape.at(0);
                    int width = self->views.at(0).shape.at(1);
                    int framesize = width * height * 1.5;
                    switch (self->format) {
                        case Pixel_Format_NV12:
                            break;
                        case Pixel_Format_P016:
                            framesize = width * height * 3;
                            break;
                        case Pixel_Format_YUV444:
                            framesize = width * height * 3;
                            break;
                        case Pixel_Format_YUV444_16Bit:
                            framesize = width * height * 6;
                            break;
                        default:
                            break;
                    }
                    return framesize;
                },
                R"pbdoc(
            return underlying views which implement CAI
            :param None: None
            )pbdoc")
            .def(
                "cuda", [](std::shared_ptr<DecodedFrameExt>& self) { return self->views; },
                R"pbdoc(
            return underlying views which implement CAI
            :param None: None
            )pbdoc")
            .def(
                "nvcv_image",
                [](std::shared_ptr<DecodedFrameExt>& self) {
                    switch (self->format) {
                        case Pixel_Format_NV12: {
                            size_t width = self->views.at(0).shape[1];
                            size_t height = self->views.at(0).shape[0] * 1.5;
                            CUdeviceptr data = self->views.at(0).data;
                            CUstream stream = self->views.at(0).stream;
                            self->views.clear();
                            self->views.push_back(
                                CAIMemoryView{{height, width, 1},
                                              {width, 2, 1},
                                              "|u1",
                                              reinterpret_cast<size_t>(stream),
                                              (data),
                                              false});  // hack for cvcuda tensor represenation
                        } break;
                        case Pixel_Format_YUV444: {
                            size_t width = self->views.at(0).shape[1];
                            size_t height = self->views.at(0).shape[0] * 3;
                            CUdeviceptr data = self->views.at(0).data;
                            CUstream stream = self->views.at(0).stream;
                            self->views.clear();
                            self->views.push_back(
                                CAIMemoryView{{height, width, 1},
                                              {width, 3, 1},
                                              "|u1",
                                              reinterpret_cast<size_t>(stream),
                                              (data),
                                              false});  // hack for cvcuda tensor represenation
                        } break;
                        default:
                            throw std::invalid_argument("[ERROR] only nv12 and yuv444 supported as of now");
                            break;
                    }
                    return self->views;
                },
                R"pbdoc(
            return underlying views which implement CAI
            :param None: None
            )pbdoc")
            // DL Pack Tensor
            .def_property_readonly(
                "shape", [](std::shared_ptr<DecodedFrameExt>& self) { return self->extBuf->shape(); },
                "Get the shape of the buffer as an array")
            .def_property_readonly(
                "strides", [](std::shared_ptr<DecodedFrameExt>& self) { return self->extBuf->strides(); },
                "Get the strides of the buffer")
            .def_property_readonly(
                "dtype", [](std::shared_ptr<DecodedFrameExt>& self) { return self->extBuf->dtype(); },
                "Get the data type of the buffer")
            .def(
                "__dlpack__",
                [](std::shared_ptr<DecodedFrameExt>& self, py::object stream) {
                    return self->extBuf->dlpack(stream);
                },
                py::arg("stream") = NULL, "Export the buffer as a DLPack tensor")
            .def(
                "__dlpack_device__",
                [](std::shared_ptr<DecodedFrameExt>& self) {
                    // DLDevice ctx;
                    // ctx.device_type = DLDeviceType::kDLCUDA;
                    // ctx.device_id = 0;
                    return py::make_tuple(py::int_(static_cast<int>(DLDeviceType::kDLCUDA)),
                                          py::int_(static_cast<int>(0)));
                },
                "Get the device associated with the buffer")

            .def(
                "GetPtrToPlane",

                [](std::shared_ptr<DecodedFrameExt>& self, int planeIdx) {
                    return self->views[planeIdx].data;
                },
                R"pbdoc(
            return pointer to base address for plane index
            :param planeIdx : index to the plane
            )pbdoc");
    }
};
