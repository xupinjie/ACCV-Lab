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
#include "NvCodecUtils.h"
#include "nvEncodeAPI.h"
#include <cuda.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
using namespace std;
// using namespace chrono;

namespace py = pybind11;

/**
 * @brief providing seek functionality within demuxer.
 */

#define ThrowOnCudaError_STRINGIFY(s) ThrowOnCudaError_STRINGIFY_(s)
#define ThrowOnCudaError_STRINGIFY_(s) #s
#define ThrowOnCudaError(expr)                                                       \
    {                                                                                \
        auto res = (expr);                                                           \
        if (CUDA_SUCCESS != res) {                                                   \
            std::stringstream ss;                                                    \
            ss << __FILE__ << ":";                                                   \
            ss << __LINE__ << std::endl;                                             \
            const char* errName = nullptr;                                           \
            if (CUDA_SUCCESS != cuGetErrorName(res, &errName)) {                     \
                ss << "CUDA error with code " << res << std::endl;                   \
            } else {                                                                 \
                ss << "CUDA error: " << errName << std::endl;                        \
            }                                                                        \
            const char* errDesc = nullptr;                                           \
            cuGetErrorString(res, &errDesc);                                         \
            if (!errDesc) {                                                          \
                ss << "No error string available" << std::endl;                      \
            } else {                                                                 \
                ss << errDesc << std::endl;                                          \
            }                                                                        \
            ss << "while executing: " ThrowOnCudaError_STRINGIFY(expr) << std::endl; \
            throw std::runtime_error(ss.str());                                      \
        }                                                                            \
    }

namespace {

class CuCtxGuard {
    CUcontext m_ctx;

   public:
    CuCtxGuard(CUcontext ctx) : m_ctx(ctx) { cuCtxPushCurrent_v2(ctx); }

    ~CuCtxGuard() { cuCtxPopCurrent(&m_ctx); }
};
}  // namespace

#define ENUM_VALUE_STRINGIFY(s) ENUM_VALUE_STRINGIFY_(s)
#define ENUM_VALUE_STRINGIFY_(s) #s
#define ENUM_VALUE(prefix, s) value(ENUM_VALUE_STRINGIFY(s), prefix##_##s)
#define DEF_CONSTANT(s) attr(ENUM_VALUE_STRINGIFY(s)) = py::cast(s)
#define DEF_READWRITE(type, s) def_readwrite(ENUM_VALUE_STRINGIFY(s), &type::s)

struct CAIMemoryView {
    std::vector<size_t> shape;
    std::vector<size_t> stride;
    std::string typestr;
    CUstream stream = nullptr;
    CUdeviceptr data;
    bool readOnly;

    CAIMemoryView(const std::vector<size_t>& _shape, const std::vector<size_t>& _stride,
                  const std::string& _typeStr, size_t _streamid, CUdeviceptr _data, bool _readOnly) {
        shape = _shape;
        stride = _stride;
        typestr = _typeStr;
        data = _data;
        readOnly = _readOnly;
        stream = reinterpret_cast<CUstream>(_streamid);
    }
    CAIMemoryView() {
        shape = {0};
        stride = {1};
        typestr = "|u1";
        data = reinterpret_cast<CUdeviceptr>(nullptr);
        readOnly = true;
        stream = (CUstream)2;
    }
    static void Export(py::module& m) {
        py::class_<CAIMemoryView, std::shared_ptr<CAIMemoryView>>(m, "CAIMemoryView", py::module_local())
            .def(py::init<std::vector<size_t>, std::vector<size_t>, std::string, size_t, CUdeviceptr, bool>())
            .def_readonly("shape", &CAIMemoryView::shape)
            .def_readonly("stride", &CAIMemoryView::stride)
            .def_readonly("dataptr", &CAIMemoryView::data)
            .def("__repr__",
                 [](std::shared_ptr<CAIMemoryView>& self) {
                     std::stringstream ss;
                     ss << "<CAIMemoryView ";
                     ss << py::str(py::cast(self->shape));
                     ss << ">";
                     return ss.str();
                 })
            .def_readonly("data", &CAIMemoryView::data)
            .def_property_readonly("__cuda_array_interface__", [](std::shared_ptr<CAIMemoryView>& self) {
                py::dict dict;
                dict["version"] = 3;
                dict["shape"] = self->shape;
                dict["strides"] = self->stride;
                dict["typestr"] = self->typestr;
                // stream field per CAI v3 spec:
                // https://nvidia.github.io/numba-cuda/user/cuda_array_interface.html#python-interface-specification
                //   None  → data already ready, no sync needed
                //   1 / 2 → legacy / per-thread default stream
                //   other → cudaStream_t handle cast to Python int; consumer must sync before use
                dict["stream"] =
                    self->stream ? py::cast(reinterpret_cast<size_t>(self->stream)) : py::cast(2);
                dict["data"] = std::make_pair(self->data, false);
                dict["gpuIdx"] = 0;  // TODO
                return dict;
            });
    }
};

CAIMemoryView coerceToCudaArrayView(py::object cuda_array, NV_ENC_BUFFER_FORMAT bufferFormat, size_t width,
                                    size_t height, int planeIdx = 0);
