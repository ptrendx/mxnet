/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2020 by Contributors
 * \file cuda_rtc.h
 * \brief Common CUDA utilities for
 *        runtime compilation.
 */

#ifndef MXNET_COMMON_CUDA_RTC_H_
#define MXNET_COMMON_CUDA_RTC_H_

#include "mxnet/base.h"
#include "mxnet/op_attr_types.h"

#if MXNET_USE_CUDA

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <mutex>
#include <string>
#include <vector>

namespace mxnet {
namespace common {
namespace cuda {
namespace rtc {

namespace util {

std::string to_string(OpReqType req);

}  // namespace util

extern std::mutex lock;

CUfunction get_function(const std::string &code,
                        const std::string &kernel_name,
                        int dev_id);

void launch(CUfunction function,
            const dim3 grid_dim,
            const dim3 block_dim,
            unsigned int shared_mem_bytes,
            mshadow::Stream<gpu> *stream,
            std::vector<const void*> *args);

}  // namespace rtc
}  // namespace cuda
}  // namespace common
}  // namespace mxnet

#endif  // MXNET_USE_CUDA

#endif  // MXNET_COMMON_CUDA_RTC_H_
