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
 *  Copyright (c) 2019 by Contributors
 * \file mrcnn_mask_target.cu
 * \brief Mask-RCNN target generator
 * \author Serge Panev
 */

#include "./mrcnn_mask_target-inl.h"
#include <cub/cub.cuh>

namespace mxnet {
namespace op {

using namespace mshadow::cuda;

// The maximum number of blocks to use in the default kernel call.
constexpr int MAXIMUM_NUM_BLOCKS = 4096;

inline int CUDA_GET_BLOCKS(const int N) {
  return std::max(
      std::min(
          (N + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock,
          MAXIMUM_NUM_BLOCKS),
      // Use at least 1 block, since CUDA does not allow empty block.
      1);
}

// Kernels

/*rle cuda kernels are cuda version of the corresponding cpu functions here
https://github.com/cocodataset/cocoapi/blob/master/common/maskApi.c
these are only a subset of rle kernels.*/

typedef unsigned int uint;

// 6144 is based on minimum shared memory size per SM
// across all pytorch-supported GPUs. Need to use blocking
// to avoid this restriction
const int SH_BUFFER_SIZE = 6144;
const int BUFFER_SIZE = 30144;
const int CNTS_SIZE = 6144;


// Batch size upgrade: Done
template <typename DType>
__global__ void crop_and_scale_cuda_kernel(double *dense_poly_data, const int *per_roi_poly_idx, const int *poly_rel_idx,
                                           int poly_count, int roi_count, DType *roi_data, int mask_size) {
    int tid = threadIdx.x;
    int block_jump = blockDim.x;
    int poly_id = blockIdx.x;
    int roi_idx;
    for (roi_idx = 0; roi_idx < roi_count; roi_idx++){
        if (poly_id < per_roi_poly_idx[roi_idx + 1]) break;
    }
    DType *roi = roi_data + (roi_idx * 4);
    DType w = roi[2] - roi[0];
    DType h = roi[3] - roi[1];
  	w = fmaxf(w, 1.0f);
  	h = fmaxf(h, 1.0f);
    DType ratio_h = ((DType) mask_size) / h;
    DType ratio_w = ((DType) mask_size) / w;

    int poly_ptr_idx_start = poly_rel_idx[poly_id];
    int poly_ptr_idx_end = poly_rel_idx[poly_id + 1];

    double *poly_data_buf = dense_poly_data + poly_ptr_idx_start;
    int len = poly_ptr_idx_end - poly_ptr_idx_start;

    for (int j = tid; j < len; j += block_jump) {
        if (j % 2 == 0) poly_data_buf[j] = ratio_w*((DType) poly_data_buf[j]- roi[0]);
        if (j % 2 == 1) poly_data_buf[j] = ratio_h*((DType) poly_data_buf[j]- roi[1]);
    }
}


// Batch size upgrade: No need, per polygon
/*cuda version of rleFrPoly function in this API:
https://github.com/cocodataset/cocoapi/blob/master/common/maskApi.c*/

#if 0
__global__ void old_rle_fr_poly_cuda_kernel(const double *dense_coordinates, int *poly_rel_idx, long h, long w,
                                        uint *cnts, int *x_in, int *y_in, int *u_in, int *v_in, uint *a_in,
                                        uint *b_in, int *num_of_cnts) {
    int poly_id = blockIdx.x;
    int tid = threadIdx.x;
    int block_jump = blockDim.x;
    long cnts_offset = poly_id * CNTS_SIZE;
    long k = (poly_rel_idx[poly_id + 1] - poly_rel_idx[poly_id]) / 2;

    const double *xy = dense_coordinates + poly_rel_idx[poly_id];
    int *x = x_in + poly_id * BUFFER_SIZE;
    int *y = y_in + poly_id * BUFFER_SIZE;
    int *u = u_in + poly_id * BUFFER_SIZE;
    int *v = v_in + poly_id * BUFFER_SIZE;
    uint *a = a_in + poly_id * BUFFER_SIZE;
    uint *b = b_in + poly_id * BUFFER_SIZE;
    /* upsample and get discrete points densely along entire boundary */
    long j, m = 0;
    double scale = 5;
    __shared__ int shbuf1[BUFFER_SIZE];
    __shared__ int shbuf2[BUFFER_SIZE];
    for(long j = tid; j < BUFFER_SIZE; j += block_jump) {
        shbuf1[j] = 0;
        shbuf2[j] = 0;
    }
    for(long j = tid; j <= k; j += block_jump)
        x[j] = j < k ? ((int) (scale * xy[2 * j + 0] + 0.5)) : ((int) (scale * xy[0] + 0.5));
    for(long j = tid; j <= k; j += block_jump)
        y[j] = j < k ? ((int) (scale * xy[2 * j + 1] + 0.5)) : ((int) (scale * xy[1] + 0.5));
    __syncthreads();

    for(int j = tid; j < k; j += block_jump){
        int xs = x[j], xe = x[j + 1], ys = y[j], ye = y[j + 1], dx, dy, t, dist;
        int flip;
        double s;
        dx = abs(xe - xs);
        dy = abs(ys - ye);
        flip = (dx >= dy && xs > xe) || (dx < dy && ys > ye);
        if (flip) {t = xs; xs = xe; xe = t; t = ys; ys = ye; ye = t;}
        s = dx >= dy ? (double) (ye - ys) / dx : (double) (xe - xs) / dy;
        dist = dx >= dy ? dx + 1 : dy + 1;
        shbuf1[j + 1] = dist;
        shbuf2[j + 1] = dist;
    }
    __syncthreads();
    //block-wide exclusive prefix scan
    int switch_buf = 0;
    for (int offset = 1; offset <= k; offset *= 2) {
        switch_buf = 1 - switch_buf;
        if (switch_buf == 0){
            for(int j = tid; j <= k; j += block_jump){
                if (j >= offset) shbuf2[j] = shbuf1[j] + shbuf1[j - offset];
                else shbuf2[j] = shbuf1[j];
            }
        }
        else if (switch_buf == 1){
            for(int j = tid; j <= k; j += block_jump){
                if (j >= offset) shbuf1[j] = shbuf2[j] + shbuf2[j - offset];
                else shbuf1[j] = shbuf2[j];
            }
        }
        __syncthreads();
    }

    for (int j = tid; j < k; j += block_jump){
        int xs = x[j], xe = x[j + 1], ys = y[j], ye = y[j + 1], dx, dy, t, d;
        int flip;
        double s;
        dx = __sad(xe, xs, 0);
        dy = __sad(ys, ye, 0);
        flip = (dx >= dy && xs > xe) || (dx < dy && ys > ye);
        if (flip) {t = xs; xs = xe; xe = t; t = ys; ys = ye; ye = t;}
        s = dx >= dy ? (double) (ye - ys) / dx : (double) (xe - xs) / dy;
        m = switch_buf == 0 ? shbuf2[j] : shbuf1[j];
        if (dx >= dy) for (d = 0; d <= dx; d++) {
          /*the multiplication statement 's*t' causes nvcc to optimize with flush-to-zero=True for
          double precision multiply, which we observe produces different results than CPU occasionally.
          To force flush-to-zero=False, we use __dmul_rn intrinsics function */
          t = flip ? dx - d : d;
          u[m] = t + xs;
          v[m] = (int) (ys + __dmul_rn(s, t) + .5);
          m++;
        }
        else for (d = 0; d <= dy; d++) {
          t = flip ? dy - d : d;
          v[m] = t + ys;
          u[m] = (int) (xs + __dmul_rn(s, t) + .5);
          m++;
        }
    }
    __syncthreads();
    m = switch_buf == 0 ? shbuf2[k] : shbuf1[k];
    int k2 = m;
    __syncthreads();
    double xd, yd;
    if (tid == 0) {
        shbuf1[tid] = 0;
        shbuf2[tid] = 0;
    }
    
    /* get points along y-boundary and downsample */
    for (int j = tid; j < k2; j += block_jump) {
        if (j > 0) {
            if (u[j] != u[j - 1]){
                xd = (double) (u[j] < u[j-1] ? u[j] : u[j] - 1);
                xd = (xd + .5) / scale - .5;
                if (floor(xd) != xd || xd < 0 || xd > w - 1 ) {
                    shbuf1[j] = 0;
                    shbuf2[j] = 0;
                    continue;
                }
                yd = (double) (v[j] < v[j - 1] ? v[j] : v[j - 1]); yd = (yd + .5) / scale - .5;
                if (yd < 0) yd = 0;
                else if (yd > h) yd = h; yd = ceil(yd);
                shbuf1[j] = 1;
                shbuf2[j] = 1;
            } else {
                shbuf1[j] = 0;
                shbuf2[j] = 0;
            }
        }
    }
    __syncthreads();
    //exclusive prefix scan
    switch_buf = 0;
    for (int offset = 1; offset < k2; offset *= 2){
        switch_buf = 1 - switch_buf;
        if (switch_buf == 0){
            for (int j = tid; j < k2; j += block_jump){
                if (j >= offset) shbuf2[j] = shbuf1[j - offset] + shbuf1[j];
                else shbuf2[j] = shbuf1[j];
            }
        }
        else if (switch_buf == 1){
            for (int j = tid; j < k2; j += block_jump){
                if (j >= offset) shbuf1[j] = shbuf2[j - offset] + shbuf2[j];
                else shbuf1[j] = shbuf2[j];
            }
        }
        __syncthreads();
    }

    for (int j = tid; j < k2; j += block_jump){
        if (j > 0){
            if(u[j] != u[j - 1]){
                xd = (double) (u[j] < u[j - 1] ? u[j] : u[j] - 1);
                xd = (xd + .5) / scale - .5;
                if (floor(xd) != xd || xd < 0 || xd > w - 1) {continue;}
                yd = (double) (v[j] < v[j - 1] ? v[j] : v[j - 1]);
                yd = (yd + .5) / scale - .5;
                if (yd < 0) yd = 0;
                else if (yd > h) yd = h; yd = ceil(yd);
                m = switch_buf == 0 ? shbuf2[j - 1]:shbuf1[j - 1];
                x[m] = (int) xd;
                y[m] = (int) yd;
                m++;
            }
        }
    }
    __syncthreads();

    /* compute rle encoding given y-boundary points */
    m = switch_buf == 0 ? shbuf2[k2 - 1] : shbuf1[k2 - 1];
    int k3 = m;
    for (int j = tid; j <= k3; j += block_jump){
       if (j < k3) a[j] = (uint) (x[j] * (int) (h) + y[j]);
       else a[j] = (uint)(h * w);
    }
    k3++;
    __syncthreads();

    //run brick sort on a for k3+1 element
    //load k3+1 elements of a into shared memory
    for(long j = tid; j < k3; j += block_jump) shbuf1[j]=a[j];
    __syncthreads();
    uint a_temp;
    for (int r = 0; r <= k3 / 2; r++){
        int evenCas = k3 / 2;
        int oddCas = (k3 - 1) / 2;
        //start with 0, need (k3+1)/2 CAS
        for (int j = tid; j < evenCas; j += block_jump){
            if (shbuf1[2 * j] > shbuf1[2 * j + 1]){
                a_temp = shbuf1[2 * j];
                shbuf1[2 * j]=shbuf1[2 * j + 1];
                shbuf1[2 * j + 1] = a_temp;
            }
        }
        __syncthreads();
        //start with 1
        for (int j = tid; j < oddCas; j += block_jump){
            if (shbuf1[2 * j + 1] > shbuf1[2 * j + 2]){
                a_temp=shbuf1[2 * j + 1];
                shbuf1[2 * j + 1] = shbuf1[2 * j + 2];
                shbuf1[2 * j + 2]=a_temp;
            }
        }
        __syncthreads();
    }

    for(long j = tid; j < k3; j += block_jump) {
        if(j>0) shbuf2[j] = shbuf1[j - 1];
        else shbuf2[j] = 0;
    }
     __syncthreads();
    for(int j = tid; j < k3; j += block_jump){
        shbuf1[j] -= shbuf2[j];
    }
    __syncthreads();
    // writing to output
    uint *cnts_buf = cnts + cnts_offset;
    if (tid == 0){
        j = m = 0;
        cnts_buf[m++] = shbuf1[j++];
        while (j < k3) if (shbuf1[j] > 0) cnts_buf[m++] = shbuf1[j++]; else {
            j++; if (j < k3) cnts_buf[m - 1] += shbuf1[j++]; }
        num_of_cnts[poly_id] = m;
    }
    __syncthreads();
}
#endif



// Batch size upgrade: No need, per polygon
/*cuda version of rleFrPoly function in this API:
https://github.com/cocodataset/cocoapi/blob/master/common/maskApi.c*/

__global__ void rle_fr_poly_cuda_kernel(const double *dense_coordinates, int *poly_rel_idx, long h, long w,
                                        uint *cnts, int *x_in, int *y_in, int *u_in, int *v_in, uint *a_in,
                                        uint *b_in, int *num_of_cnts) {
    int poly_id = blockIdx.x;
    int tid = threadIdx.x;
    int block_jump = blockDim.x;
    long cnts_offset = poly_id * CNTS_SIZE;
    long k = (poly_rel_idx[poly_id + 1] - poly_rel_idx[poly_id]) / 2;

    const double *xy = dense_coordinates + poly_rel_idx[poly_id];
    int *x = x_in + poly_id * BUFFER_SIZE;
    int *y = y_in + poly_id * BUFFER_SIZE;
    int *u = u_in + poly_id * BUFFER_SIZE;
    int *v = v_in + poly_id * BUFFER_SIZE;
    uint *a = a_in + poly_id * BUFFER_SIZE;
    uint *b = b_in + poly_id * BUFFER_SIZE;
    /* upsample and get discrete points densely along entire boundary */
    long j, m = 0;
    double scale = 5;
    __shared__ int shbuf1[SH_BUFFER_SIZE];
    __shared__ int shbuf2[SH_BUFFER_SIZE];
    for(long j = tid; j < SH_BUFFER_SIZE; j += block_jump) {
        shbuf1[j] = 0;
        shbuf2[j] = 0;
    }
    for(long j = tid; j <= k; j += block_jump)
        x[j] = j < k ? ((int) (scale * xy[2 * j + 0] + 0.5)) : ((int) (scale * xy[0] + 0.5));
    for(long j = tid; j <= k; j += block_jump)
        y[j] = j < k ? ((int) (scale * xy[2 * j + 1] + 0.5)) : ((int) (scale * xy[1] + 0.5));
    __syncthreads();

    for(int j = tid; j < k; j += block_jump){
        int xs = x[j], xe = x[j + 1], ys = y[j], ye = y[j + 1], dx, dy, t, dist;
        int flip;
        double s;
        dx = abs(xe - xs);
        dy = abs(ys - ye);
        flip = (dx >= dy && xs > xe) || (dx < dy && ys > ye);
        if (flip) {t = xs; xs = xe; xe = t; t = ys; ys = ye; ye = t;}
        s = dx >= dy ? (double) (ye - ys) / dx : (double) (xe - xs) / dy;
        dist = dx >= dy ? dx + 1 : dy + 1;
        shbuf1[j + 1] = dist;
        shbuf2[j + 1] = dist;
    }
    __syncthreads();
    //block-wide exclusive prefix scan
    int switch_buf = 0;
    for (int offset = 1; offset <= k; offset *= 2){
        switch_buf = 1 - switch_buf;
        if (switch_buf == 0){
            for(int j = tid; j <= k; j += block_jump){
                if (j >= offset) shbuf2[j] = shbuf1[j] + shbuf1[j - offset];
                else shbuf2[j] = shbuf1[j];
            }
        }
        else if (switch_buf == 1){
            for(int j = tid; j <= k; j += block_jump){
                if (j >= offset) shbuf1[j] = shbuf2[j] + shbuf2[j - offset];
                else shbuf1[j] = shbuf2[j];
            }
        }
        __syncthreads();
    }

    for (int j = tid; j < k; j += block_jump){
        int xs = x[j], xe = x[j + 1], ys = y[j], ye = y[j + 1], dx, dy, t, d;
        int flip;
        double s;
        dx = __sad(xe, xs, 0);
        dy = __sad(ys, ye, 0);
        flip = (dx >= dy && xs > xe) || (dx < dy && ys > ye);
        if (flip) {t = xs; xs = xe; xe = t; t = ys; ys = ye; ye = t;}
        s = dx >= dy ? (double) (ye - ys) / dx : (double) (xe - xs) / dy;
        m = switch_buf == 0 ? shbuf2[j] : shbuf1[j];
        if (dx >= dy) for (d = 0; d <= dx; d++) {
          /*the multiplication statement 's*t' causes nvcc to optimize with flush-to-zero=True for
          double precision multiply, which we observe produces different results than CPU occasionally.
          To force flush-to-zero=False, we use __dmul_rn intrinsics function */
          t = flip ? dx - d : d;
          u[m] = t + xs;
          v[m] = (int) (ys + __dmul_rn(s, t) + .5);
          m++;
        }
        else for (d = 0; d <= dy; d++) {
          t = flip ? dy - d : d;
          v[m] = t + ys;
          u[m] = (int) (xs + __dmul_rn(s, t) + .5);
          m++;
        }
    }
    __syncthreads();
    m = switch_buf == 0 ? shbuf2[k] : shbuf1[k];
    int k2 = m;
    __syncthreads();
    double xd, yd;
    if (tid == 0) {
        shbuf1[tid] = 0;
        shbuf2[tid] = 0;
    }
    

    /*************************************************
    *** get points along y-boundary and downsample ***
    *************************************************/

    // We need to use blocking as k2 might be > shmem size
    int previous_block_sum = 0;
    int current_tid = tid;
    int current_k2 = 0;
    for (int current_k2_offset = 0; current_k2_offset < k2; current_k2_offset += SH_BUFFER_SIZE) {
      
      current_k2 = (k2 - current_k2_offset) > SH_BUFFER_SIZE ? SH_BUFFER_SIZE : k2;

      for (int j = current_tid; j < current_k2_offset + current_k2; j += block_jump) {
          if (j > 0) {
              int shbuf_j = j - current_k2_offset;
              if (u[j] != u[j - 1]){
                  xd = (double) (u[j] < u[j-1] ? u[j] : u[j] - 1);
                  xd = (xd + .5) / scale - .5;
                  if (floor(xd) != xd || xd < 0 || xd > w - 1 ) {
                      shbuf1[shbuf_j] = 0;
                      shbuf2[shbuf_j] = 0;
                      continue;
                  }
                  yd = (double) (v[j] < v[j - 1] ? v[j] : v[j - 1]); yd = (yd + .5) / scale - .5;
                  if (yd < 0) yd = 0;
                  else if (yd > h) yd = h; yd = ceil(yd);
                  shbuf1[shbuf_j] = 1;
                  shbuf2[shbuf_j] = 1;
              } else {
                  shbuf1[shbuf_j] = 0;
                  shbuf2[shbuf_j] = 0;
              }
          }
      }
      __syncthreads();
      //exclusive prefix scan
      if (tid == 0) {
        shbuf1[0] += previous_block_sum;
        shbuf2[0] += previous_block_sum;
      }
      __syncthreads();
      switch_buf = 0;
      for (int offset = 1; offset < current_k2; offset *= 2) {
          switch_buf = 1 - switch_buf;
          if (switch_buf == 0){
              for (int j = tid; j < current_k2; j += block_jump){
                  if (j >= offset) shbuf2[j] = shbuf1[j - offset] + shbuf1[j];
                  else shbuf2[j] = shbuf1[j];
              }
          }
          else if (switch_buf == 1){
              for (int j = tid; j < current_k2; j += block_jump){
                  if (j >= offset) shbuf1[j] = shbuf2[j - offset] + shbuf2[j];
                  else shbuf1[j] = shbuf2[j];
              }
          }
          __syncthreads();
      }

      if (tid == 0) {
        previous_block_sum = switch_buf == 0 ? shbuf2[current_k2 - 1] : shbuf1[current_k2 - 1];
      }
      __syncthreads();

      int j = 0;
      for (j = current_tid; j < current_k2_offset + current_k2; j += block_jump){
        int shbuf_j = j - current_k2_offset;
        if (j > 0){
            if(u[j] != u[j - 1]){
                xd = (double) (u[j] < u[j - 1] ? u[j] : u[j] - 1);
                xd = (xd + .5) / scale - .5;
                if (floor(xd) != xd || xd < 0 || xd > w - 1) {continue;}
                yd = (double) (v[j] < v[j - 1] ? v[j] : v[j - 1]);
                yd = (yd + .5) / scale - .5;
                if (yd < 0) yd = 0;
                else if (yd > h) yd = h; yd = ceil(yd);
                m = switch_buf == 0 ? shbuf2[shbuf_j - 1]:shbuf1[shbuf_j   - 1];
                x[m] = (int) xd;
                y[m] = (int) yd;
                m++;
            }
        }
      }
      current_tid = j;
      __syncthreads();
    }

    /*******************************************************
    ***** compute rle encoding given y-boundary points *****
    *******************************************************/

    m = switch_buf == 0 ? shbuf2[current_k2 - 1] : shbuf1[current_k2 - 1];
    int k3 = m;
    for (int j = tid; j <= k3; j += block_jump){
       if (j < k3) a[j] = (uint) (x[j] * (int) (h) + y[j]);
       else a[j] = (uint)(h * w);
    }
    k3++;
    __syncthreads();


    /*

    //run brick sort on a for k3+1 element
    //load k3+1 elements of a into shared memory
    for(long j = tid; j < k3; j += block_jump) shbuf1[j]=a[j];
    __syncthreads();
    uint a_temp;
    for (int r = 0; r <= k3 / 2; r++){
        int evenCas = k3 / 2;
        int oddCas = (k3 - 1) / 2;
        //start with 0, need (k3+1)/2 CAS
        for (int j = tid; j < evenCas; j += block_jump){
            if (shbuf1[2 * j] > shbuf1[2 * j + 1]){
                a_temp = shbuf1[2 * j];
                shbuf1[2 * j]=shbuf1[2 * j + 1];
                shbuf1[2 * j + 1] = a_temp;
            }
        }
        __syncthreads();
        //start with 1
        for (int j = tid; j < oddCas; j += block_jump){
            if (shbuf1[2 * j + 1] > shbuf1[2 * j + 2]){
                a_temp=shbuf1[2 * j + 1];
                shbuf1[2 * j + 1] = shbuf1[2 * j + 2];
                shbuf1[2 * j + 2]=a_temp;
            }
        }
        __syncthreads();
    }

    for(long j = tid; j < k3; j += block_jump) {
        if(j>0) shbuf2[j] = shbuf1[j - 1];
        else shbuf2[j] = 0;
    }
     __syncthreads();
    for(int j = tid; j < k3; j += block_jump) {
        shbuf1[j] -= shbuf2[j];
    }

    uint *cnts_buf = cnts + cnts_offset;
    if (tid == 0){
        j = m = 0;
        cnts_buf[m++] = shbuf1[j++];
        while (j < k3) if (shbuf1[j] > 0) cnts_buf[m++] = shbuf1[j++]; else {
            j++; if (j < k3) cnts_buf[m - 1] += shbuf1[j++]; }
        num_of_cnts[poly_id] = m;
    }
    __syncthreads();
    */

    // TODO: implement shmem sort
    uint a_temp;
    for (int r = 0; r <= k3 / 2; r++){
      int evenCas = k3 / 2;
      int oddCas = (k3 - 1) / 2;
      //start with 0, need (k3+1)/2 CAS
      for (int j = tid; j < evenCas; j += block_jump){
          if (a[2 * j] > a[2 * j + 1]){
              a_temp = a[2 * j];
              a[2 * j]=a[2 * j + 1];
              a[2 * j + 1] = a_temp;
          }
      }
      __syncthreads();
      //start with 1
      for (int j = tid; j < oddCas; j += block_jump){
          if (a[2 * j + 1] > a[2* j + 2]){
              a_temp=a[2 * j + 1];
              a[2 * j + 1] = a[2 * j + 2];
              a[2 * j + 2]=a_temp;
          }
      }
      __syncthreads();
    }

    for(long j = tid; j < k3; j += block_jump) {
      if(j>0) b[j] = a[j - 1];
      else b[j] = 0;
    }

    __syncthreads();
    for(int j = tid; j < k3; j += block_jump) {
      a[j] -= b[j];
    }

    __syncthreads();
    /**************************************
    ********* writing to output ***********
    **************************************/
    uint *cnts_buf = cnts + cnts_offset;
    if (tid == 0){
        j = m = 0;
        cnts_buf[m++] = a[j++];
        while (j < k3) if (a[j] > 0) cnts_buf[m++] = a[j++]; else {
            j++; if (j < k3) cnts_buf[m - 1] += a[j++]; }
        num_of_cnts[poly_id] = m;
    }
    __syncthreads();
}


// Batch size upgrade: No need, per polygon
/*cuda version of rleDecode function in this API:
https://github.com/cocodataset/cocoapi/blob/master/common/maskApi.c*/
__global__ void decode_rle_cuda_kernel(const int *num_of_cnts, uint *cnts, long h, long w, unsigned char *mask)
{
    int poly_id = blockIdx.x;
    int tid = threadIdx.x;
    int block_jump = blockDim.x;

    int m = num_of_cnts[poly_id];
    uint *cnts_buf = cnts + CNTS_SIZE * poly_id;
    unsigned char *mask_ptr = mask + poly_id * h * w;

    __shared__ uint shbuf1[CNTS_SIZE];
    __shared__ uint shbuf2[CNTS_SIZE];

    //initialize shbuf for scan. first element is 0 (exclusive scan)
    for (long i = tid; i < CNTS_SIZE; i += block_jump){
        shbuf1[i] = (i <= m & i > 0) ? cnts_buf[i - 1]:0;
        shbuf2[i] = (i <= m & i > 0) ? cnts_buf[i - 1]:0;
    }
    __syncthreads();

    //double buffering for scan
    int switch_buf = 0;
    for (int offset = 1; offset <= m; offset *= 2){
        switch_buf = 1 - switch_buf;
        if(switch_buf == 0){
            for(int j = tid;j <= m;j += block_jump){
                if(j >= offset) shbuf2[j] = shbuf1[j]+shbuf1[j - offset];
                else shbuf2[j] = shbuf1[j];
            }
        }else if (switch_buf == 1){
            for(int j = tid;j <= m;j += block_jump){
                if(j >= offset) shbuf1[j] = shbuf2[j] + shbuf2[j - offset];
                else shbuf1[j] = shbuf2[j];
            }

        }
        __syncthreads();
    }
    uint *scanned_buf = switch_buf == 0 ? shbuf2 : shbuf1;

    //find which bin pixel j falls into , which determines the pixel value
    //use binary search
    for(int j = tid; j < h * w; j += block_jump){
        int min_idx = 0;
        int max_idx = m;
        int mid_idx = m / 2;
        while(max_idx > min_idx){
            if(j > scanned_buf[mid_idx]) {
                min_idx = mid_idx+1;
                mid_idx = (min_idx + max_idx) / 2;
            }
            else if (j < scanned_buf[mid_idx]) {
                max_idx = mid_idx;
                mid_idx = (min_idx + max_idx) / 2;
            }
            else {
                mid_idx++;
                break;
            }
        }
        int k = mid_idx;
        unsigned char pixel = k % 2 == 0 ? 1 : 0;
        mask_ptr[j] = pixel;
    }
}


// Batch size upgrade: Done
// merging masks happens on mask format, not RLE format.
template<typename DType>
__global__ void merge_masks_cuda_kernel(unsigned char *masks_in, DType *masks_out, const int mask_size,
                                        int *per_roi_poly_idx, const DType *cls_targets,
                                        int num_classes) {

    int roi_idx = blockIdx.x / num_classes;
    int class_idx = blockIdx.x % num_classes;
    int tid = threadIdx.x;
    int jump_block = blockDim.x;
    int mask_start_idx = per_roi_poly_idx[roi_idx];
    int num_of_masks_to_merge = per_roi_poly_idx[roi_idx + 1] - per_roi_poly_idx[roi_idx];

    int mask_offset = (roi_idx * num_classes + class_idx) * mask_size * mask_size;

    for(int j = tid; j < mask_size * mask_size; j += jump_block){
        int transposed_pixel = (j % mask_size) * mask_size + j / mask_size;
        unsigned char pixel = 0;
        if (class_idx == cls_targets[roi_idx]) {
            for(int k = 0; k < num_of_masks_to_merge; k++){
                if (masks_in[(mask_start_idx + k) * mask_size * mask_size + j] == 1)
                  pixel = 1;
                if (pixel == 1) break;
            }
        }
        masks_out[mask_offset + transposed_pixel] = (DType)pixel;
    }
}


template <typename T>
__device__ T bilinear_interpolate(
    const T* in_data,
    const int height,
    const int width,
    T y,
    T x,
    const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    return 0;
  }

  if (y <= 0) {
    y = 0;
  }
  if (x <= 0) {
    x = 0;
  }

  int y_low = static_cast<int>(y);
  int x_low = static_cast<int>(x);
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  T v1 = in_data[y_low * width + x_low];
  T v2 = in_data[y_low * width + x_high];
  T v3 = in_data[y_high * width + x_low];
  T v4 = in_data[y_high * width + x_high];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

template<typename DType>
__global__ void MRCNNMaskTargetKernel(const DType *rois,
                                      const DType *matches,
                                      const DType *cls_targets,
                                      DType* sampled_masks,
                                      DType* mask_cls,
                                      const int total_out_el,
                                      int batch_size,
                                      int num_classes,
                                      int num_rois,
                                      int mask_size_h,
                                      int mask_size_w,
                                      int sample_ratio) {

  // computing mask_cls
  int num_masks = batch_size * num_rois * num_classes;
  int mask_vol = mask_size_h * mask_size_w;
  for (int mask_idx = blockIdx.x; mask_idx < num_masks; mask_idx += gridDim.x) {
    int cls_idx = mask_idx % num_classes;
    int roi_idx = (mask_idx / num_classes) % num_rois;
    int batch_idx = (mask_idx / num_classes / num_rois);

    DType* mask_cls_out = mask_cls + mask_idx * mask_vol;

    DType cls_target = cls_targets[batch_idx * num_rois + roi_idx];
    DType out_val = (cls_target == cls_idx);
    for (int mask_pixel = threadIdx.x; mask_pixel < mask_vol; mask_pixel += blockDim.x) {
      mask_cls_out[mask_pixel] = out_val;
    }
  }
}

template <typename xpu, typename DType>
mshadow::Tensor<xpu, 1, DType> get_1d_tensor(int size,
                                             mshadow::Stream<xpu> *stream,
                                             const OpContext &ctx) {
  return ctx.requested[mrcnn_index::kTempSpace]
          .get_space_typed<xpu, 1, DType>(mshadow::Shape1(size), stream);
}

constexpr int NTHREADS = 256;
constexpr int NEDGES = NTHREADS;
constexpr int THREADS_PER_WARP = 32;

__device__ inline int binary_search(double x, double* table, const int N) {
  int min_idx = 0;
  int max_idx = N;
  int mid_idx = N / 2;
  /*if (blockIdx.x == 0 && threadIdx.x == 32) {*/
    /*printf("HAHA %f\n", x);*/
  /*}*/
  while(max_idx > min_idx){
      if(x > table[mid_idx]) {
          min_idx = mid_idx+1;
  /*if (blockIdx.x == 0 && threadIdx.x == 32) {*/
    /*printf("HOHO %d %d %d\n", min_idx, mid_idx, max_idx);*/
  /*}*/
      }
      else if (x < table[mid_idx]) {
          max_idx = mid_idx;
  /*if (blockIdx.x == 0 && threadIdx.x == 32) {*/
    /*printf("HIHI %d %d %d\n", min_idx, mid_idx, max_idx);*/
  /*}*/
      }
      else {
          mid_idx++;
          break;
      }
      mid_idx = (min_idx + max_idx) / 2;
  }
  /*if (blockIdx.x == 0 && threadIdx.x == 32) {*/
    /*printf("HLEHLE %d %d %d\n", min_idx, mid_idx, max_idx);*/
  /*}*/
  return mid_idx;
}

__global__ void rasterize_kernel(const double* dense_coordinates,
                                 const int* poly_rel_idx,
                                 const long h,
                                 const long w,
                                 unsigned char* mask) {
  const int poly_id = blockIdx.x;
  const int tid = threadIdx.x;
  long k = (poly_rel_idx[poly_id + 1] - poly_rel_idx[poly_id]) / 2;
  __shared__ double2 vertices[NEDGES];
  typedef cub::BlockRadixSort<double, NTHREADS, 1> cub_sort;
  constexpr int NWARPS = NTHREADS / THREADS_PER_WARP;
  __shared__ union {
    double intersections[NEDGES * NWARPS];
    typename cub_sort::TempStorage temp_storage;
  } scratch;

  double temp_intersections[NWARPS];

  assert(k < NEDGES);

  /*if (threadIdx.x == 0) {*/
    /*printf("%d k: %d %d %d\n", blockIdx.x, (int)k, poly_rel_idx[poly_id], poly_rel_idx[poly_id + 1]);*/
  /*}*/

  const double2 *xy = reinterpret_cast<const double2*>(dense_coordinates + poly_rel_idx[poly_id]);

  unsigned char* const mask_ptr = mask + poly_id * h * w;

  for (int i = tid; i < k; i += blockDim.x) {
    vertices[i] = xy[i];
  }
  __syncthreads();

  /*if (blockIdx.x == 0 && threadIdx.x == 0) {*/
    /*printf("[\n");*/
    /*for (int i = 0; i < k; ++i) {*/
      /*printf("[%f, %f],\n", vertices[i].x, vertices[i].y);*/
    /*}*/
    /*printf("]\n");*/
  /*}*/

  const int warp_id = tid / THREADS_PER_WARP;
  const int lane_id = tid % THREADS_PER_WARP;
  const double invalid_intersection = 2 * w;
  const long aligned_h = ((h + NWARPS - 1) / NWARPS) * NWARPS;
  for (int current_y = warp_id;
       current_y < aligned_h;
       current_y += NWARPS) {
    double my_y = current_y + 0.5;
    for (int edge = lane_id; edge < NEDGES; edge += THREADS_PER_WARP) {
      scratch.intersections[warp_id * NEDGES + edge] = invalid_intersection;
    }
    __syncthreads();
    if (current_y < h) {
      for (int edge = lane_id; edge < k; edge += THREADS_PER_WARP) {
        const int previous_edge = (edge - 1 + k) % k;
        const double2 vert1 = vertices[previous_edge];
        const double2 vert2 = vertices[edge];
        if (vert1.y != vert2.y) {
          double my_intersection = (my_y * (vert1.x - vert2.x) +
                                    vert2.x * vert1.y -
                                    vert1.x * vert2.y) /
                                   (vert1.y - vert2.y);
          if (my_intersection <= fmax(vert1.x, vert2.x) &&
              my_intersection >= fmin(vert1.x, vert2.x)) {
            scratch.intersections[warp_id * NEDGES + edge] = my_intersection;
          }
        }
      }
    }

    __syncthreads();

#pragma unroll
    for (int i = 0; i < NWARPS; ++i) {
      temp_intersections[i] = scratch.intersections[i * NEDGES + tid];
    }

    __syncthreads();

#pragma unroll
    for (int i = 0; i < NWARPS; ++i) {
      double temp[1] = { temp_intersections[i] };
      cub_sort(scratch.temp_storage).Sort(temp);
      temp_intersections[i] = temp[0];
    }

    __syncthreads();

#pragma unroll
    for (int i = 0; i < NWARPS; ++i) {
      scratch.intersections[i * NEDGES + tid] = temp_intersections[i];
    }

    __syncthreads();
    /*if (current_y == 1 && lane_id == 0 && blockIdx.x == 0) {*/
      /*printf("Intersections:\n");*/
      /*for (int i = 0; i < k; ++i) {*/
        /*printf("%d: %f\n", i, scratch.intersections[i]);*/
      /*}*/
      /*printf("End intersections\n");*/
    /*}*/

    if (current_y < h) {
      for (int x = lane_id; x < w; x += THREADS_PER_WARP) {
        const double my_x = x + 0.5;
        const int place = binary_search(my_x, scratch.intersections + NEDGES * warp_id, NEDGES);
        /*if (blockIdx.x == 0 && current_y == 1) {*/
          /*printf("%d: %f My binary search result is %d and I write %d to %d\n", x, my_x, place, place % 2, (int)(x + current_y * w));*/
        /*}*/
        mask_ptr[x + current_y * w] = place % 2;
      }
    }

    __syncthreads();
  }
}


template<>
void MRCNNMaskTargetRun<gpu>(const MRCNNMaskTargetParam& param, const std::vector<TBlob> &inputs,
                             const std::vector<TBlob> &outputs, const OpContext &ctx,
                             mshadow::Stream<gpu> *s) {
  const int block_dim_size = kMaxThreadsPerBlock;
  using namespace mxnet_op;
  using mshadow::Tensor;

  // only works with square mask targets
  const int M = param.mask_size[0];
  assert (M < 32);
  //if M >= 32, shared memory buffer size may not be
  //sufficient. Need to fix this by blocking

  // Fixed type inputs
  const auto polys_per_instance = inputs[mrcnn_index::kPolysPerInstance].FlatTo1D<gpu, int>(s);
  const auto poly_rel_idx = inputs[mrcnn_index::kPolyRelIdx].FlatTo1D<gpu, int>(s);
  auto dense_polys = inputs[mrcnn_index::kDensePolys].FlatTo2D<gpu, double>(s);

  int num_of_poly = poly_rel_idx.shape_[0] - 1;

  // Buffers
  Tensor<gpu, 1, int> d_xyuv_t = get_1d_tensor<gpu, int>(4 * BUFFER_SIZE * num_of_poly, s, ctx);
  Tensor<gpu, 1, uint> d_a_t = get_1d_tensor<gpu, uint>(BUFFER_SIZE * num_of_poly, s, ctx);
  Tensor<gpu, 1, uint> d_b_t = get_1d_tensor<gpu, uint>(BUFFER_SIZE * num_of_poly, s, ctx);
  Tensor<gpu, 1, unsigned char> d_mask_t = get_1d_tensor<gpu, unsigned char>(M * M * num_of_poly, s, ctx);
  Tensor<gpu, 1, int> d_num_of_counts_t = get_1d_tensor<gpu, int>(num_of_poly, s, ctx);
  Tensor<gpu, 1, uint> d_cnts_t = get_1d_tensor<gpu, uint>(CNTS_SIZE * num_of_poly, s, ctx);


  MSHADOW_REAL_TYPE_SWITCH(inputs[mrcnn_index::kRoi].type_flag_, DType, {
    const auto rois = inputs[mrcnn_index::kRoi].FlatToKD<gpu, 3, DType>(s);
    const auto matches = inputs[mrcnn_index::kMatches].FlatTo2D<gpu, DType>(s);
    const auto cls_targets = inputs[mrcnn_index::kClasses].FlatTo2D<gpu, DType>(s);
    // dense_polys need to be non-const

    auto out_masks = outputs[mrcnn_index::kMask].FlatToKD<gpu, 5, DType>(s);
    auto out_mask_cls = outputs[mrcnn_index::kMaskClasses].FlatToKD<gpu, 5, DType>(s);

    int batch_size = rois.shape_[0];

    int num_el = outputs[mrcnn_index::kMask].Size();

    // Mask Target generation
    int num_of_rois = rois.shape_[1];

    auto stream = mshadow::Stream<gpu>::GetStream(s);

    crop_and_scale_cuda_kernel<<<num_of_poly, 256, 0, stream>>>(dense_polys.dptr_,
                                                                polys_per_instance.dptr_,
                                                                poly_rel_idx.dptr_,
                                                                num_of_poly,
                                                                num_of_rois * batch_size,
                                                                rois.dptr_,
                                                                M);

    std::cout << M << std::endl;
    std::cout << num_of_poly << std::endl;
    std::cout << num_of_rois << std::endl;
    std::cout << batch_size << std::endl;
    std::cout << dense_polys.shape_ << std::endl;
    cudaDeviceSynchronize();

    rasterize_kernel<<<num_of_poly, 256, 0, stream>>>(dense_polys.dptr_,
                                                      poly_rel_idx.dptr_,
                                                      M, M,
                                                      d_mask_t.dptr_);
    cudaDeviceSynchronize();
    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
    unsigned char* test = new unsigned char[28*28];
    cudaMemcpy(test, d_mask_t.dptr_, 28*28*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
    for (int i = 0; i < 28; ++i) {
      for (int j = 0; j < 28; ++j) {
        if (test[i*28 + j] == 0) {
          std::cout << " ";
        } else {
          std::cout << "#";
        }
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;



    // TODO: larger threads-per-block might be better here, because each CTA uses 32 KB of shmem,
    // and occupancy is likely shmem capacity bound
    /*rle_fr_poly_cuda_kernel<<<num_of_poly, 1024, 0, stream>>>(dense_polys.dptr_,*/
                                                              /*poly_rel_idx.dptr_,*/
                                                              /*M, M,*/
                                                              /*d_cnts_t.dptr_,*/
                                                              /*d_xyuv_t.dptr_,*/
                                                              /*d_xyuv_t.dptr_ + BUFFER_SIZE * num_of_poly,*/
                                                              /*d_xyuv_t.dptr_ + 2 * BUFFER_SIZE * num_of_poly,*/
                                                              /*d_xyuv_t.dptr_ + 3 * BUFFER_SIZE * num_of_poly,*/
                                                              /*d_a_t.dptr_,*/
                                                              /*d_b_t.dptr_,*/
                                                              /*d_num_of_counts_t.dptr_);*/

    /*decode_rle_cuda_kernel<<<num_of_poly, 256, 0, stream>>>(d_num_of_counts_t.dptr_,*/
                                                            /*d_cnts_t.dptr_,*/
                                                            /*M, M,*/
                                                            /*d_mask_t.dptr_);*/

    // out: 2 * (B, N, C, MS, MS)
    int total_num_of_masks = batch_size * num_of_rois * param.num_classes;
    merge_masks_cuda_kernel<<<total_num_of_masks, 256, 0, stream>>>(d_mask_t.dptr_,
                                                                    out_masks.dptr_,
                                                                    M, polys_per_instance.dptr_,
                                                                    cls_targets.dptr_,
                                                                    param.num_classes);

    dim3 dimGrid = dim3(CUDA_GET_BLOCKS(num_el));
    dim3 dimBlock = dim3(block_dim_size);

    MRCNNMaskTargetKernel<<<dimGrid, dimBlock, 0, stream>>>
    (rois.dptr_, matches.dptr_, cls_targets.dptr_,
    out_masks.dptr_, out_mask_cls.dptr_,
    num_el, batch_size, param.num_classes, num_of_rois,
    param.mask_size[0], param.mask_size[1], param.sample_ratio);
    MSHADOW_CUDA_POST_KERNEL_CHECK(MRCNNMaskTargetKernel);
  });
}

DMLC_REGISTER_PARAMETER(MRCNNMaskTargetParam);

NNVM_REGISTER_OP(_contrib_mrcnn_mask_target)
.describe("Generate mask targets for Mask-RCNN.")
.set_num_inputs(6)
.set_num_outputs(2)
.set_attr_parser(ParamParser<MRCNNMaskTargetParam>)
.set_attr<mxnet::FInferShape>("FInferShape", MRCNNMaskTargetShape)
.set_attr<nnvm::FInferType>("FInferType", MRCNNMaskTargetType)
.set_attr<FCompute>("FCompute<gpu>", MRCNNMaskTargetCompute<gpu>)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.add_argument("rois", "NDArray-or-Symbol", "Bounding box coordinates, a 3D array")
.add_argument("dense_polys", "NDArray-or-Symbol", "Dense coordinates representing all the"
"polygons, a 2D array.")
.add_argument("polys_per_instance", "NDArray-or-Symbol", "The number of polygons per instance"
"(last value is the total number of   polygons), a 1D array of size B * num_rois + 1.")
.add_argument("poly_rel_idx", "NDArray-or-Symbol", "The per-polygon offset in `dense_polys`,"
"a 1D array of size size num_of_polys + 1.")
.add_argument("matches", "NDArray-or-Symbol", "Index to a gt_mask, a 2D array")
.add_argument("cls_targets", "NDArray-or-Symbol",
              "Value [0, num_class), excluding background class, a 2D array")
.add_arguments(MRCNNMaskTargetParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
