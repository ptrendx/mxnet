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
const int BUFFER_SIZE = 6144;
const int CNTS_SIZE = 6144;


// Batch size upgrade: Done
__global__ void crop_and_scale_cuda_kernel(double *dense_poly_data, const int *per_roi_poly_idx, const int *poly_rel_idx,
                                           int poly_count, int roi_count, float4 *roi_data, int mask_size, int batch_size) {
    int tid = threadIdx.x;
    int block_jump = blockDim.x;
    int poly_id = blockIdx.x;
    int roi_idx;
    for (roi_idx = 0; roi_idx < roi_count * batch_size; roi_idx++){
        if (poly_id < per_roi_poly_idx[roi_idx + 1]) break;
    }
    float w = roi_data[roi_idx].z - roi_data[roi_idx].x;
  	float h = roi_data[roi_idx].w - roi_data[roi_idx].y;
  	w = fmaxf(w, 1.0f);
  	h = fmaxf(h, 1.0f);
    float ratio_h = ((float) mask_size) / h;
    float ratio_w = ((float) mask_size) / w;

    int poly_ptr_idx_start = poly_rel_idx[poly_id];
    int poly_ptr_idx_end = poly_rel_idx[poly_id + 1];

    double *poly_data_buf = dense_poly_data + poly_ptr_idx_start;
    int len = poly_ptr_idx_end - poly_ptr_idx_start;

    for (int j = tid; j < len; j += block_jump) {
        if (j % 2 == 0) poly_data_buf[j] = ratio_w*((float) poly_data_buf[j]- roi_data[roi_idx].x);
        if (j % 2 == 1) poly_data_buf[j] = ratio_h*((float) poly_data_buf[j]- roi_data[roi_idx].y);
    }
}

// Batch size upgrade: TODO
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
        int xs = x[j], xe = x[j + 1], ys = y[j], ye = y[j + 1], dx, dy, t, d, dist;
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
        int xs = x[j], xe = x[j + 1], ys = y[j], ye = y[j + 1], dx, dy, t, d, dist;
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

// Batch size upgrade: TODO
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
        int bin = 0;
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
__global__ void merge_masks_cuda_kernel(unsigned char *masks_in, float *masks_out, const int mask_size,
                                        int *per_roi_poly_idx, const float *matches, int roi_count,
                                        int num_classes) {

    int roi_idx = blockIdx.x;
    int tid = threadIdx.x;
    int jump_block = blockDim.x;
    int mask_start_idx = per_roi_poly_idx[roi_idx];
    int num_of_masks_to_merge = per_roi_poly_idx[roi_idx + 1] - per_roi_poly_idx[roi_idx];

    int class_idx = matches[roi_idx];
    int mask_offset = (roi_idx * num_classes + class_idx) * mask_size * mask_size;

    for(int j = tid; j < mask_size * mask_size; j += jump_block){
        int transposed_pixel = (j % mask_size) * mask_size + j / mask_size;
        unsigned char pixel = 0;
        for(int k = 0; k < num_of_masks_to_merge; k++){
            if (masks_in[(mask_start_idx + k) * mask_size * mask_size + j] == 1) pixel = 1;
            if (pixel == 1) break;
        }
        masks_out[mask_offset + transposed_pixel] = (float) pixel;
    }
}

template <typename xpu, typename DType>
Tensor<xpu, 1, DType> get_1d_tensor(int size, mshadow::Stream<xpu> *stream, const OpContext &ctx) {
  return ctx.requested[mrcnn_index::kTempSpace]
          .get_space_typed<xpu, 1, DType>(Shape1(size), stream);
}

template
void GenerateMaskTargetsFromPolygons(const Tensor<gpu, 1, double>& dense_vector,
                                     const Tensor<gpu, 1, int>& per_roi_poly_idx,
                                     const Tensor<gpu, 1, int>& poly_rel_idx,
                                     const Tensor<gpu, 3, float>& rois,
                                     const int mask_size,
                                     const int batch_size) {

    // only works with square mask targets
    const int M = mask_size;
    assert (M < 32);
    //if M >= 32, shared memory buffer size may not be
    //sufficient. Need to fix this by blocking

    int num_of_rois = rois.shape_[1];
    /*
      Tensor<gpu, 1, int> per_roi_poly_idx = NewTensor<gpu>(Shape1(num_of_rois + 1), 0, stream);
      int num_of_poly = 0;
      for (int i = 0; i < num_of_rois; i++){
          *(per_roi_poly_idx.data<int>() + i) = num_of_poly;
          num_of_poly += polygons[i].size();
      }
      *(per_roi_poly_idx.data<int>() + num_of_rois) = num_of_poly;

      auto poly_rel_idx = at::empty({num_of_poly + 1}, at::CPU(at::kInt));
      int start_idx = 0;
      int poly_count = 0;

      for(int i = 0; i < polygons.size(); i++){
          for(int j=0; j < polygons[i].size(); j++) {
            *(poly_rel_idx.data<int>() + poly_count) = start_idx;
            start_idx += polygons[i][j].size(0);
            poly_count++;
          }
      }
      *(poly_rel_idx.data<int>() + poly_count) = start_idx;
    */

    // maybe:
    Tensor<gpu, 1, int> d_x_t = get_1d_tensor<gpu, int>(BUFFER_SIZE * num_of_poly, stream, ctx);
    Tensor<gpu, 1, int> d_y_t = get_1d_tensor<gpu, int>(BUFFER_SIZE * num_of_poly, stream, ctx);
    Tensor<gpu, 1, int> d_u_t = get_1d_tensor<gpu, int>(BUFFER_SIZE * num_of_poly, stream, ctx);
    Tensor<gpu, 1, int> d_v_t = get_1d_tensor<gpu, int>(BUFFER_SIZE * num_of_poly, stream, ctx);
    Tensor<gpu, 1, uint> d_a_t = get_1d_tensor<gpu, uint>(BUFFER_SIZE * num_of_poly, stream, ctx); //used with uint* pointer
    Tensor<gpu, 1, uint> d_b_t = get_1d_tensor<gpu, uint>(BUFFER_SIZE * num_of_poly, stream, ctx);  //used with uint* pointer

    Tensor<gpu, 1, char> d_mask_t = get_1d_tensor<gpu, char>(M * M * num_of_poly, stream, ctx);
    Tensor<gpu, 1, int> d_num_of_counts_t = get_1d_tensor<gpu, int>(num_of_poly, stream, ctx);
    Tensor<gpu, 1, int> d_cnts_t = get_1d_tensor<gpu, int>(CNTS_SIZE * num_of_poly, stream, ctx);

    const float *d_roi_data = rois.data<float>();

    // Output
    auto result =  torch::empty({num_of_rois, M, M}, torch::CUDA(at::kFloat));
    auto out_masks = outputs[mrcnn_index::kMask].FlatToKD<gpu, 5, DType>(s);

    auto stream = at::cuda::getCurrentCUDAStream();

    crop_and_scale_cuda_kernel<<<num_of_poly, 256, 0, stream.stream()>>>(dense_vector.dptr_,
                                                                         per_roi_poly_idx.dptr_,
                                                                         poly_rel_idx.dptr_,
                                                                         poly_count,
                                                                         num_of_rois,
                                                                         (float4*) d_roi_data,
                                                                         M,
                                                                         B);

    // TODO: larger threads-per-block might be better here, because each CTA uses 32 KB of shmem,
    // and occupancy is likely shmem capacity bound
    rle_fr_poly_cuda_kernel<<<num_of_poly, 1024, 0, stream>>>(d_dense_vector.data<double>(),
                                                                   d_poly_rel_idx.data<int>(),
                                                                   M, M,
                                                                   (uint*) d_cnts_t.data<int>(),
                                                                   d_x_t.data<int>(),
                                                                   d_y_t.data<int>(),
                                                                   d_u_t.data<int>(),
                                                                   d_v_t.data<int>(),
                                                                   (uint*) d_a_t.data<int>(),
                                                                   (uint*) d_b_t.data<int>(),
                                                                   d_num_of_counts_t.data<int>(),
                                                                   B);

    decode_rle_cuda_kernel<<<num_of_poly, 256, 0, stream>>>(d_num_of_counts_t.data<int>(),
                                                                     (uint*) d_cnts_t.data<int>(),
                                                                     M, M,
                                                                     d_mask_t.data<unsigned char>(),
                                                                     B);

    // out: 2 * (B, N, C, MS, MS)
    merge_masks_cuda_kernel<<<B * num_of_rois, 256, 0, stream)>>>(d_mask_t.data<unsigned char>(),
                                                                  out_masks.data<float>(),
                                                                  M, d_per_roi_poly_idx.data<int>(),
                                                                  num_of_rois,
                                                                  matches,
                                                                  num_classes);
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

// Modified version of RoIAlignForwardKernel from Caffe (in roi_align.cu)
// Main modifications:
// - We don't need position_sensitive neither spatial_scale from the original RoIAlign kernel.
// - We replace `channels` by `num_classes` and modify the logic consequently (e.g. offset_in_data
//   does not use `c` anymore).
template <typename T>
__device__ void RoIAlignForward(
    const T* in_data,  // (B, M, H, W)
    const T* rois,  // (B, N, 4)
    const T* matches,  // (B, N)
    const int num_el,
    const int num_classes,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    const int num_rois,
    const int num_gtmasks,
    T* out_data) {  // (B, N, C, H, W)
  // Update kernel
  for (size_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < num_el;
       index += blockDim.x * gridDim.x) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    // int c = (index / pooled_width / pooled_height) % num_classes;
    int n = (index / pooled_width / pooled_height / num_classes) % num_rois;
    int batch_idx = (index / pooled_width / pooled_height / num_classes / num_rois);

    int roi_batch_ind = matches[batch_idx * num_rois + n];

    const T* offset_rois = rois + batch_idx * (4 * num_rois) + n * 4;
    // Do not using rounding; this implementation detail is critical
    T roi_start_w = offset_rois[0];
    T roi_start_h = offset_rois[1];
    T roi_end_w = offset_rois[2];
    T roi_end_h = offset_rois[3];

    // Force malformed ROIs to be 1x1
    T roi_width = max(roi_end_w - roi_start_w, (T)1.);
    T roi_height = max(roi_end_h - roi_start_h, (T)1.);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    const T* offset_in_data =
        in_data + batch_idx * num_gtmasks * height * width
        + roi_batch_ind * height * width;

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / pooled_height);  // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w;  // e.g. = 4

    T output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) {  // e.g., iy = 0, 1
      const T y = roi_start_h + ph * bin_size_h +
          static_cast<T>(iy + .5f) * bin_size_h /
              static_cast<T>(roi_bin_grid_h);  // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = roi_start_w + pw * bin_size_w +
            static_cast<T>(ix + .5f) * bin_size_w /
                static_cast<T>(roi_bin_grid_w);

        T val = bilinear_interpolate(
            offset_in_data, height, width, y, x, index);
        output_val += val;
      }
    }
    output_val /= count;

    out_data[index] = output_val;
  }
}


template<typename DType>
__global__ void MRCNNMaskTargetKernel(const DType *rois,
                                      const DType *gt_masks,
                                      const DType *matches,
                                      const DType *cls_targets,
                                      DType* sampled_masks,
                                      DType* mask_cls,
                                      const int total_out_el,
                                      int batch_size,
                                      int num_classes,
                                      int num_rois,
                                      int num_gtmasks,
                                      int gt_height,
                                      int gt_width,
                                      int mask_size_h,
                                      int mask_size_w,
                                      int sample_ratio) {
  // computing sampled_masks
  // RoIAlignForward(gt_masks, rois, matches, total_out_el,
  //                 num_classes, gt_height, gt_width, mask_size_h, mask_size_w,
  //                 sample_ratio, num_rois, num_gtmasks, sampled_masks);
  // new computing sampled_masks

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

template<>
void MRCNNMaskTargetRun<gpu>(const MRCNNMaskTargetParam& param, const std::vector<TBlob> &inputs,
                             const std::vector<TBlob> &outputs, mshadow::Stream<gpu> *s) {
  const int block_dim_size = kMaxThreadsPerBlock;
  using namespace mxnet_op;
  using mshadow::Tensor;

  auto stream = mshadow::Stream<gpu>::GetStream(s);

  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    auto rois = inputs[mrcnn_index::kRoi].FlatToKD<gpu, 3, DType>(s);
    auto gt_masks = inputs[mrcnn_index::kGtMask].FlatToKD<gpu, 4, DType>(s);
    auto matches = inputs[mrcnn_index::kMatches].FlatTo2D<gpu, DType>(s);
    auto cls_targets = inputs[mrcnn_index::kClasses].FlatTo2D<gpu, DType>(s);

    auto out_masks = outputs[mrcnn_index::kMask].FlatToKD<gpu, 5, DType>(s);
    auto out_mask_cls = outputs[mrcnn_index::kMaskClasses].FlatToKD<gpu, 5, DType>(s);

    int batch_size = gt_masks.shape_[0];
    int num_gtmasks = gt_masks.shape_[1];
    int gt_height = gt_masks.shape_[2];
    int gt_width = gt_masks.shape_[3];

    int num_el = outputs[mrcnn_index::kMask].Size();


    GenerateMaskTargetsFromPolygons();

    dim3 dimGrid = dim3(CUDA_GET_BLOCKS(num_el));
    dim3 dimBlock = dim3(block_dim_size);

    MRCNNMaskTargetKernel<<<dimGrid, dimBlock, 0, stream>>>
    (rois.dptr_, gt_masks.dptr_, matches.dptr_, cls_targets.dptr_,
    out_masks.dptr_, out_mask_cls.dptr_,
    num_el, batch_size, param.num_classes, param.num_rois,
    num_gtmasks, gt_height, gt_width,
    param.mask_size[0], param.mask_size[1], param.sample_ratio);
    MSHADOW_CUDA_POST_KERNEL_CHECK(MRCNNMaskTargetKernel);
  });
}

DMLC_REGISTER_PARAMETER(MRCNNMaskTargetParam);

NNVM_REGISTER_OP(_contrib_mrcnn_mask_target)
.describe("Generate mask targets for Mask-RCNN.")
.set_num_inputs(4)
.set_num_outputs(2)
.set_attr_parser(ParamParser<MRCNNMaskTargetParam>)
.set_attr<mxnet::FInferShape>("FInferShape", MRCNNMaskTargetShape)
.set_attr<nnvm::FInferType>("FInferType", MRCNNMaskTargetType)
.set_attr<FCompute>("FCompute<gpu>", MRCNNMaskTargetCompute<gpu>)
.add_argument("rois", "NDArray-or-Symbol", "Bounding box coordinates, a 3D array")
.add_argument("gt_masks", "NDArray-or-Symbol", "Input masks of full image size, a 4D array")
.add_argument("matches", "NDArray-or-Symbol", "Index to a gt_mask, a 2D array")
.add_argument("cls_targets", "NDArray-or-Symbol",
              "Value [0, num_class), excluding background class, a 2D array")
.add_arguments(MRCNNMaskTargetParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
