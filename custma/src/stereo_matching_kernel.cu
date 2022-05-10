// Copyright 2022 Zhihao Liang
#include "stereo_matching.hpp"

const float EPSILON = 1e-8;

// KERNELS
__global__ void get_cost_volume_kernel(
    const int32_t elements,
    const int32_t W,
    const int32_t D,
    const int32_t patch,
    const float* __restrict__ camera_ptr,       // [H, W, patch * patch]
    const float* __restrict__ projector_ptr,    // [H, W, patch * patch]
    // output
    float* __restrict__ cost_volume_ptr         // [H, crop_w, D + 1]
) {
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= elements) { return; }
    const int32_t crop_w = W - D;
    const int32_t d_idx = tid % (D + 1);
    const int32_t w_idx = (tid / (D + 1)) % crop_w;
    const int32_t h_idx = tid / ((D + 1) * crop_w);

    const int32_t off_w = patch * patch, off_h = patch * patch * W;
    
    const float* camera_patch_ptr = camera_ptr + h_idx * off_h + (w_idx + D) * off_w;
    const float* projector_patch_ptr = projector_ptr + h_idx * off_h + (w_idx + d_idx) * off_w;

    // loop patch
    float exy = 0.f, ex2 = 0.f, ey2 = 0.f;
#pragma unroll
    for (int32_t i = 0; i < patch * patch; ++i) {
        exy += camera_patch_ptr[i] * projector_patch_ptr[i];
        ex2 += camera_patch_ptr[i] * camera_patch_ptr[i];
        ey2 += projector_patch_ptr[i] * projector_patch_ptr[i];
    }
    cost_volume_ptr[tid] = (exy + EPSILON) / (sqrtf(ex2 * ey2 + EPSILON));
}

Tensor stereo::stereo_matching_forward_wrapper(
    const Tensor& camera_flatten,    // [H, W, p*p]
    const Tensor& projector_flatten, // [H, W, p*p]
    const int32_t D,
    const int32_t patch,
    // output
    Tensor& disparity    // [H, W]
) {

    const int32_t H = camera_flatten.size(0), W = camera_flatten.size(1);
    const int32_t crop_w = W - D;
    const int32_t elements = H * crop_w * (D + 1), threads = 1024;
    const int32_t blocks = ceil((elements - 1) / threads) + 1;

    Tensor cost_volume = torch::zeros({H, crop_w, D + 1},
            torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));

    get_cost_volume_kernel<<<blocks, threads>>>(
        elements,
        W,
        D,
        patch,
        camera_flatten.data_ptr<float>(),
        projector_flatten.data_ptr<float>(),
        // output
        cost_volume.data_ptr<float>()
    );
    
    return cost_volume;
}
