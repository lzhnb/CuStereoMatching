// Copyright 2022 Zhihao Liang
#include "stereo_matching.hpp"

const float EPSILON = 1e-8;

__device__ float query_ij(
    const float* __restrict__ img_ptr, // [H, W]
    const int32_t H,
    const int32_t W,
    const int32_t i,
    const int32_t j
) { return (i < 0 || i >= H || j < 0 || j >= W) ? 0.f : img_ptr[i * W + j]; }

// KERNELS
__global__ void get_cost_volume_kernel(
    const int32_t elements,
    const int32_t H,
    const int32_t W,
    const int32_t D,
    const int32_t patch,
    const float* __restrict__ camera_ptr,       // [H, W]
    const float* __restrict__ projector_ptr,    // [H, W]
    // output
    float* __restrict__ cost_volume_ptr         // [H, crop_w, D + 1]
) {
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= elements) { return; }
    const int32_t crop_w = W - D;
    const int32_t d_idx = tid % (D + 1);
    const int32_t w_idx = (tid / (D + 1)) % crop_w;
    const int32_t h_idx = tid / ((D + 1) * crop_w);

    // loop patch to get the mean value
    float cam_mean = 0, proj_mean = 0;
#pragma unroll
    for (int32_t i = 0; i < patch; ++i) {
        for (int32_t j = 0; j < patch; ++j) {
            const int32_t cam_proj_i = h_idx + i - patch / 2;
            const int32_t cam_j = (w_idx + D) + j - patch / 2;
            const int32_t proj_j = (w_idx + d_idx) + j - patch / 2;
            const float cam = query_ij(camera_ptr, H, W, cam_proj_i, cam_j);
            const float proj = query_ij(projector_ptr, H, W, cam_proj_i, proj_j);
            cam_mean += cam;
            proj_mean += proj;
        }
    }
    cam_mean /= (patch * patch);
    proj_mean /= (patch * patch);

    float exy = 0, ex2 = 0, ey2 = 0;
#pragma unroll
    for (int32_t i = 0; i < patch; ++i) {
        for (int32_t j = 0; j < patch; ++j) {
            const int32_t cam_proj_i = h_idx + i - patch / 2;
            const int32_t cam_j = (w_idx + D) + j - patch / 2;
            const int32_t proj_j = (w_idx + d_idx) + j - patch / 2;
            const float cam = query_ij(camera_ptr, H, W, cam_proj_i, cam_j) - cam_mean;
            const float proj = query_ij(projector_ptr, H, W, cam_proj_i, proj_j) - proj_mean;

            exy += cam * proj;
            ex2 += cam * cam;
            ey2 += proj * proj;
        }
    }
    // for (int32_t i = 0; i < patch * patch; ++i) {
    //     exy += camera_patch_ptr[i] * projector_patch_ptr[i];
    //     ex2 += camera_patch_ptr[i] * camera_patch_ptr[i];
    //     ey2 += projector_patch_ptr[i] * projector_patch_ptr[i];
    // }
    cost_volume_ptr[tid] = (exy + EPSILON) / (sqrtf(ex2 * ey2 + EPSILON));
}

Tensor stereo::stereo_matching_forward_wrapper(
    const Tensor& camera,    // [H, W]
    const Tensor& projector, // [H, W]
    const int32_t D,
    const int32_t patch,
    // output
    Tensor& disparity    // [H, W]
) {

    const int32_t H = camera.size(0), W = camera.size(1);
    const int32_t crop_w = W - D;
    const int32_t elements = H * crop_w * (D + 1), threads = 1024;
    const int32_t blocks = ceil((elements - 1) / threads) + 1;

    Tensor cost_volume = torch::zeros({H, crop_w, D + 1},
            torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));

    get_cost_volume_kernel<<<blocks, threads>>>(
        elements,
        H,
        W,
        D,
        patch,
        camera.data_ptr<float>(),
        projector.data_ptr<float>(),
        // output
        cost_volume.data_ptr<float>()
    );
    
    return cost_volume;
}
