// Copyright 2022 Zhihao Liang
#include "stereo_matching.hpp"

// __device__ float query_ij(
//     const float* __restrict__ val_ptr, // [H, W]
//     const int32_t H,
//     const int32_t W,
//     const int32_t i,
//     const int32_t j) {
//     return (i < 0 || i >= H || j < 0 || j >= W) ? 0.f : val_ptr[i * W + j];
// }


__global__ void exy_grad_to_image_kernel(
    const int32_t H,
    const int32_t W,
    const int32_t D,
    const int32_t kernel_size,
    const float* __restrict__ exy_grad_ptr, // [H, W - D, W]
    const float* __restrict__ projector_ptr, // [H, W]
    const float* __restrict__ ey2_mean_ptr, // [H, W]
    // output
    float* __restrict__ camera_grad_ptr // [H, W]
) {
    // the coordinate of pixel
    const int32_t h_idx = blockIdx.x;
    const int32_t w_idx = blockIdx.y;
    const int32_t d_idx = blockIdx.z;
    // relative coordinate in the patch
    const int32_t i = threadIdx.x;
    const int32_t j = threadIdx.y;

    const float temp_exy_grad = exy_grad_ptr[h_idx * (W - D) * W + w_idx * W + d_idx];

    const int32_t cam_patch_i = h_idx + i - kernel_size / 2;
    const int32_t cam_patch_j = w_idx + D + j - kernel_size / 2;
    const int32_t proj_patch_i = h_idx + i - kernel_size / 2;
    const int32_t proj_patch_j = d_idx + j - kernel_size / 2;
    const float temp_proj = (proj_patch_i < 0 || proj_patch_i >= H || proj_patch_j < 0 || proj_patch_j >= W) ? 0.f : projector_ptr[proj_patch_i * W + proj_patch_j];
    const float temp_grad =
        (temp_proj - ey2_mean_ptr[h_idx * W + d_idx]) * temp_exy_grad;

    if (cam_patch_i < 0 || cam_patch_i >= H || cam_patch_j < 0 ||
        cam_patch_j >= W) {
        return;
    }
    atomicAdd(
        &camera_grad_ptr[cam_patch_i * W + cam_patch_j], temp_grad);
    __syncthreads();
}


__global__ void ex2_grad_to_image_kernel(
    const int32_t H,
    const int32_t W,
    const int32_t D,
    const int32_t kernel_size,
    const float* __restrict__ ex2_grad_ptr, // [H, W - D]
    const float* __restrict__ camera_ptr, // [H, W]
    const float* __restrict__ ex2_mean_ptr, // [H, W - D]
    // output
    float* __restrict__ camera_grad_ptr // [H, W]
) {
    // the coordinate of pixel
    const int32_t h_idx = blockIdx.x;
    const int32_t w_idx = blockIdx.y;
    // relative coordinate in the patch
    const int32_t i = threadIdx.x;
    const int32_t j = threadIdx.y;

    const int32_t patch_i = h_idx + i - kernel_size / 2;
    const int32_t patch_j = w_idx + D + j - kernel_size / 2;
    const float val = (patch_i < 0 || patch_i >= H || patch_j < 0 || patch_j >= W) ? 0.f : camera_ptr[patch_i * W + patch_j];

    if (patch_i < 0 || patch_i >= H || patch_j < 0 || patch_j >= W) {
        return;
    }
    atomicAdd(
        &camera_grad_ptr[patch_i * W + patch_j],
        2 * (val - ex2_mean_ptr[h_idx * (W - D) + w_idx]) * ex2_grad_ptr[h_idx * (W - D) + w_idx]);
    __syncthreads();
}


Tensor stereo::exy_grad_to_image(
    const Tensor& exy_grad, // [H, W - D, W]
    const Tensor& camera, // [H, W]
    const Tensor& projector, // [H, W]
    const Tensor& ey2_mean, // [H, W]
    const int32_t kernel_size) {
    // check
    CHECK_INPUT(exy_grad);
    CHECK_INPUT(camera);
    CHECK_INPUT(projector);
    CHECK_INPUT(ey2_mean);

    // get parameters
    const int32_t H = camera.size(0), W = camera.size(1),
                  crop_w = exy_grad.size(1);
    const int32_t D = W - crop_w;

    Tensor camera_grad = torch::zeros_like(camera);
    
    dim3 dim_block(kernel_size, kernel_size);
    dim3 exy_dim_grid(H, crop_w, W);
    exy_grad_to_image_kernel<<<exy_dim_grid, dim_block>>>(
        H,
        W,
        D,
        kernel_size,
        exy_grad.data_ptr<float>(),
        projector.data_ptr<float>(),
        ey2_mean.data_ptr<float>(),
        // output
        camera_grad.data_ptr<float>());

    return camera_grad;
}


Tensor stereo::ex2_grad_to_image(
    const Tensor& ex2_grad, // [H, W - D]
    const Tensor& camera, // [H, W]
    const Tensor& ex2_mean, // [H, W - D]
    const int32_t kernel_size) {
    // check
    CHECK_INPUT(ex2_grad);
    CHECK_INPUT(camera);
    CHECK_INPUT(ex2_mean);

    // get parameters
    const int32_t H = camera.size(0), W = camera.size(1),
                  crop_w = ex2_mean.size(1);
    const int32_t D = W - crop_w;

    Tensor camera_grad = torch::zeros_like(camera);
    // Tensor camera_mean_grad = torch::zeros(
    //     {H, crop_w},
    //     torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));

    dim3 dim_block(kernel_size, kernel_size);
    dim3 ex2_dim_grid(H, crop_w);
    ex2_grad_to_image_kernel<<<ex2_dim_grid, dim_block>>>(
        H,
        W,
        D,
        kernel_size,
        ex2_grad.data_ptr<float>(),
        camera.data_ptr<float>(),
        ex2_mean.data_ptr<float>(),
        // output
        camera_grad.data_ptr<float>());

    // printf("\ncamera_grad-{0, 0}: %f (refer to 136.4016)\n\n",
    // camera_grad.index({0, 0}).item<float>());
    return camera_grad;
}


