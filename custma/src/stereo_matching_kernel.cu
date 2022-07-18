// Copyright 2022 Zhihao Liang
#include "stereo_matching.hpp"

/* NOTE: now ignore the crop_w and D */
const float EPSILON = 1e-8;

__device__ float query_ij(
    const float* __restrict__ val_ptr, // [H, W]
    const int32_t H,
    const int32_t W,
    const int32_t i,
    const int32_t j) {
    return (i < 0 || i >= H || j < 0 || j >= W) ? 0.f : val_ptr[i * W + j];
}

// KERNELS
__global__ void unfold_kernel(
    const int32_t H,
    const int32_t W,
    const int32_t ks,
    const float* __restrict__ inputs_ptr, // [H, W]
    // output
    float* __restrict__ outputs_ptr // [H, W, ks * ks]
) {
    const int32_t h_idx = blockIdx.x;
    const int32_t w_idx = blockIdx.y;
    const int32_t i = threadIdx.x;
    const int32_t j = threadIdx.y;
    const int32_t patch_i = h_idx + i - ks / 2;
    const int32_t patch_j = w_idx + j - ks / 2;
    const int32_t off = ks * ks;

    const float val = query_ij(inputs_ptr, H, W, patch_i, patch_j);
    outputs_ptr[h_idx * W * off + w_idx * off + i * ks + j] = val;
}


__global__ void get_image_patch_grad_kernel(
    const int32_t H,
    const int32_t W,
    const int32_t D,
    const int32_t ks,
    const bool record_grad,
    const float* __restrict__ camera_ptr, // [H, W]
    const float* __restrict__ projector_ptr, // [H, W]
    const float* __restrict__ cost_volume_grad_ptr, // [H, W - D, W]
    const float* __restrict__ ex2_ptr, // [H, W - D]
    const float* __restrict__ ey2_ptr, // [H, W]
    const float* __restrict__ exy_ptr, // [H, W - D, W]
    const float* __restrict__ ex2_mean_ptr, // [H, W - D]
    const float* __restrict__ ey2_mean_ptr, // [H, W]
    // output
    float* __restrict__ camera_grad_patch_ptr, // [H, W, ks, ks]
    float* __restrict__ ex2_grad_ptr, // [H, W - D]
    float* __restrict__ exy_grad_ptr // [H, W - D, W]
) {
    // the coordinate of pixel
    const int32_t h_idx = blockIdx.x;
    const int32_t w_idx = blockIdx.y;
    const int32_t d_idx = blockIdx.z;
    // relative coordinate in the patch
    const int32_t i = threadIdx.x;
    const int32_t j = threadIdx.y;
    const int32_t off = ks * ks;

    const float ex2 = ex2_ptr[h_idx * (W - D) + w_idx];
    const float ey2 = ey2_ptr[h_idx * W + d_idx];
    const float exy = exy_ptr[h_idx * (W - D) * W + w_idx * W + d_idx];

    // const float factor = (sqrtf(ex2 * ey2 + EPSILON));
    const float deno = 1 / (sqrtf(ex2 * ey2 + EPSILON)),
                deno3 = 1 / powf((sqrtf(ex2 * ey2 + EPSILON)), 3);
    const float cost_volume_grad =
        cost_volume_grad_ptr[h_idx * (W - D) * W + w_idx * W + d_idx];
    const float ex2_grad =
        -ey2_ptr[h_idx * W + d_idx] * (exy + EPSILON) * deno3 / 2;
    const float exy_grad = cost_volume_grad * deno;
    const float ex2_mean = ex2_mean_ptr[h_idx * (W - D) + w_idx];
    const float ey2_mean = ey2_mean_ptr[h_idx * W + d_idx];

    const int32_t cam_patch_i = h_idx + i - ks / 2;
    const int32_t cam_patch_j = w_idx + D + j - ks / 2;
    const int32_t proj_patch_i = h_idx + i - ks / 2;
    const int32_t proj_patch_j = d_idx + j - ks / 2;

    // record the intermediate gradient for debug
    if (i == 0 && j == 0) {
        if (record_grad) {
            atomicAdd(&ex2_grad_ptr[h_idx * (W - D) + w_idx], ex2_grad);
            exy_grad_ptr[h_idx * (W - D) * W + w_idx * W + d_idx] = exy_grad;
        }
    }

    /* ex2 term */
    const float cam_val = query_ij(camera_ptr, H, W, cam_patch_i, cam_patch_j);
    const float cam_grad_ex2_term = 2 * (cam_val - ex2_mean) * ex2_grad;
    camera_grad_patch_ptr[h_idx * W * off + w_idx * off + i * ks + j] +=
        cam_grad_ex2_term;

    /* exy term */
    const float proj_val =
        query_ij(projector_ptr, H, W, proj_patch_i, proj_patch_j);
    const float cam_grad_exy_term = (proj_val - ey2_mean) * exy_grad;
    camera_grad_patch_ptr[h_idx * W * off + w_idx * off + i * ks + j] +=
        cam_grad_exy_term;
}

__global__ void patch_grad_to_image_kernel(
    const int32_t H,
    const int32_t W,
    const int32_t ks,
    const float* __restrict__ camera_patches_grad_ptr, // [H, W, ks, ks]
    // output
    float* __restrict__ camera_grad_ptr // [H, W]
) {
    // the coordinate of pixel
    const int32_t h_idx = blockIdx.x;
    const int32_t w_idx = blockIdx.y;
    // relative coordinate in the patch
    const int32_t i = threadIdx.x;
    const int32_t j = threadIdx.y;
    const int32_t off = ks * ks;

    const int32_t cam_i = h_idx + i - ks / 2;
    const int32_t cam_j = w_idx + j - ks / 2;
    if (cam_i < 0 || cam_i >= H || cam_j < 0 || cam_j >= W) {
        return;
    }
    atomicAdd(
        camera_grad_ptr + cam_i * W + cam_j,
        camera_patches_grad_ptr[h_idx * W * off + w_idx * off + i * ks + j]);
}

vector<Tensor> stereo::stereo_matching_forward(
    const Tensor& camera, // [H, W]
    const Tensor& projector, // [H, W]
    const int32_t D,
    const int32_t kernel_size) {
    // check
    CHECK_INPUT(camera);
    CHECK_INPUT(projector);

    // get parameters
    const int32_t H = camera.size(0), W = camera.size(1);
    const int32_t crop_w = W - D;
    const int32_t threads = 1024;
    assert(projector.size(0) == H && projector.size(1) == W);

    // unfold operation
    Tensor camera_patch = torch::zeros(
        {H, W, kernel_size * kernel_size},
        torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    Tensor projector_patch = torch::zeros(
        {H, W, kernel_size * kernel_size},
        torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));

    dim3 dim_grid(H, W);
    dim3 dim_block(kernel_size, kernel_size);
    unfold_kernel<<<dim_grid, dim_block>>>(
        H,
        W,
        kernel_size,
        camera.data_ptr<float>(),
        // output
        camera_patch.data_ptr<float>());
    unfold_kernel<<<dim_grid, dim_block>>>(
        H,
        W,
        kernel_size,
        projector.data_ptr<float>(),
        // output
        projector_patch.data_ptr<float>());
    Tensor camera_patch_mean = torch::mean(camera_patch, 2, true);
    Tensor projector_patch_mean = torch::mean(projector_patch, 2, true);
    camera_patch -= camera_patch_mean;
    projector_patch -= projector_patch_mean;

    Tensor ex2 =
        torch::bmm(
            camera_patch.reshape({H * W, 1, kernel_size * kernel_size}),
            camera_patch.reshape({H * W, kernel_size * kernel_size, 1}))
            .reshape({H, W});
    Tensor ey2 =
        torch::bmm(
            projector_patch.reshape({H * W, 1, kernel_size * kernel_size}),
            projector_patch.reshape({H * W, kernel_size * kernel_size, 1}))
            .reshape({H, W});

    Tensor exy = torch::bmm(camera_patch, projector_patch.permute({0, 2, 1}));
    Tensor cost_volume = (exy + EPSILON) / torch::sqrt(ex2 * ey2 + EPSILON).unsqueeze_(2);

    vector<Tensor> results(8);

    results[0] = ex2;
    results[1] = ey2;
    results[2] = exy;
    results[3] = camera_patch_mean;
    results[4] = projector_patch_mean;
    results[5] = cost_volume;
    results[6] = camera_patch;
    results[7] = projector_patch;

    return results;
}

vector<Tensor> stereo::stereo_matching_backward(
    const Tensor& cost_volume_grad, // [H, W - D, W]
    const Tensor& camera, // [H, W]
    const Tensor& projector, // [H, W]
    const Tensor& ex2,
    const Tensor& ey2,
    const Tensor& exy,
    const Tensor& ex2_mean,
    const Tensor& ey2_mean,
    const int32_t kernel_size,
    const bool record) {
    // check
    CHECK_INPUT(cost_volume_grad);
    CHECK_INPUT(camera);
    CHECK_INPUT(projector);
    CHECK_INPUT(ex2);
    CHECK_INPUT(ey2);
    CHECK_INPUT(exy);
    CHECK_INPUT(ex2_mean);
    CHECK_INPUT(ey2_mean);

    // get parameters
    const int32_t H = camera.size(0), W = camera.size(1),
                  crop_w = cost_volume_grad.size(1);
    const int32_t D = W - crop_w;
    const int32_t threads = 1024;

    Tensor ex2_grad;
    Tensor exy_grad;
    if (record) {
        ex2_grad = torch::zeros_like(ex2);
        exy_grad = torch::zeros_like(exy);
    }
    Tensor camera_grad = torch::zeros_like(camera);
    Tensor camera_patch_grad = torch::zeros(
        {H, W, kernel_size, kernel_size},
        torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));

    dim3 dim_block(kernel_size, kernel_size);
    dim3 dim_grid(H, crop_w, W);
    get_image_patch_grad_kernel<<<dim_grid, dim_block>>>(
        H,
        W,
        D,
        kernel_size,
        record,
        camera.data_ptr<float>(),
        projector.data_ptr<float>(),
        cost_volume_grad.data_ptr<float>(),
        ex2.data_ptr<float>(),
        ey2.data_ptr<float>(),
        exy.data_ptr<float>(),
        ex2_mean.data_ptr<float>(),
        ey2_mean.data_ptr<float>(),
        // output
        camera_patch_grad.data_ptr<float>(),
        record ? ex2_grad.data_ptr<float>() : nullptr,
        record ? exy_grad.data_ptr<float>() : nullptr);

    dim3 img_dim_grid(H, W);
    patch_grad_to_image_kernel<<<img_dim_grid, dim_block>>>(
        H,
        W,
        kernel_size,
        camera_patch_grad.data_ptr<float>(),
        // output
        camera_grad.data_ptr<float>());

    vector<Tensor> results(3);

    results[0] = camera_grad;
    results[1] = ex2_grad;
    results[2] = exy_grad;

    return results;
}
