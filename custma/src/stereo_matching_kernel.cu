// Copyright 2022 Zhihao Liang
#include "stereo_matching.hpp"

#define MAX_KERNEL_SIZE 15

__device__ const float EPSILON = 1e-8;

__device__ float query_ij(
    const float* __restrict__ img_ptr, // [H, W]
    const int32_t H,
    const int32_t W,
    const int32_t i,
    const int32_t j) {
    return (i < 0 || i >= H || j < 0 || j >= W) ? 0.f : img_ptr[i * W + j];
}

/* NOTE: now ignore the crop_w and D */

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

__global__ void gather_exy_cost_volume_kernel(
    const int32_t H,
    const int32_t W,
    const int32_t D,
    const float* __restrict__ full_exy_ptr, // [H, W, W]
    const float* __restrict__ full_cost_volume_ptr, // [H, W, W]
    // output
    float* __restrict__ exy_ptr, // [H, W, D]
    float* __restrict__ cost_volume_ptr // [H, W, D]
) {
    const int32_t h_idx = blockIdx.x, w_idx = blockIdx.y, d_idx = threadIdx.x;

    exy_ptr[h_idx * W * D + (w_idx + D) * D + d_idx] =
        full_exy_ptr[h_idx * W * W + (w_idx + D) * W + w_idx + d_idx];
    cost_volume_ptr[h_idx * W * D + (w_idx + D) * D + d_idx] =
        full_cost_volume_ptr[h_idx * W * W + (w_idx + D) * W + w_idx + d_idx];
}

__global__ void get_patches_grad_kernel(
    const int32_t H,
    const int32_t W,
    const int32_t D,
    const int32_t ks,
    const float* __restrict__ cost_volume_grad, // [H, W, D]
    const float* __restrict__ cam_ptr, // [H, W]
    const float* __restrict__ proj_ptr, // [H, W]
    const float* __restrict__ ex2_ptr, // [H, W]
    const float* __restrict__ ey2_ptr, // [H, W]
    const float* __restrict__ exy_ptr, // [H, W, D]
    const float* __restrict__ cam_patch_mean_ptr, // [H, W]
    const float* __restrict__ proj_patch_mean_ptr, // [H, W]
    // output
    float* __restrict__ camera_patches_grad_ptr // [H, W, ks, ks]
) {
    const int32_t h_idx = blockIdx.x, w_idx = blockIdx.y;
    const int32_t row = threadIdx.x, col = threadIdx.y;

    const float cam_mean = cam_patch_mean_ptr[h_idx * W + w_idx + D];

    const int32_t cam_i = h_idx + row - ks / 2;
    const int32_t cam_j = w_idx + col - ks / 2 + D;
    const float cam = query_ij(cam_ptr, H, W, cam_i, cam_j) - cam_mean;

    const float ex2 = ex2_ptr[h_idx * W + w_idx + D];

    for (int32_t d_idx = 0; d_idx < D; ++d_idx) {
        const float proj_mean = proj_patch_mean_ptr[h_idx * W + w_idx + d_idx];

        const int32_t proj_i = h_idx + row - ks / 2;
        const int32_t proj_j = d_idx + col - ks / 2 + w_idx;
        const float proj = query_ij(proj_ptr, H, W, proj_i, proj_j) - proj_mean;

        const float ey2 = ey2_ptr[h_idx * W + w_idx + d_idx];
        const float exy = exy_ptr[h_idx * W * D + (w_idx + D) * D + d_idx];

        const float cost_grad =
            cost_volume_grad[h_idx * W * D + (w_idx + D) * D + d_idx];
        const int32_t off_w = ks * ks, off_h = ks * ks * W;
        float* curr_camera_patches_grad_ptr =
            camera_patches_grad_ptr + h_idx * off_h + (w_idx + D) * off_w;

        // calculate 1 time to save time
        const float deno = 1 / (sqrtf(ex2 * ey2 + EPSILON)),
                    deno3 = 1 / powf((sqrtf(ex2 * ey2 + EPSILON)), 3);

        // exy term
        float exy_factor = proj * deno;
        // ex2 term
        float ex2_factor = -(ey2 * cam * (exy + EPSILON)) * deno3;
        const float grad = cost_grad * (exy_factor + ex2_factor);
        curr_camera_patches_grad_ptr[row * ks + col] += grad;
    }
}

__global__ void patches_grad_to_image_kernel(
    const int32_t elements,
    const int32_t H,
    const int32_t W,
    const int32_t ks,
    const float* __restrict__ camera_patches_grad_ptr, // [H, W, ks, ks]
    // output
    float* __restrict__ camera_grad_ptr // [H, W]
) {
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= elements) {
        return;
    }
    const int32_t k2_idx = tid % ks;
    const int32_t k1_idx = (tid / ks) % ks;
    const int32_t w_idx = (tid / (ks * ks)) % W;
    const int32_t h_idx = tid / (W * ks * ks);

    const int32_t cam_i = h_idx + k1_idx - ks / 2;
    const int32_t cam_j = w_idx + k2_idx - ks / 2;
    if (cam_i < 0 || cam_i >= H || cam_j < 0 || cam_j >= W) {
        return;
    }
    atomicAdd(
        camera_grad_ptr + cam_i * W + cam_j, camera_patches_grad_ptr[tid]);
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
            .reshape({H, W, 1});
    Tensor ey2 =
        torch::bmm(
            projector_patch.reshape({H * W, 1, kernel_size * kernel_size}),
            projector_patch.reshape({H * W, kernel_size * kernel_size, 1}))
            .reshape({H, -1, W});

    Tensor full_exy =
        torch::bmm(camera_patch, projector_patch.permute({0, 2, 1}));
    Tensor full_cost_volume =
        (full_exy + EPSILON) / torch::sqrt(torch::bmm(ex2, ey2) + EPSILON);

    Tensor exy = torch::zeros(
        {H, W, D},
        torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    Tensor cost_volume = torch::zeros_like(exy);
    dim3 block_dim(H, crop_w);
    gather_exy_cost_volume_kernel<<<block_dim, D>>>(
        H,
        W,
        D,
        full_exy.data_ptr<float>(),
        full_cost_volume.data_ptr<float>(),
        // output
        exy.data_ptr<float>(),
        cost_volume.data_ptr<float>());

    vector<Tensor> results(6);

    results[0] = ex2;
    results[1] = ey2;
    results[2] = exy;
    results[3] = cost_volume;
    results[4] = camera_patch_mean;
    results[5] = projector_patch_mean;

    return results;
}

Tensor stereo::stereo_matching_backward(
    const Tensor& cost_volume_grad, // [H, W, D]
    const Tensor& camera, // [H, W]
    const Tensor& projector, // [H, W]
    const Tensor& ex2, // [H, W]
    const Tensor& ey2, // [H, W]
    const Tensor& exy, // [H, W, D]
    const Tensor& cam_patch_mean, // [H, W]
    const Tensor& proj_patch_mean, // [H, W]
    const int32_t kernel_size) {
    // check
    CHECK_INPUT(cost_volume_grad);

    // get parameters
    const int32_t H = cost_volume_grad.size(0), W = cost_volume_grad.size(1),
                  D = cost_volume_grad.size(2);
    const int32_t crop_w = W - D;

    assert(kernel_size <= MAX_KERNEL_SIZE);
    Tensor camera_patches_grad = torch::zeros(
        {H, W, kernel_size, kernel_size},
        torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));

    const dim3 dim_block(H, crop_w), thread_block(kernel_size, kernel_size);
    get_patches_grad_kernel<<<dim_block, thread_block>>>(
        H,
        W,
        D,
        kernel_size,
        cost_volume_grad.data_ptr<float>(),
        camera.data_ptr<float>(),
        projector.data_ptr<float>(),
        ex2.data_ptr<float>(),
        ey2.data_ptr<float>(),
        exy.data_ptr<float>(),
        cam_patch_mean.data_ptr<float>(),
        proj_patch_mean.data_ptr<float>(),
        // output
        camera_patches_grad.data_ptr<float>());

    Tensor camera_grad = torch::zeros(
        {H, W},
        torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    const int32_t elements2 = H * W * kernel_size * kernel_size, threads = 1024;
    const int32_t blocks2 = ceil((elements2 - 1) / threads) + 1;
    patches_grad_to_image_kernel<<<blocks2, threads>>>(
        elements2,
        H,
        W,
        kernel_size,
        camera_patches_grad.data_ptr<float>(),
        // output
        camera_grad.data_ptr<float>());

    return camera_grad;
}
