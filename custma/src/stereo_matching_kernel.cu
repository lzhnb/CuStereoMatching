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

__global__ void get_ex2_grad_kernel(
    const int32_t H,
    const int32_t W,
    const int32_t D,
    const int32_t ks,
    const float* __restrict__ cost_volume_grad_ptr, // [H, W - D, W]
    const float* __restrict__ ex2_ptr, // [H, W - D]
    const float* __restrict__ ey2_ptr, // [H, W]
    const float* __restrict__ exy_ptr, // [H, W - D, W]
    // output
    float* __restrict__ ex2_grad_ptr // [H, W - D]
) {
    // the coordinate of pixel
    const int32_t h_idx = threadIdx.x;
    const int32_t w_idx = blockIdx.x;
    const int32_t d_idx = blockIdx.y;

    const float ex2 = ex2_ptr[h_idx * (W - D) + w_idx];
    const float ey2 = ey2_ptr[h_idx * W + d_idx];
    const float exy = exy_ptr[h_idx * (W - D) * W + w_idx * W + d_idx];

    const float deno = 1 / (sqrtf(ex2 * ey2 + EPSILON)),
                deno3 = 1 / powf((sqrtf(ex2 * ey2 + EPSILON)), 3);
    const float cost_volume_grad =
        cost_volume_grad_ptr[h_idx * (W - D) * W + w_idx * W + d_idx];

    const float ex2_grad = cost_volume_grad * -ey2_ptr[h_idx * W + d_idx] *
        (exy + EPSILON) * deno3 / 2;
    atomicAdd(&ex2_grad_ptr[h_idx * (W - D) + w_idx], ex2_grad);
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
        &camera_grad_ptr[cam_i * W + cam_j],
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

    Tensor exy = torch::bmm(camera_patch, projector_patch.permute({0, 2, 1}));

    Tensor cost_volume =
        (exy + EPSILON) / torch::sqrt(torch::bmm(ex2, ey2) + EPSILON);

    vector<Tensor> results(4);

    results[0] = ex2;
    results[1] = ey2;
    results[2] = exy;
    results[3] = cost_volume;

    return results;
}

vector<Tensor> stereo::stereo_matching_backward(
    const Tensor& cost_volume_grad, // [H, W - D, W]
    const Tensor& camera, // [H, W]
    const Tensor& projector, // [H, W]
    const Tensor& ex2,
    const Tensor& ey2,
    const Tensor& exy,
    const int32_t kernel_size) {
    // check
    CHECK_INPUT(cost_volume_grad);
    CHECK_INPUT(camera);
    CHECK_INPUT(projector);
    CHECK_INPUT(ex2);
    CHECK_INPUT(ey2);
    CHECK_INPUT(exy);

    // get parameters
    const int32_t H = camera.size(0), W = camera.size(1),
                  crop_w = cost_volume_grad.size(1);

    /*
        forward:
        CV = (exy + EPS) / sqrt(torch::bmm(ex2, ey2) + EPS)
        backward:
        let `X = torch::bmm(ex2, ey2) + EPS`, so:
        CV = (exy + EPS) / sqrt(X) -> d_CV / d_X = -1/2 * (exy + EPS) /
       (sqrt(X))^3 d_CV / d_ex2 = (d_CV / d_X) * (d_X / d_ex2)
     */
    Tensor factor = torch::sqrt(torch::bmm(ex2, ey2) + EPSILON);
    // Tensor X_grad_temp = -cost_volume_grad * (exy + EPSILON) /
    // torch::pow(factor, 3) / 2.f; Tensor ex2_grad = torch::bmm(X_grad_temp,
    // ey2.permute({0, 2, 1}));
    Tensor ex2_grad = torch::bmm(
        -cost_volume_grad * (exy + EPSILON) / torch::pow(factor, 3) / 2.f,
        ey2.permute({0, 2, 1}));
    Tensor exy_grad = cost_volume_grad / factor;

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

    // combine exy term and ex2 term
    Tensor camera_patch_grad = torch::add(
        torch::bmm(exy_grad, projector_patch),
        2 *
            torch::bmm(
                ex2_grad.reshape({H * W, 1, 1}),
                camera_patch.reshape({H * W, 1, kernel_size * kernel_size}))
                .reshape({H, W, kernel_size * kernel_size}));

    Tensor camera_grad = torch::zeros_like(camera);
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
