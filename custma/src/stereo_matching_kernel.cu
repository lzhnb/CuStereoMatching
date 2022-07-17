// Copyright 2022 Zhihao Liang
#include "stereo_matching.hpp"

const float EPSILON = 1e-8;

__device__ float query_ij(
    const float* __restrict__ val_ptr, // [H, W]
    const int32_t H,
    const int32_t W,
    const int32_t i,
    const int32_t j) {
    return (i < 0 || i >= H || j < 0 || j >= W) ? 0.f : val_ptr[i * W + j];
}

/* NOTE: now ignore the crop_w and D */

// KERNELS
__global__ void self_cov_kernel(
    const int32_t H,
    const int32_t W,
    const int32_t D,
    const int32_t kernel_size,
    const float* __restrict__ inputs_ptr, // [H, W]
    // output
    float* __restrict__ mean_buffer, // [H, W - D]
    float* __restrict__ self_cov_ptr // [H, W - D]
) {
    // the coordinate of pixel
    const int32_t h_idx = blockIdx.x;
    const int32_t w_idx = blockIdx.y;
    // relative coordinate in the patch
    const int32_t i = threadIdx.x;
    const int32_t j = threadIdx.y;

    const int32_t patch_i = h_idx + i - kernel_size / 2;
    const int32_t patch_j = w_idx + D + j - kernel_size / 2;
    const float val = query_ij(inputs_ptr, H, W, patch_i, patch_j);

    atomicAdd(&mean_buffer[h_idx * (W - D) + w_idx], val);
    __syncthreads();

    const float _mean_tmp =
        mean_buffer[h_idx * (W - D) + w_idx] / (kernel_size * kernel_size);
    const float norm_val = val - _mean_tmp;
    atomicAdd(&self_cov_ptr[h_idx * (W - D) + w_idx], norm_val * norm_val);
    __syncthreads();
}

__global__ void cross_cov_kernel(
    const int32_t elements,
    const int32_t H,
    const int32_t W,
    const int32_t D,
    const int32_t kernel_size,
    const float* __restrict__ cam_ptr, // [H, W]
    const float* __restrict__ proj_ptr, // [H, W]
    const float* __restrict__ cam_mean_buffer, // [H, W - D]
    const float* __restrict__ proj_mean_buffer, // [H, W]
    // output
    float* __restrict__ cross_cov_ptr // [H, W - D, W]
) {
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= elements) {
        return;
    }
    // the coordinate of pixel
    const int32_t d_idx = tid % W;
    const int32_t w_idx = (tid / W) % (W - D);
    const int32_t h_idx = tid / (W * (W - D));

    const float cam_mean = cam_mean_buffer[h_idx * (W - D) + w_idx];
    const float proj_mean = proj_mean_buffer[h_idx * W + d_idx];
#pragma unroll
    for (int32_t i = 0; i < kernel_size; ++i) {
        for (int32_t j = 0; j < kernel_size; ++j) {
            const int32_t cam_patch_i = h_idx + i - kernel_size / 2;
            const int32_t cam_patch_j = w_idx + D + j - kernel_size / 2;
            const float cam_val =
                query_ij(cam_ptr, H, W, cam_patch_i, cam_patch_j) - cam_mean;

            const int32_t proj_patch_i = h_idx + i - kernel_size / 2;
            const int32_t proj_patch_j = d_idx + j - kernel_size / 2;
            const float proj_val =
                query_ij(proj_ptr, H, W, proj_patch_i, proj_patch_j) -
                proj_mean;

            cross_cov_ptr[tid] += cam_val * proj_val;
        }
    }
}

__global__ void cost_volume_kernel(
    const int32_t elements,
    const int32_t H,
    const int32_t W,
    const int32_t D,
    const float* __restrict__ ex2_ptr, // [H, W - D]
    const float* __restrict__ ey2_ptr, // [H, W]
    const float* __restrict__ exy_ptr, // [H, W - D, W]
    // output
    float* __restrict__ cost_volume_ptr // [H, W - D, W]
) {
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= elements) {
        return;
    }
    // the coordinate of pixel
    const int32_t d_idx = tid % W;
    const int32_t w_idx = (tid / W) % (W - D);
    const int32_t h_idx = tid / (W * (W - D));

    const float ex2 = ex2_ptr[h_idx * (W - D) + w_idx];
    const float ey2 = ey2_ptr[h_idx * W + d_idx];
    const float exy = exy_ptr[tid];

    cost_volume_ptr[tid] = (exy + EPSILON) / (sqrtf(ex2 * ey2 + EPSILON));
}

__global__ void get_self_grad_kernel(
    const int32_t H,
    const int32_t W,
    const int32_t D,
    const int32_t kernel_size,
    const float* __restrict__ inputs_ptr, // [H, W]
    const float* __restrict__ cov_mean_ptr, // [H, W - D]
    const float* __restrict__ cov_grad_ptr, // [H, W - D]
    const float* __restrict__ cov_ptr, // [H, W - D]
    // output
    float* __restrict__ input_grad_ptr // [H, W]
) {
    // the coordinate of pixel
    const int32_t h_idx = blockIdx.x;
    const int32_t w_idx = blockIdx.y;
    // relative coordinate in the patch
    const int32_t i = threadIdx.x;
    const int32_t j = threadIdx.y;

    const int32_t patch_i = h_idx + i - kernel_size / 2;
    const int32_t patch_j = w_idx + D + j - kernel_size / 2;

    if (patch_i < 0 || patch_i >= H || patch_j < 0 || patch_j >= W) {
        return;
    }

    const float temp_input = query_ij(inputs_ptr, H, W, patch_i, patch_j);
    const float temp_mean = cov_mean_ptr[h_idx * (W - D) + w_idx];
    const float temp_grad = cov_grad_ptr[h_idx * (W - D) + w_idx];
    const float factor =
        2 * (temp_input - temp_mean / (kernel_size * kernel_size));

    atomicAdd(&input_grad_ptr[patch_i * W + patch_j], factor * temp_grad);
    __syncthreads();
}

__global__ void get_cross_grad_kernel(
    const int32_t elements,
    const int32_t H,
    const int32_t W,
    const int32_t D,
    const int32_t kernel_size,
    const float* __restrict__ projector_ptr, // [H, W]
    const float* __restrict__ cost_volume_grad_ptr, // [H, W - D, W]
    const float* __restrict__ ex2_ptr, // [H, W - D]
    const float* __restrict__ ey2_ptr, // [H, W]
    const float* __restrict__ exy_ptr, // [H, W - D, W]
    const float* __restrict__ ex2_mean_ptr, // [H, W - D]
    const float* __restrict__ ey2_mean_ptr, // [H, W]
    // output
    float* __restrict__ ex2_grad_ptr, // [H, W - D]
    float* __restrict__ exy_grad_ptr, // [H, W - D, W]
    float* __restrict__ camera_grad_ptr // [H, W]
) {
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= elements) {
        return;
    }
    // the coordinate of pixel
    const int32_t d_idx = tid % W;
    const int32_t w_idx = (tid / W) % (W - D);
    const int32_t h_idx = tid / (W * (W - D));

    const float ex2 = ex2_ptr[h_idx * (W - D) + w_idx];
    const float ey2 = ey2_ptr[h_idx * W + d_idx];
    const float exy = exy_ptr[tid];

    const float temp_cost_volume_grad = cost_volume_grad_ptr[tid];

    const float factor = (sqrtf(ex2 * ey2 + EPSILON));
    const float temp_exy_grad = temp_cost_volume_grad / factor;
    exy_grad_ptr[tid] = temp_exy_grad;
    const float temp_ex2_grad = -ey2_ptr[h_idx * W + d_idx] *
        (exy_ptr[tid] + EPSILON) / (2 * powf(factor, 3));
    atomicAdd(&ex2_grad_ptr[h_idx * (W - D) + w_idx], temp_ex2_grad);
    __syncthreads();
}


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
    const float temp_proj =
        query_ij(projector_ptr, H, W, proj_patch_i, proj_patch_j);
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
    const float* __restrict__ ex2_grad_ptr, // [H, W - D, W]
    const float* __restrict__ camera_ptr, // [H, W]
    const float* __restrict__ ex2_mean_ptr, // [H, W]
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
    const float val = query_ij(camera_ptr, H, W, patch_i, patch_j);

    if (patch_i < 0 || patch_i >= H || patch_j < 0 || patch_j >= W) {
        return;
    }
    atomicAdd(
        &camera_grad_ptr[patch_i * W + patch_j],
        2 * (val - ex2_mean_ptr[h_idx * (W - D) + w_idx]));
    __syncthreads();
}


vector<Tensor> stereo::stereo_matching_forward_wrapper(
    const Tensor& camera, // [H, W]
    const Tensor& projector, // [H, W]
    const int32_t D,
    const int32_t kernel_size,
    // output
    Tensor& disparity // [H, W]
) {
    const int32_t H = camera.size(0), W = camera.size(1);
    const int32_t crop_w = W - D;
    /* self cov */
    Tensor ex2_mean = torch::zeros(
        {H, crop_w},
        torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    Tensor ex2 = torch::zeros(
        {H, crop_w},
        torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    dim3 dim_block(kernel_size, kernel_size);
    dim3 ex2_dim_grid(H, crop_w);
    self_cov_kernel<<<ex2_dim_grid, dim_block>>>(
        H,
        W,
        D,
        kernel_size,
        camera.data_ptr<float>(),
        // output
        ex2_mean.data_ptr<float>(),
        ex2.data_ptr<float>());
    ex2_mean /= (kernel_size * kernel_size);

    Tensor ey2_mean = torch::zeros(
        {H, W},
        torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    Tensor ey2 = torch::zeros(
        {H, W},
        torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    dim3 ey2_dim_grid(H, W);
    self_cov_kernel<<<ey2_dim_grid, dim_block>>>(
        H,
        W,
        0,
        kernel_size,
        projector.data_ptr<float>(),
        // output
        ey2_mean.data_ptr<float>(),
        ey2.data_ptr<float>());
    ey2_mean /= (kernel_size * kernel_size);

    /* cross cov */
    Tensor exy = torch::zeros(
        {H, crop_w, W},
        torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    const int32_t elements = H * crop_w * W, threads = 1024;
    const int32_t blocks = ceil((elements - 1) / threads) + 1;

    // NOTE: atomicAdd is too slow for sync while the blocks are too many
    cross_cov_kernel<<<blocks, threads>>>(
        elements,
        H,
        W,
        0,
        kernel_size,
        camera.data_ptr<float>(),
        projector.data_ptr<float>(),
        ex2_mean.data_ptr<float>(),
        ey2_mean.data_ptr<float>(),
        // output
        exy.data_ptr<float>());

    Tensor cost_volume = torch::zeros(
        {H, crop_w, W},
        torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    cost_volume_kernel<<<blocks, threads>>>(
        elements,
        H,
        W,
        D,
        ex2.data_ptr<float>(),
        ey2.data_ptr<float>(),
        exy.data_ptr<float>(),
        // output
        cost_volume.data_ptr<float>());

    vector<Tensor> results(6);

    results[0] = ex2;
    results[1] = ey2;
    results[2] = exy;
    results[3] = ex2_mean;
    results[4] = ey2_mean;
    results[5] = cost_volume;

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
    const int32_t kernel_size) {
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

    Tensor ex2_grad = torch::zeros_like(ex2);
    Tensor exy_grad = torch::zeros_like(exy);
    Tensor camera_grad = torch::zeros_like(camera);

    const int32_t elements = H * crop_w * W, threads = 1024;
    const int32_t blocks = ceil((elements - 1) / threads) + 1;
    get_cross_grad_kernel<<<blocks, threads>>>(
        elements,
        H,
        W,
        D,
        kernel_size,
        projector.data_ptr<float>(),
        cost_volume_grad.data_ptr<float>(),
        ex2.data_ptr<float>(),
        ey2.data_ptr<float>(),
        exy.data_ptr<float>(),
        ex2_mean.data_ptr<float>(),
        ey2_mean.data_ptr<float>(),
        // output
        ex2_grad.data_ptr<float>(),
        exy_grad.data_ptr<float>(),
        camera_grad.data_ptr<float>());

    // exy term
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

    // ex2 term
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

    vector<Tensor> results(3);

    results[0] = ex2_grad;
    results[1] = exy_grad;
    results[2] = camera_grad;

    return results;
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
    // Tensor camera_mean_grad = torch::zeros(
    //     {H, crop_w},
    //     torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));

    const int32_t elements = H * crop_w * W * kernel_size * kernel_size, threads = 1024;
    const int32_t blocks = ceil((elements - 1) / threads) + 1;
    
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
