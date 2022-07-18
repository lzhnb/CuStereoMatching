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
__global__ void self_cov_kernel(
    const int32_t elements,
    const int32_t H,
    const int32_t W,
    const int32_t D,
    const int32_t ks,
    const float* __restrict__ inputs_ptr, // [H, W]
    // output
    float* __restrict__ mean_buffer, // [H, W - D]
    float* __restrict__ self_cov_ptr // [H, W - D]
) {
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= elements) {
        return;
    }
    // the coordinate of pixel
    const int32_t w_idx = tid % (W - D);
    const int32_t h_idx = tid / (W - D);

    // fullfill the mean buffer
    float temp_mean = 0;
    for (int32_t i = 0; i < ks; ++i) {
        for (int32_t j = 0; j < ks; ++j) {
            const int32_t patch_i = h_idx + i - ks / 2;
            const int32_t patch_j = w_idx + D + j - ks / 2;
            const float val = query_ij(inputs_ptr, H, W, patch_i, patch_j);
            temp_mean += val;
        }
    }
    temp_mean /= (ks * ks);
    mean_buffer[h_idx * (W - D) + w_idx] = temp_mean;

    // get the self cov
    for (int32_t i = 0; i < ks; ++i) {
        for (int32_t j = 0; j < ks; ++j) {
            const int32_t patch_i = h_idx + i - ks / 2;
            const int32_t patch_j = w_idx + D + j - ks / 2;
            const float val = query_ij(inputs_ptr, H, W, patch_i, patch_j);
            const float norm_val = val - temp_mean;
            self_cov_ptr[h_idx * (W - D) + w_idx] += powf(norm_val, 2);
        }
    }
}

__global__ void cross_cov_kernel(
    const int32_t elements,
    const int32_t H,
    const int32_t W,
    const int32_t D,
    const int32_t ks,
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

    const int32_t d_idx = tid % W;
    const int32_t w_idx = (tid / W) % (W - D);
    const int32_t h_idx = tid / (W * (W - D));

    for (int32_t i = 0; i < ks; ++i) {
        for (int32_t j = 0; j < ks; ++j) {
            const float cam_mean = cam_mean_buffer[h_idx * (W - D) + w_idx];
            const float proj_mean = proj_mean_buffer[h_idx * W + d_idx];
            const int32_t cam_patch_i = h_idx + i - ks / 2;
            const int32_t cam_patch_j = w_idx + D + j - ks / 2;
            const float cam_val =
                query_ij(cam_ptr, H, W, cam_patch_i, cam_patch_j) - cam_mean;

            const int32_t proj_patch_i = h_idx + i - ks / 2;
            const int32_t proj_patch_j = d_idx + j - ks / 2;
            const float proj_val =
                query_ij(proj_ptr, H, W, proj_patch_i, proj_patch_j) -
                proj_mean;

            cross_cov_ptr[h_idx * (W - D) * W + w_idx * W + d_idx] +=
                cam_val * proj_val;
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

__global__ void get_cross_grad_kernel(
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
    float* __restrict__ camera_grad_ptr, // [H, W]
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

    const float ex2 = ex2_ptr[h_idx * (W - D) + w_idx];
    const float ey2 = ey2_ptr[h_idx * W + d_idx];
    const float exy = exy_ptr[h_idx * (W - D) * W + w_idx * W + d_idx];

    // const float factor = (sqrtf(ex2 * ey2 + EPSILON));
    const float deno = 1 / (sqrtf(ex2 * ey2 + EPSILON)), deno3 = 1 / powf((sqrtf(ex2 * ey2 + EPSILON)), 3);
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

    // skip while outside the image
    if (cam_patch_i < 0 || cam_patch_i >= H || cam_patch_j < 0 ||
        cam_patch_j >= W) {
        return;
    }

    /* ex2 term */
    __syncthreads();
    const float cam_val = query_ij(camera_ptr, H, W, cam_patch_i, cam_patch_j);
    const float cam_grad_ex2_term = 2 * (cam_val - ex2_mean) * ex2_grad;
    atomicAdd(
        &camera_grad_ptr[cam_patch_i * W + cam_patch_j], cam_grad_ex2_term);

    /* exy term */
    __syncthreads();
    const float proj_val =
        query_ij(projector_ptr, H, W, proj_patch_i, proj_patch_j);
    const float cam_grad_exy_term = (proj_val - ey2_mean) * exy_grad;
    atomicAdd(
        &camera_grad_ptr[cam_patch_i * W + cam_patch_j], cam_grad_exy_term);

    __syncthreads();
}

__global__ void exy_grad_to_image_kernel2(
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

    const float exy_grad =
        exy_grad_ptr[h_idx * (W - D) * W + w_idx * W + d_idx];

    const int32_t cam_patch_i = h_idx + i - kernel_size / 2;
    const int32_t cam_patch_j = w_idx + D + j - kernel_size / 2;
    const int32_t proj_patch_i = h_idx + i - kernel_size / 2;
    const int32_t proj_patch_j = d_idx + j - kernel_size / 2;
    const float proj_val =
        query_ij(projector_ptr, H, W, proj_patch_i, proj_patch_j);
    const float grad = (proj_val - ey2_mean_ptr[h_idx * W + d_idx]) * exy_grad;

    if (cam_patch_i < 0 || cam_patch_i >= H || cam_patch_j < 0 ||
        cam_patch_j >= W) {
        return;
    }
    atomicAdd(&camera_grad_ptr[cam_patch_i * W + cam_patch_j], grad);
    __syncthreads();
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

    /* self cov */
    Tensor ex2_mean = torch::zeros(
        {H, crop_w},
        torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    Tensor ex2 = torch::zeros(
        {H, crop_w},
        torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    const int32_t ex2_elements = H * crop_w;
    const int32_t ex2_blocks = ceil((ex2_elements - 1) / threads) + 1;
    self_cov_kernel<<<ex2_blocks, threads>>>(
        ex2_elements,
        H,
        W,
        D,
        kernel_size,
        camera.data_ptr<float>(),
        // output
        ex2_mean.data_ptr<float>(),
        ex2.data_ptr<float>());

    Tensor ey2_mean = torch::zeros(
        {H, W},
        torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    Tensor ey2 = torch::zeros(
        {H, W},
        torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    const int32_t ey2_elements = H * crop_w;
    const int32_t ey2_blocks = ceil((ey2_elements - 1) / threads) + 1;
    self_cov_kernel<<<ey2_blocks, threads>>>(
        ey2_elements,
        H,
        W,
        0,
        kernel_size,
        projector.data_ptr<float>(),
        // output
        ey2_mean.data_ptr<float>(),
        ey2.data_ptr<float>());

    /* cross cov */
    Tensor exy = torch::zeros(
        {H, crop_w, W},
        torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));

    // NOTE: atomicAdd is too slow for sync while the blocks are too many
    const int32_t exy_elements = H * crop_w * W;
    const int32_t exy_blocks = ceil((exy_elements - 1) / threads) + 1;
    cross_cov_kernel<<<exy_blocks, threads>>>(
        exy_elements,
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

    const int32_t elements = H * crop_w * W;
    const int32_t blocks = ceil((elements - 1) / threads) + 1;
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

    Tensor ex2_grad;
    Tensor exy_grad;
    if (record) {
        ex2_grad = torch::zeros_like(ex2);
        exy_grad = torch::zeros_like(exy);
    }
    Tensor camera_grad = torch::zeros_like(camera);

    dim3 dim_block(kernel_size, kernel_size);
    dim3 dim_grid(H, crop_w, W);
    get_cross_grad_kernel<<<dim_grid, dim_block>>>(
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
        camera_grad.data_ptr<float>(),
        record ? ex2_grad.data_ptr<float>() : nullptr,
        record ? exy_grad.data_ptr<float>() : nullptr);

    vector<Tensor> results(3);

    results[0] = camera_grad;
    results[1] = ex2_grad;
    results[2] = exy_grad;

    return results;
}
