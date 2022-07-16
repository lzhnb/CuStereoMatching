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

/* NOTE: now ignore the crop_w and D */

// KERNELS
__global__ void self_cov_kernel(
    const int32_t H,
    const int32_t W,
    const int32_t D,
    const int32_t kernel_size,
    const float* __restrict__ inputs_ptr, // [H, W]
    // output
    float* __restrict__ mean_buffer,    // [H, W - D]
    float* __restrict__ self_cov_ptr    // [H, W - D]
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

    const float _mean_tmp = mean_buffer[h_idx * (W - D) + w_idx];
    const float norm_val = val - _mean_tmp / (kernel_size * kernel_size);
    atomicAdd(&self_cov_ptr[h_idx * (W - D) + w_idx], norm_val * norm_val);
}


__global__ void cross_cov_kernel(
    const int32_t elements,
    const int32_t H,
    const int32_t W,
    const int32_t D,
    const int32_t kernel_size,
    const float* __restrict__ cam_ptr,  // [H, W]
    const float* __restrict__ proj_ptr, // [H, W]
    const float* __restrict__ cam_mean_buffer,  // [H, W - D]
    const float* __restrict__ proj_mean_buffer, // [H, W]
    // output
    float* __restrict__ cross_cov_ptr   // [H, W - D, W]
) {
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= elements) { return; }
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
            const float cam_val = query_ij(cam_ptr, H, W, cam_patch_i, cam_patch_j) - cam_mean;

            const int32_t proj_patch_i = h_idx + i - kernel_size / 2;
            const int32_t proj_patch_j = d_idx + j - kernel_size / 2;
            const float proj_val = query_ij(proj_ptr, H, W, proj_patch_i, proj_patch_j) - proj_mean;

            cross_cov_ptr[tid] += cam_val * proj_val;
        }
    }
}


__global__ void cost_volume_kernel(
    const int32_t elements,
    const int32_t H,
    const int32_t W,
    const int32_t D,
    const float* __restrict__ ex2_ptr,  // [H, W - D]
    const float* __restrict__ ey2_ptr,  // [H, W]
    const float* __restrict__ exy_ptr,  // [H, W - D, W]
    // output
    float* __restrict__ cost_volume_ptr // [H, W - D, W]
) {
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= elements) { return; }
    // the coordinate of pixel
    const int32_t d_idx = tid % W;
    const int32_t w_idx = (tid / W) % (W - D);
    const int32_t h_idx = tid / (W * (W - D));

    const float ex2 = ex2_ptr[h_idx * (W - D) + w_idx];
    const float ey2 = ey2_ptr[h_idx * W + d_idx];
    const float exy = exy_ptr[tid];

    cost_volume_ptr[tid] = (exy + EPSILON) / (sqrtf(ex2 * ey2 + EPSILON));
}


__global__ void forward_cost_volume_kernel(
    const int32_t elements,
    const int32_t H,
    const int32_t W,
    const int32_t D,
    const int32_t kernel_size,
    const float* __restrict__ camera_ptr,       // [H, W]
    const float* __restrict__ projector_ptr,    // [H, W]
    // output
    float* __restrict__ cost_volume_ptr         // [H, crop_w, D + 1]
) {
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= elements) { return; }
    // const int32_t crop_w = W - D;
    // const int32_t d_idx = tid % (D + 1);
    // const int32_t w_idx = (tid / (D + 1)) % crop_w;
    // const int32_t h_idx = tid / ((D + 1) * crop_w);

    const int32_t d_idx = tid % W;
    const int32_t w_idx = (tid / W) % W;
    const int32_t h_idx = tid / (W * W);

    // loop patch to get the mean value
    float cam_mean = 0, proj_mean = 0;
#pragma unroll
    for (int32_t i = 0; i < kernel_size; ++i) {
        for (int32_t j = 0; j < kernel_size; ++j) {
            const int32_t cam_proj_i = h_idx + i - kernel_size / 2;
            const int32_t cam_j = w_idx + j - kernel_size / 2;
            const int32_t proj_j = d_idx + j - kernel_size / 2;
            const float cam = query_ij(camera_ptr, H, W, cam_proj_i, cam_j);
            const float proj = query_ij(projector_ptr, H, W, cam_proj_i, proj_j);
            cam_mean += cam;
            proj_mean += proj;
        }
    }
    cam_mean /= (kernel_size * kernel_size);
    proj_mean /= (kernel_size * kernel_size);

    float exy = 0, ex2 = 0, ey2 = 0;
#pragma unroll
    for (int32_t i = 0; i < kernel_size; ++i) {
        for (int32_t j = 0; j < kernel_size; ++j) {
            const int32_t cam_proj_i = h_idx + i - kernel_size / 2;
            const int32_t cam_j = w_idx + j - kernel_size / 2;
            const int32_t proj_j = d_idx + j - kernel_size / 2;
            const float cam = query_ij(camera_ptr, H, W, cam_proj_i, cam_j) - cam_mean;
            const float proj = query_ij(projector_ptr, H, W, cam_proj_i, proj_j) - proj_mean;

            exy += cam * proj;
            ex2 += cam * cam;
            ey2 += proj * proj;
        }
    }
    cost_volume_ptr[tid] = (exy + EPSILON) / (sqrtf(ex2 * ey2 + EPSILON));
}


__global__ void get_patches_grad_kernel(
    const int32_t elements,
    const int32_t H,
    const int32_t W,
    const int32_t D,
    const int32_t kernel_size,
    const float* __restrict__ cost_volume_grad, // [H, crop_w, D + 1]
    const float* __restrict__ camera_ptr,       // [H, W]
    const float* __restrict__ projector_ptr,    // [H, W]
    // output
    float* __restrict__ camera_patches_grad_ptr // [H, W, kernel_size, kernel_size]
) {
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= elements) { return; }
    // const int32_t crop_w = W - D;
    // const int32_t d_idx = tid % (D + 1);
    // const int32_t w_idx = (tid / (D + 1)) % crop_w;
    // const int32_t h_idx = tid / ((D + 1) * crop_w);
    const int32_t d_idx = tid % W;
    const int32_t w_idx = (tid / W) % W;
    const int32_t h_idx = tid / (W * W);

    // loop patch to get the mean value
    float cam_mean = 0, proj_mean = 0;
#pragma unroll
    for (int32_t i = 0; i < kernel_size; ++i) {
        for (int32_t j = 0; j < kernel_size; ++j) {
            const int32_t cam_proj_i = h_idx + i - kernel_size / 2;
            const int32_t cam_j = w_idx + j - kernel_size / 2;
            const int32_t proj_j = d_idx + j - kernel_size / 2;
            const float cam = query_ij(camera_ptr, H, W, cam_proj_i, cam_j);
            const float proj = query_ij(projector_ptr, H, W, cam_proj_i, proj_j);
            cam_mean += cam;
            proj_mean += proj;
        }
    }
    cam_mean /= (kernel_size * kernel_size);
    proj_mean /= (kernel_size * kernel_size);

    float exy = 0, ex2 = 0, ey2 = 0;
#pragma unroll
    for (int32_t i = 0; i < kernel_size; ++i) {
        for (int32_t j = 0; j < kernel_size; ++j) {
            const int32_t cam_proj_i = h_idx + i - kernel_size / 2;
            const int32_t cam_j = w_idx + j - kernel_size / 2;
            const int32_t proj_j = d_idx + j - kernel_size / 2;
            const float cam = query_ij(camera_ptr, H, W, cam_proj_i, cam_j) - cam_mean;
            const float proj = query_ij(projector_ptr, H, W, cam_proj_i, proj_j) - proj_mean;

            exy += cam * proj;
            ex2 += cam * cam;
            ey2 += proj * proj;
        }
    }

    const float cost_grad = cost_volume_grad[tid];
    const int32_t off_w = kernel_size * kernel_size, off_h = kernel_size * kernel_size * W;
    float* curr_camera_patches_grad_ptr = camera_patches_grad_ptr + h_idx * off_h + w_idx * off_w;

    // calculate 1 time to save time
    const float deno = 1 / (sqrtf(ex2 * ey2 + EPSILON)), deno3 = 1 / powf((sqrtf(ex2 * ey2 + EPSILON)), 3);
#pragma unroll
    for (int32_t i = 0; i < kernel_size; ++i) {
        for (int32_t j = 0; j < kernel_size; ++j) {
            const int32_t cam_proj_i = h_idx + i - kernel_size / 2;
            const int32_t cam_j = w_idx + j - kernel_size / 2;
            const int32_t proj_j = d_idx + j - kernel_size / 2;
            const float cam = query_ij(camera_ptr, H, W, cam_proj_i, cam_j) - cam_mean;
            const float proj = query_ij(projector_ptr, H, W, cam_proj_i, proj_j) - proj_mean;
            // exy term
            float exy_factor = proj * deno;
            // ex2 term
            float ex2_factor = -(ey2 * cam * (exy + EPSILON)) * deno3;
            const float grad = cost_grad * (exy_factor + ex2_factor);
            atomicAdd(curr_camera_patches_grad_ptr + i * kernel_size + j, grad);
        }
    }
}


__global__ void patches_grad_to_image_kernel(
    const int32_t elements,
    const int32_t H,
    const int32_t W,
    const int32_t kernel_size,
    const float* __restrict__ camera_patches_grad_ptr,  // [H, W, kernel_size, kernel_size]
    // output
    float* __restrict__ camera_grad_ptr         // [H, W]
) {
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= elements) { return; }
    // const int32_t crop_w = W - D;
    // const int32_t d_idx = tid % (D + 1);
    // const int32_t w_idx = (tid / (D + 1)) % crop_w;
    // const int32_t h_idx = tid / ((D + 1) * crop_w);
    const int32_t k2_idx = tid % kernel_size;
    const int32_t k1_idx = (tid / kernel_size) % kernel_size;
    const int32_t w_idx = (tid / (kernel_size * kernel_size)) % W;
    const int32_t h_idx = tid / (W * kernel_size * kernel_size);

    const int32_t cam_i = h_idx + k1_idx - kernel_size / 2;
    const int32_t cam_j = w_idx + k2_idx - kernel_size / 2;
    if (cam_i < 0 || cam_i >= H || cam_j < 0 || cam_j >= W) { return; }
    atomicAdd(camera_grad_ptr + cam_i * W + cam_j, camera_patches_grad_ptr[tid]);
}


vector<Tensor> stereo::stereo_matching_forward_wrapper(
    const Tensor& camera,    // [H, W]
    const Tensor& projector, // [H, W]
    const int32_t D,
    const int32_t kernel_size,
    // output
    Tensor& disparity    // [H, W]
) {
    const int32_t H = camera.size(0), W = camera.size(1);
    const int32_t crop_w = W - D;
    /* self cov */
    Tensor ex2_mean_buffer = torch::zeros({H, crop_w},
            torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    Tensor ex2 = torch::zeros({H, crop_w},
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
        ex2_mean_buffer.data_ptr<float>(),
        ex2.data_ptr<float>()
    );
    ex2_mean_buffer /= (kernel_size * kernel_size);

    Tensor ey2_mean_buffer = torch::zeros({H, crop_w},
            torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    Tensor ey2 = torch::zeros({H, W},
            torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    dim3 ey2_dim_grid(H, W);
    self_cov_kernel<<<ey2_dim_grid, dim_block>>>(
        H,
        W,
        0,
        kernel_size,
        projector.data_ptr<float>(),
        // output
        ey2_mean_buffer.data_ptr<float>(),
        ey2.data_ptr<float>()
    );
    ey2_mean_buffer /= (kernel_size * kernel_size);

    /* cross cov */
    Tensor exy = torch::zeros({H, crop_w, W},
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
        ex2_mean_buffer.data_ptr<float>(),
        ey2_mean_buffer.data_ptr<float>(),
        // output
        exy.data_ptr<float>()
    );

    Tensor cost_volume = torch::zeros({H, crop_w, W},
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
        cost_volume.data_ptr<float>()
    );
    
    vector<Tensor> results(6);

    results[0] = ex2;
    results[1] = ey2;
    results[2] = exy;
    results[3] = ex2_mean_buffer;
    results[4] = ey2_mean_buffer;
    results[5] = cost_volume;

    return results;
}

void stereo::stereo_matching_backward_wrapper(
    const Tensor& cost_volume_grad,  // [H, crop_w, D + 1]
    const Tensor& camera,    // [H, W]
    const Tensor& projector, // [H, W]
    const int32_t kernel_size,
    // output
    Tensor& camera_grad      // [H, W]
) {
    // get parameters
    const int32_t H = cost_volume_grad.size(0), W = cost_volume_grad.size(1), D = cost_volume_grad.size(2);
    // const int32_t H = cost_volume_grad.size(0), crop_w = cost_volume_grad.size(1), D = cost_volume_grad.size(2) - 1;
    // const int32_t W = crop_w + D;
    const int32_t elements1 = H * W * W, threads = 1024;
    // const int32_t elements1 = H * crop_w * (D + 1), threads = 1024;

    Tensor camera_patches_grad = torch::zeros({H, W, kernel_size, kernel_size},
            torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));

    const int32_t blocks1 = ceil((elements1 - 1) / threads) + 1;
    get_patches_grad_kernel<<<blocks1, threads>>>(
        elements1,
        H,
        W,
        D,
        kernel_size,
        cost_volume_grad.data_ptr<float>(),
        camera.data_ptr<float>(),
        projector.data_ptr<float>(),
        // output
        camera_patches_grad.data_ptr<float>()
    );

    const int32_t elements2 = H * W * kernel_size * kernel_size;
    const int32_t blocks2 = ceil((elements2 - 1) / threads) + 1;
    patches_grad_to_image_kernel<<<blocks2, threads>>>(
        elements2,
        H,
        W,
        kernel_size,
        camera_patches_grad.data_ptr<float>(),
        // output
        camera_grad.data_ptr<float>()
    );
}
