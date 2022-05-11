// Copyright 2022 Zhihao Liang
#include "stereo_matching.hpp"

__device__ const float EPSILON = 1e-8;

// KERNELS
__global__ void forward_cost_volume_kernel(
    const int32_t elements,
    const int32_t W,
    const int32_t D,
    const int32_t kernel_size,
    const float* __restrict__ camera_ptr,       // [H, W, kernel_size * kernel_size]
    const float* __restrict__ projector_ptr,    // [H, W, kernel_size * kernel_size]
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

    const int32_t off_w = kernel_size * kernel_size, off_h = kernel_size * kernel_size * W;
    
    // const float* camera_patch_ptr = camera_ptr + h_idx * off_h + (w_idx + D) * off_w;
    const float* camera_patch_ptr = camera_ptr + h_idx * off_h + w_idx * off_w;
    // const float* projector_patch_ptr = projector_ptr + h_idx * off_h + (w_idx + d_idx) * off_w;
    const float* projector_patch_ptr = projector_ptr + h_idx * off_h + d_idx * off_w;

    // loop patch
    float exy = 0.f, ex2 = 0.f, ey2 = 0.f;
#pragma unroll
    for (int32_t i = 0; i < kernel_size * kernel_size; ++i) {
        exy += camera_patch_ptr[i] * projector_patch_ptr[i];
        ex2 += camera_patch_ptr[i] * camera_patch_ptr[i];
        ey2 += projector_patch_ptr[i] * projector_patch_ptr[i];
    }
    cost_volume_ptr[tid] = (exy + EPSILON) / (sqrtf(ex2 * ey2 + EPSILON));
}


__global__ void backward_cost_volume_kernel(
    const int32_t elements,
    const int32_t W,
    const int32_t D,
    const int32_t kernel_size,
    const float* __restrict__ cost_volume_grad, // [H, crop_w, D + 1]
    const float* __restrict__ camera_ptr,       // [H, W, kernel_size * kernel_size]
    const float* __restrict__ projector_ptr,    // [H, W, kernel_size * kernel_size]
    // output
    float* __restrict__ camera_grad_ptr         // [H, W, kernel_size * kernel_size]
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

    const int32_t off_w = kernel_size * kernel_size, off_h = kernel_size * kernel_size * W;
    
    // float* camera_grad_patch_ptr = camera_grad_ptr + h_idx * off_h + (w_idx + D) * off_w;
    // const float* camera_patch_ptr = camera_ptr + h_idx * off_h + (w_idx + D) * off_w;
    // const float* projector_patch_ptr = projector_ptr + h_idx * off_h + (w_idx + d_idx) * off_w;
    float* camera_grad_patch_ptr = camera_grad_ptr + h_idx * off_h + w_idx * off_w;
    const float* camera_patch_ptr = camera_ptr + h_idx * off_h + w_idx * off_w;
    const float* projector_patch_ptr = projector_ptr + h_idx * off_h + d_idx * off_w;

    const float cost_grad = cost_volume_grad[tid];

    // loop patch
    float exy = 0.f, ex2 = 0.f, ey2 = 0.f;
#pragma unroll
    for (int32_t i = 0; i < kernel_size * kernel_size; ++i) {
        exy += camera_patch_ptr[i] * projector_patch_ptr[i];
        ex2 += camera_patch_ptr[i] * camera_patch_ptr[i];
        ey2 += projector_patch_ptr[i] * projector_patch_ptr[i];
    }
#pragma unroll
    for (int32_t i = 0; i < kernel_size * kernel_size; ++i) {
        // exy term
        float exy_factor = projector_patch_ptr[i] / (sqrtf(ex2 * ey2 + EPSILON));
        atomicAdd(camera_grad_patch_ptr + i, cost_grad * exy_factor);
        // ex2 term
        float ex2_factor = -(ey2 * camera_patch_ptr[i] * (exy + EPSILON)) / powf((sqrtf(ex2 * ey2 + EPSILON)), 3);
        atomicAdd(camera_grad_patch_ptr + i, cost_grad * ex2_factor);
    }  
}


Tensor stereo::stereo_matching_forward_wrapper(
    const Tensor& camera_flatten,    // [H, W, p*p]
    const Tensor& projector_flatten, // [H, W, p*p]
    const int32_t D,
    const int32_t kernel_size,
    // output
    Tensor& disparity    // [H, W]
) {

    const int32_t H = camera_flatten.size(0), W = camera_flatten.size(1);
    // const int32_t crop_w = W - D;
    // const int32_t elements = H * crop_w * (D + 1), threads = 1024;
    const int32_t elements = H * W * W, threads = 1024;
    const int32_t blocks = ceil((elements - 1) / threads) + 1;

    // Tensor cost_volume = torch::zeros({H, crop_w, D + 1},
    //         torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));

    Tensor cost_volume = torch::zeros({H, W, W},
            torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));

    forward_cost_volume_kernel<<<blocks, threads>>>(
        elements,
        W,
        D,
        kernel_size,
        camera_flatten.data_ptr<float>(),
        projector_flatten.data_ptr<float>(),
        // output
        cost_volume.data_ptr<float>()
    );
    
    return cost_volume;
}

void stereo::stereo_matching_backward_wrapper(
    const Tensor& cost_volume_grad,    // [H, crop_w, D + 1]
    const Tensor& camera_flatten,    // [H, W, p*p]
    const Tensor& projector_flatten, // [H, W, p*p]
    const int32_t kernel_size,
    // output
    Tensor& camera_grad    // [H, W, kernel_size, kernel_size]
) {
    // get parameters
    const int32_t H = cost_volume_grad.size(0), W = cost_volume_grad.size(1), D = cost_volume_grad.size(2);
    // const int32_t H = cost_volume_grad.size(0), crop_w = cost_volume_grad.size(1), D = cost_volume_grad.size(2) - 1;
    // const int32_t W = crop_w + D;
    const int32_t elements = H * W * W, threads = 1024;
    // const int32_t elements = H * crop_w * (D + 1), threads = 1024;
    const int32_t blocks = ceil((elements - 1) / threads) + 1;

    backward_cost_volume_kernel<<<blocks, threads>>>(
        elements,
        W,
        D,
        kernel_size,
        cost_volume_grad.data_ptr<float>(),
        camera_flatten.data_ptr<float>(),
        projector_flatten.data_ptr<float>(),
        // output
        camera_grad.data_ptr<float>()
    );
}
