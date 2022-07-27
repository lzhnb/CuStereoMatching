// Copyright 2022 Zhihao Liang
#include "stereo_matching.hpp"

#define MAX_KERNEL_SIZE 15

__device__ const float EPSILON = 1e-8;

__device__ float query_ij(
    const float *__restrict__ img_ptr, // [H, W]
    const int32_t H,
    const int32_t W,
    const int32_t i,
    const int32_t j)
{
    return (i < 0 || i >= H || j < 0 || j >= W) ? 0.f : img_ptr[i * W + j];
}

/* NOTE: now ignore the crop_w and D */

// KERNELS
__global__ void forward_cost_volume_kernel(
    const int32_t H,
    const int32_t W,
    const int32_t D,
    const int32_t kernel_size,
    const float* __restrict__ cam_ptr,       // [H, W]
    const float* __restrict__ proj_ptr,    // [H, W]
    // output
    float* __restrict__ cost_volume_ptr         // [H, crop_w, D + 1]
) {
    const int32_t h_idx = blockIdx.x,
                  w_idx = blockIdx.y,
                  d_idx = threadIdx.x;

    float cam_patch[MAX_KERNEL_SIZE][MAX_KERNEL_SIZE];
    float proj_patch[MAX_KERNEL_SIZE][MAX_KERNEL_SIZE];

    // loop patch to get the mean value
    float cam_mean = 0, proj_mean = 0;
#pragma unroll
    for (int32_t row = 0; row < kernel_size; ++row) {
        for (int32_t col = 0; col < kernel_size; ++col) {
            const int32_t cam_i = h_idx + row - kernel_size / 2,
                          cam_j = w_idx + col - kernel_size / 2,
                          proj_i = h_idx + row - kernel_size / 2,
                          proj_j = d_idx + col - kernel_size / 2;
            const float cam = query_ij(cam_ptr, H, W, cam_i, cam_j);
            const float proj = query_ij(proj_ptr, H, W, proj_i, proj_j);
            cam_patch[row][col] = cam;
            proj_patch[row][col] = proj;
            cam_mean += cam;
            proj_mean += proj;
        }
    }
    cam_mean /= (kernel_size * kernel_size);
    proj_mean /= (kernel_size * kernel_size);

    float exy = 0, ex2 = 0, ey2 = 0;
#pragma unroll
    for (int32_t row = 0; row < kernel_size; ++row) {
        for (int32_t col = 0; col < kernel_size; ++col) {
            const float cam = cam_patch[row][col] - cam_mean;
            const float proj = proj_patch[row][col] - proj_mean;

            exy += cam * proj;
            ex2 += cam * cam;
            ey2 += proj * proj;
        }
    }
    cost_volume_ptr[h_idx * (W * (W - D)) + w_idx * W + d_idx] = (exy + EPSILON) / (sqrtf(ex2 * ey2 + EPSILON));
}

__global__ void get_patches_grad_kernel(
    // const int32_t elements,
    const int32_t H,
    const int32_t W,
    const int32_t D,
    const int32_t ks,
    const float *__restrict__ cost_volume_grad, // [H, crop_w, D + 1]
    const float *__restrict__ cam_ptr,          // [H, W]
    const float *__restrict__ proj_ptr,         // [H, W]
    // output
    float *__restrict__ camera_patches_grad_ptr // [H, W, ks, ks]
)
{
    // const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    // if (tid >= elements)
    // {
    //     return;
    // }
    const int32_t h_idx = blockIdx.x,
                  w_idx = blockIdx.y,
                  row = threadIdx.x,
                  col = threadIdx.y;

    __shared__ float cam_patch[MAX_KERNEL_SIZE][MAX_KERNEL_SIZE];
    __shared__ float proj_patch[MAX_KERNEL_SIZE][MAX_KERNEL_SIZE];
    const int32_t cam_i = h_idx + row - ks / 2;
    const int32_t cam_j = w_idx + col - ks / 2;
    cam_patch[row][col] = query_ij(cam_ptr, H, W, cam_i, cam_j);
    __syncthreads();

    float cam_mean = 0;
#pragma unroll
    for (int32_t i = 0; i < ks; ++i)
    {
        for (int32_t j = 0; j < ks; ++j)
        {
            cam_mean += cam_patch[i][j];
        }
    }

    for (int32_t d_idx = 0; d_idx < W; ++d_idx) {
        const int32_t proj_i_ = h_idx + row - ks / 2;
        const int32_t proj_j_ = d_idx + col - ks / 2;
        proj_patch[row][col] = query_ij(proj_ptr, H, W, proj_i_, proj_j_);
        __syncthreads();
        // loop patch to get the mean value
        float proj_mean = 0;
#pragma unroll
        for (int32_t i = 0; i < ks; ++i)
        {
            for (int32_t j = 0; j < ks; ++j)
            {
                const float proj = proj_patch[i][j];
                // const int32_t proj_i = h_idx + i - ks / 2;
                // const int32_t proj_j = d_idx + j - ks / 2;
                // const float proj = query_ij(proj_ptr, H, W, proj_i, proj_j);
                proj_mean += proj;
            }
        }
        proj_mean /= (ks * ks);

        float exy = 0, ex2 = 0, ey2 = 0;
#pragma unroll
        for (int32_t i = 0; i < ks; ++i)
        {
            for (int32_t j = 0; j < ks; ++j)
            {
                const float cam = cam_patch[i][j] - cam_mean;
                const float proj = proj_patch[i][j] - proj_mean;
                // const int32_t proj_i = h_idx + i - ks / 2;
                // const int32_t proj_j = d_idx + j - ks / 2;
                // const float proj = query_ij(proj_ptr, H, W, proj_i, proj_j) - proj_mean;

                exy += cam * proj;
                ex2 += cam * cam;
                ey2 += proj * proj;
            }
        }

        const float cost_grad = cost_volume_grad[h_idx * W * W + w_idx * W + d_idx];
        const int32_t off_w = ks * ks, off_h = ks * ks * W;
        float *curr_camera_patches_grad_ptr = camera_patches_grad_ptr + h_idx * off_h + w_idx * off_w;

        // calculate 1 time to save time
        const float deno = 1 / (sqrtf(ex2 * ey2 + EPSILON)), deno3 = 1 / powf((sqrtf(ex2 * ey2 + EPSILON)), 3);

        const float cam = cam_patch[row][col] - cam_mean;
        const float proj = proj_patch[row][col] - proj_mean;
        // const int32_t proj_i = h_idx + row - ks / 2;
        // const int32_t proj_j = d_idx + col - ks / 2;
        // const float proj = query_ij(proj_ptr, H, W, proj_i, proj_j) - proj_mean;
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
    const float *__restrict__ camera_patches_grad_ptr, // [H, W, ks, ks]
    // output
    float *__restrict__ camera_grad_ptr // [H, W]
)
{
    const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= elements)
    {
        return;
    }
    // const int32_t crop_w = W - D;
    // const int32_t d_idx = tid % (D + 1);
    // const int32_t w_idx = (tid / (D + 1)) % crop_w;
    // const int32_t h_idx = tid / ((D + 1) * crop_w);
    const int32_t k2_idx = tid % ks;
    const int32_t k1_idx = (tid / ks) % ks;
    const int32_t w_idx = (tid / (ks * ks)) % W;
    const int32_t h_idx = tid / (W * ks * ks);

    const int32_t cam_i = h_idx + k1_idx - ks / 2;
    const int32_t cam_j = w_idx + k2_idx - ks / 2;
    if (cam_i < 0 || cam_i >= H || cam_j < 0 || cam_j >= W)
    {
        return;
    }
    atomicAdd(camera_grad_ptr + cam_i * W + cam_j, camera_patches_grad_ptr[tid]);
}

Tensor stereo::stereo_matching_forward(
    const Tensor& camera,    // [H, W]
    const Tensor& projector, // [H, W]
    const int32_t D,
    const int32_t kernel_size
) {
    // check
    CHECK_INPUT(camera);
    CHECK_INPUT(projector);

    const int32_t H = camera.size(0), W = camera.size(1);
    const int32_t elements = H * W * W, threads = 1024;
    const int32_t blocks = ceil((elements - 1) / threads) + 1;
    
    assert(projector.size(0) == H && projector.size(1) == W);

    // Tensor cost_volume = torch::zeros({H, crop_w, D + 1},
    //         torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));

    Tensor cost_volume = torch::zeros({H, W, W},
            torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));

    dim3 dim_block(H, W);
    forward_cost_volume_kernel<<<dim_block, W>>>(
        H,
        W,
        D,
        kernel_size,
        camera.data_ptr<float>(),
        projector.data_ptr<float>(),
        // output
        cost_volume.data_ptr<float>()
    );
    
    return cost_volume;
}

Tensor stereo::stereo_matching_backward(
    const Tensor &cost_volume_grad, // [H, crop_w, D + 1]
    const Tensor &camera,           // [H, W]
    const Tensor &projector,        // [H, W]
    const int32_t kernel_size)
{
    // check
    CHECK_INPUT(cost_volume_grad);

    // get parameters
    const int32_t H = cost_volume_grad.size(0), W = cost_volume_grad.size(1), D = cost_volume_grad.size(2);
    // const int32_t H = cost_volume_grad.size(0), crop_w = cost_volume_grad.size(1), D = cost_volume_grad.size(2) - 1;
    // const int32_t W = crop_w + D;
    const int32_t elements1 = H * W, threads = 1024;
    // const int32_t elements1 = H * crop_w * (D + 1), threads = 1024;

    assert(kernel_size <= MAX_KERNEL_SIZE);
    Tensor camera_patches_grad = torch::zeros({H, W, kernel_size, kernel_size},
                                              torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));

    const int32_t blocks1 = ceil((elements1 - 1) / threads) + 1;
    const dim3 dim_block(H, W), thread_block(kernel_size, kernel_size);
    get_patches_grad_kernel<<<dim_block, thread_block>>>(
        // elements1,
        H,
        W,
        D,
        kernel_size,
        cost_volume_grad.data_ptr<float>(),
        camera.data_ptr<float>(),
        projector.data_ptr<float>(),
        // output
        camera_patches_grad.data_ptr<float>());

    Tensor camera_grad = torch::zeros({H, W},
                                      torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    const int32_t elements2 = H * W * kernel_size * kernel_size;
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
