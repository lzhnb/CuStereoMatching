// Copyright 2022 Zhihao Liang
#include <cstdint>
#include <iostream>
#include <pybind11/functional.h>

#include "stereo_matching.hpp"

void stereo::centerize_patches(Tensor& patches    // [H, W, p_h * p_w]
) {
    const int32_t H = patches.size(0), W = patches.size(1);
    const torch::Tensor mean_patches = patches.mean(2).reshape({H, W, 1}); // [H, W, 1]
    patches -= mean_patches;
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

    // get parameters
    const int32_t H = camera.size(0), W = camera.size(1);
    assert(projector.size(0) == H && projector.size(1) == W);

    torch::Tensor disparity = torch::zeros({H, W},
            torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));

    // forward
    Tensor cost_volume = stereo_matching_forward_wrapper(
        camera,
        projector,
        D,
        kernel_size,
        disparity
    );
    return cost_volume;
}


Tensor stereo::stereo_matching_backward(
    const Tensor& cost_volume_grad, // [H, crop_w, D + 1]
    const Tensor& camera,           // [H, W, ph, pw]
    const Tensor& projector,        // [H, W, ph, pw]
    const int32_t kernel_size
) {
    // check
    CHECK_INPUT(cost_volume_grad);

    // get parameters
    const int32_t H = cost_volume_grad.size(0), W = cost_volume_grad.size(1), D = cost_volume_grad.size(2);
    // const int32_t H = cost_volume_grad.size(0), crop_w = cost_volume_grad.size(1), D = cost_volume_grad.size(2) - 1;
    // const int32_t W = crop_w + D;

    // reshape
    Tensor camera_flatten = camera.reshape({H, W, -1}); // [H, W, p_h * p_w]
    Tensor projector_flatten = projector.reshape({H, W, -1}); // [H, W, p_h * p_w]
    centerize_patches(camera_flatten);
    centerize_patches(projector_flatten);

    Tensor camera_grad = torch::zeros({H, W, kernel_size, kernel_size},
            torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));

    // forward
    stereo_matching_backward_wrapper(
        cost_volume_grad,
        camera_flatten,
        projector_flatten,
        kernel_size,
        // output
        camera_grad
    );
    return camera_grad;
}


