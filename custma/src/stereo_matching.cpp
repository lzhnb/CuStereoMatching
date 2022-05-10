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

// Tensor stereo::stereo_matching_forward(
//     const Tensor& camera_patches,    // [H, W, p_h, p_w]
//     const Tensor& projector_patches  // [H, W, p_h, p_w]
// ) {
//     // check
//     CHECK_INPUT(camera_patches);
//     CHECK_INPUT(projector_patches);

//     // get parameters
//     const int32_t H = camera_patches.size(0), W = camera_patches.size(1);
//     assert(projector_patches.size(0) == H && projector_patches.size(1) == W);

//     // reshape
//     Tensor camera_patches_flatten = camera_patches.reshape({H, W, -1}); // [H, W, p_h * p_w]
//     Tensor projector_patches_flatten = projector_patches.reshape({H, W, -1}); // [H, W, p_h * p_w]
//     centerize_patches(camera_patches_flatten);
//     centerize_patches(projector_patches_flatten);

//     Tensor disparity = zeros_like(camera_patches);
//     return disparity;
// }

Tensor stereo::stereo_matching_forward(
    const Tensor& camera,    // [H, W, ph, pw]
    const Tensor& projector, // [H, W, ph, pw]
    const int32_t D,
    const int32_t patch
) {
    // check
    CHECK_INPUT(camera);
    CHECK_INPUT(projector);

    // get parameters
    const int32_t H = camera.size(0), W = camera.size(1);
    assert(projector.size(0) == H && projector.size(1) == W);

    // reshape
    Tensor camera_flatten = camera.reshape({H, W, -1}); // [H, W, p_h * p_w]
    Tensor projector_flatten = projector.reshape({H, W, -1}); // [H, W, p_h * p_w]
    centerize_patches(camera_flatten);
    centerize_patches(projector_flatten);

    torch::Tensor disparity = torch::zeros({H, W},
            torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));

    // forward
    Tensor cost_volume = stereo_matching_forward_wrapper(
        camera_flatten,
        projector_flatten,
        D,
        patch,
        disparity
    );
    return cost_volume;
}


