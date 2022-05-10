// Copyright 2022 Zhihao Liang
#include <cstdint>
#include <iostream>
#include <pybind11/functional.h>

#include "stereo_matching.hpp"

torch::Tensor stereo::stereo_matching_forward(
    const torch::Tensor& camera_patches,
    const torch::Tensor& projector_patches
) {
    // check
    CHECK_INPUT(camera_patches);
    CHECK_INPUT(projector_patches);

    torch::Tensor disparity = torch::zeros_like(camera_patches);
    return disparity;
}


