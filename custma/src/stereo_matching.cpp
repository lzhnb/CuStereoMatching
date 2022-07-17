// Copyright 2022 Zhihao Liang
#include <cstdint>
#include <iostream>
#include <pybind11/functional.h>

#include "stereo_matching.hpp"


vector<Tensor> stereo::stereo_matching_forward(
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
    return stereo_matching_forward_wrapper(
        camera,
        projector,
        D,
        kernel_size,
        disparity
    );
}


