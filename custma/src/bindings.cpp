// Copyright 2022 Zhihao Liang
#include "stereo_matching.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("stereo_matching_forward", &stereo::stereo_matching_forward);
    m.def("stereo_matching_backward", &stereo::stereo_matching_backward);
    // use to verify the backward
    m.def("exy_grad_to_image", &stereo::exy_grad_to_image);
    m.def("ex2_grad_to_image", &stereo::ex2_grad_to_image);
}

