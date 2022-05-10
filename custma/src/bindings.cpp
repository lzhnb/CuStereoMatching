// Copyright 2022 Zhihao Liang
#include "stereo_matching.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("stereo_matching_forward", &stereo::stereo_matching_forward);
}

