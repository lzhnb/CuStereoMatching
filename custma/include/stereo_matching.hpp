// Copyright 2022 Zhihao Liang
#pragma once
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <torch/extension.h>
#include <vector>

using std::vector;
using torch::Tensor;

namespace stereo {
vector<Tensor> stereo_matching_forward(
    const Tensor&,
    const Tensor&,
    const int32_t,
    const int32_t);
vector<Tensor> stereo_matching_backward(
    const Tensor&,
    const Tensor&,
    const Tensor&,
    const Tensor&,
    const Tensor&,
    const Tensor&,
    const Tensor&,
    const Tensor&,
    const int32_t,
    const bool);
// use to verify the backward
Tensor exy_grad_to_image(
    const Tensor&,
    const Tensor&,
    const Tensor&,
    const Tensor&,
    const int32_t
);
Tensor ex2_grad_to_image(
    const Tensor&,
    const Tensor&,
    const Tensor&,
    const int32_t
);
} // namespace stereo

// Utils
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor")

#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

#define CHECK_CPU_INPUT(x) \
    CHECK_CPU(x);          \
    CHECK_CONTIGUOUS(x)
