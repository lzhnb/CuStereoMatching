// Copyright 2022 Zhihao Liang
#pragma once
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <thrust/sort.h>

using torch::Tensor;

namespace stereo
{
    void centerize_patches(Tensor&);
    Tensor stereo_matching_forward(const Tensor&, const Tensor&, const int32_t, const int32_t);
    Tensor stereo_matching_forward_wrapper(const Tensor&, const Tensor&, const int32_t, const int32_t, Tensor&);
    Tensor stereo_matching_backward(const Tensor&, const Tensor&, const Tensor&, const int32_t);
    void stereo_matching_backward_wrapper(const Tensor&, const Tensor&, const Tensor&, const int32_t, Tensor&);
} // namespace stereo


// Utils
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor")

#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_INPUT(x)      \
    CHECK_CUDA(x);          \
    CHECK_CONTIGUOUS(x)

#define CHECK_CPU_INPUT(x)  \
    CHECK_CPU(x);           \
    CHECK_CONTIGUOUS(x)

