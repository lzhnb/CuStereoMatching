from typing import Tuple

import torch

from .src import stereo_matching_forward, stereo_matching_backward


class _StereoMatching(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        camera_image: torch.Tensor,  # [H, W]
        projector_image: torch.Tensor,  # [H, W]
        D: int,
        kernel_size: int,
        record: bool = False,
    ) -> torch.Tensor:
        ctx.kernel_size = kernel_size
        ctx.record = record
        (
            ex2,
            ey2,
            exy,
            ex2_mean,
            ey2_mean,
            cost_volume,
            camera_patch,
            projector_patch,
        ) = stereo_matching_forward(
            camera_image, projector_image, D, kernel_size
        )
        ctx.save_for_backward(
            camera_image, projector_image, ex2, ey2, exy, ex2_mean, ey2_mean
        )

        return cost_volume

    @staticmethod
    def backward(ctx, cost_volume_grad: torch.Tensor) -> Tuple:
        (
            camera_image,
            projector_image,
            ex2,
            ey2,
            exy,
            ex2_mean,
            ey2_mean,
        ) = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        record = ctx.record
        camera_grad, _, _ = stereo_matching_backward(
            cost_volume_grad,
            camera_image,
            projector_image,
            ex2,
            ey2,
            exy,
            ex2_mean,
            ey2_mean,
            kernel_size,
            record,
        )
        return camera_grad, None, None, None, None


stereo_matching = _StereoMatching.apply
