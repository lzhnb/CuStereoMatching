from typing import Tuple

import torch

from .src import stereo_matching_forward, stereo_matching_backward

class _StereoMatching(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        camera_image: torch.Tensor,     # [H, W]
        projector_image: torch.Tensor,  # [H, W]
        D: int,
        kernel_size: int
    ) -> torch.Tensor:
        ctx.save_for_backward(camera_image, projector_image)
        ctx.D = D
        ctx.kernel_size = kernel_size
        cost_volume = stereo_matching_forward(camera_image, projector_image, D, kernel_size)

        return cost_volume

    @staticmethod
    def backward(ctx, cost_volume_grad: torch.Tensor) -> Tuple:
        camera_image, projector_image = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        stereo_matching_backward_grad = stereo_matching_backward(
            cost_volume_grad,
            camera_image,
            projector_image,
            kernel_size
        )
        return stereo_matching_backward_grad, None, None, None

stereo_matching = _StereoMatching.apply


