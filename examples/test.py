import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import custma

kernel_size = 15
H, W, D = 330, 422, 200
softargmax_beta = 50.0
cost_volume_threshold = 0.6
# cost_volume_threshold = torch.tensor(0.6).cuda() # Non-Differentiable



def extract_image_patch_pytoch(
    img: torch.Tensor,
    kernel: int,
    stride: int,
    pad: int
) -> torch.Tensor:

    img = F.pad(img, (pad, pad, pad, pad), mode='constant', value=0)
    img_patches = img.unfold(3, kernel, stride).unfold(2, kernel, stride).permute(0, 1, 2, 3, 5, 4)
    
    return img_patches


def soft_argmax(x: torch.Tensor, beta: float=50.0) -> torch.Tensor:
 
    N, C, L = x.shape
    soft_max = torch.softmax(x * beta, dim=2)
    soft_max = soft_max.view(x.shape)
    indices_kernel = torch.arange(start=0, end=L).unsqueeze(0).cuda()
    conv = soft_max * indices_kernel
    indices = conv.sum(2)
    return indices

def differentiable_disparity_calculating(
    camera_image: torch.Tensor,
    projector_image: torch.Tensor,
    kernel_size: int,
    softargmax_beta: float,
    cost_volume_threshold: float
) -> torch.Tensor:

    camera_image.requires_grad_(True)
    camera_image_patches = extract_image_patch_pytoch(
        camera_image.unsqueeze(0).unsqueeze(0),
        kernel = kernel_size,
        stride = 1,
        pad = int((kernel_size - 1) / 2)
    ).squeeze(0).squeeze(0)

    projector_image_patches = extract_image_patch_pytoch(
        projector_image.unsqueeze(0).unsqueeze(0),
        kernel = kernel_size,
        stride = 1,
        pad = int((kernel_size - 1) / 2)
    ).squeeze(0).squeeze(0)

    camera_image_patches.retain_grad()
    cost_volume = custma.stereo_matching(camera_image_patches.contiguous(), projector_image_patches.contiguous(), 200, kernel_size)
    import ipdb; ipdb.set_trace()
    cost_volume.backward(torch.ones_like(cost_volume))

    print("Cost Volume shape:", cost_volume.shape)
    # Detach To Calculate Cost Volume Mask
    cost_volume_max, _ = torch.max(cost_volume.detach().contiguous().reshape(H * (W - D), -1), dim=-1)
    cost_volume_max = cost_volume_max.reshape(H, (W - D))
    cost_volume_mask = torch.where(cost_volume_max > cost_volume_threshold, torch.ones_like(cost_volume_max), torch.zeros_like(cost_volume_max))

    cv2.imwrite(os.path.join(os.path.dirname(__file__), "temp.png"), np.array(cost_volume_mask.cpu()) * 255)

    # cv2.imshow("Cost Volume Mask", np.array(cost_volume_mask.cpu()))
    # correspondence_argmax = torch.argmax(cost_volume, dim=-1)
    # correspondence_softargmax = soft_argmax(cost_volume.contiguous().reshape(1, H * W, -1), softargmax_beta).reshape(H, W)
    # _, template = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    # template = template.cuda()

    # disparity = template - correspondence_argmax
    # disparity = torch.mul(disparity, cost_volume_mask)
    # disparity_softargmax = template - correspondence_softargmax
    # disparity_softargmax = torch.mul(disparity_softargmax, cost_volume_mask)

    # return disparity


if __name__ == "__main__":
    root_dir = os.path.dirname(__file__)
    
    camera_image = torch.ones(100, 200).cuda()
    camera_image.requires_grad_(True)
    camera_image_patches = extract_image_patch_pytoch(
        camera_image.unsqueeze(0).unsqueeze(0),
        kernel = kernel_size,
        stride = 1,
        pad = int((kernel_size - 1) / 2)
    ).squeeze(0).squeeze(0)
    camera_image_patches.backward(torch.ones_like(camera_image_patches))
    import ipdb; ipdb.set_trace()
    pass


