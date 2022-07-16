import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import custma
import custma.src as _C

kernel_size = 15
H, W, D = 330, 422, 200
softargmax_beta = 50.0
cost_volume_threshold = 0.6
# cost_volume_threshold = torch.tensor(0.6).cuda() # Non-Differentiable


def extract_image_patch_pytoch(
    img: torch.Tensor, kernel: int, stride: int, pad: int
) -> torch.Tensor:

    img = F.pad(img, (pad, pad, pad, pad), mode="constant", value=0)
    img_patches = (
        img.unfold(3, kernel, stride)
        .unfold(2, kernel, stride)
        .permute(0, 1, 2, 3, 5, 4)
    )

    return img_patches


def soft_argmax(x: torch.Tensor, beta: float = 50.0) -> torch.Tensor:

    N, C, L = x.shape
    soft_max = torch.softmax(x * beta, dim=2)
    soft_max = soft_max.view(x.shape)
    indices_kernel = torch.arange(start=0, end=L).unsqueeze(0).cuda()
    conv = soft_max * indices_kernel
    indices = conv.sum(2)
    return indices


def cuda_cost_volume_backward(
    camera_image: torch.Tensor,
    projector_image: torch.Tensor,
    kernel_size: int,
    softargmax_beta: float,
    cost_volume_threshold: float,
) -> torch.Tensor:

    camera_image.requires_grad_(True)
    # camera_image_patches = extract_image_patch_pytoch(
    #     camera_image.unsqueeze(0).unsqueeze(0),
    #     kernel = kernel_size,
    #     stride = 1,
    #     pad = int((kernel_size - 1) / 2)
    # ).squeeze(0).squeeze(0)

    # projector_image_patches = extract_image_patch_pytoch(
    #     projector_image.unsqueeze(0).unsqueeze(0),
    #     kernel = kernel_size,
    #     stride = 1,
    #     pad = int((kernel_size - 1) / 2)
    # ).squeeze(0).squeeze(0)
    # camera_image_patches.retain_grad()

    with custma.Timer("cuda forward time: {:.6f}s"):
        # cost_volume = custma.stereo_matching(
        #     camera_image.contiguous(), projector_image.contiguous(), 200, kernel_size
        # )
        ex2, ey2, exy, ex2_mean, ey2_mean, cost_volume = _C.stereo_matching_forward(
            camera_image.contiguous(), projector_image.contiguous(), 0, kernel_size
        )
        # ex2 = custma.stereo_matching(
        #     camera_image.contiguous(), projector_image.contiguous(), 0, kernel_size
        # )
    # with custma.Timer("cuda backward time: {:.6f}s"):
    #     cost_volume.backward(torch.ones_like(cost_volume))

    # print("Cost Volume shape:", cost_volume.shape)
    # # Detach To Calculate Cost Volume Mask
    # cost_volume_max, _ = torch.max(
    #     cost_volume.detach().contiguous().reshape(H * W, -1), dim=-1
    # )
    # cost_volume_max = cost_volume_max.reshape(H, W)
    # cost_volume_mask = torch.where(
    #     cost_volume_max > cost_volume_threshold,
    #     torch.ones_like(cost_volume_max),
    #     torch.zeros_like(cost_volume_max),
    # )

    # cv2.imwrite(os.path.join(os.path.dirname(__file__), "temp.png"), np.array(cost_volume_mask.cpu()) * 255)

    return ex2, ey2, exy, ex2_mean, ey2_mean, cost_volume


def torch_cost_volume_backward(
    camera_image: torch.Tensor,
    projector_image: torch.Tensor,
    kernel_size: int,
    softargmax_beta: float,
    cost_volume_threshold: float,
) -> torch.Tensor:
    camera_image.requires_grad_(True)

    with custma.Timer("torch forward time: {:.6f}s"):
        camera_img_patches_ = (
            extract_image_patch_pytoch(
                camera_image.unsqueeze(0).unsqueeze(0),
                kernel=kernel_size,
                stride=1,
                pad=int((kernel_size - 1) / 2),
            )
            .squeeze(0)
            .squeeze(0)
        )
        camera_img_patches_.retain_grad()

        projector_img_patches = (
            extract_image_patch_pytoch(
                projector_image.unsqueeze(0).unsqueeze(0),
                kernel=kernel_size,
                stride=1,
                pad=int((kernel_size - 1) / 2),
            )
            .squeeze(0)
            .squeeze(0)
        )

        H, W = camera_img_patches_.shape[:2]
        camera_img_patches = camera_img_patches_.contiguous().reshape(H, W, -1)
        projector_img_patches = projector_img_patches.contiguous().reshape(H, W, -1)
        camera_img_patches_mean = torch.mean(camera_img_patches, dim=-1, keepdim=True)
        projector_img_patches_mean = torch.mean(
            projector_img_patches, dim=-1, keepdim=True
        )
        eps = 1e-8
        camera_img_patches -= camera_img_patches_mean
        projector_img_patches -= projector_img_patches_mean
        # cost_volume = torch.zeros(H, W, W)
        EXY = torch.bmm(camera_img_patches, projector_img_patches.permute(0, 2, 1))
        # To faster calculate EX2 and EY2, reshape (H, W, Nd) to (H*W, 1, Nd), which maybe more efficient
        EX2 = (
            torch.bmm(
                camera_img_patches.reshape(H * W, 1, -1),
                camera_img_patches.reshape(H * W, 1, -1).permute(0, 2, 1),
            )
            .reshape(H, W)
            .unsqueeze(-1)
        )
        EY2 = (
            torch.bmm(
                projector_img_patches.reshape(H * W, 1, -1),
                projector_img_patches.reshape(H * W, 1, -1).permute(0, 2, 1),
            )
            .reshape(H, W)
            .unsqueeze(-2)
        )
        cost_volume = (EXY + eps) / (torch.sqrt(EX2 * EY2 + eps))

    with custma.Timer("torch backward time: {:.6f}s"):
        cost_volume.backward(torch.ones_like(cost_volume))

    # print("Cost Volume shape:", cost_volume.shape)
    # # Detach To Calculate Cost Volume Mask
    # cost_volume_max, _ = torch.max(cost_volume.detach().contiguous().reshape(H * W, -1), dim=-1)
    # cost_volume_max = cost_volume_max.reshape(H, W)
    # cost_volume_mask = torch.where(cost_volume_max > cost_volume_threshold, torch.ones_like(cost_volume_max), torch.zeros_like(cost_volume_max))

    # cv2.imwrite(os.path.join(os.path.dirname(__file__), "temp_torch.png"), np.array(cost_volume_mask.cpu()) * 255)

    return EX2, EY2, EXY, camera_img_patches_mean, projector_img_patches_mean, cost_volume, camera_image.grad


if __name__ == "__main__":
    root_dir = os.path.dirname(__file__)
    rgb = (
        np.array(
            cv2.imread(os.path.join(root_dir, "test_chair_0003_rgb_20220504_3.png"))
        )
        / 255
    )
    proj = (
        np.array(
            cv2.imread(
                os.path.join(root_dir, "SpecklePattern_PreRender_D51/4/0.png"), 0
            )
        )
        / 255
    )

    rgb = torch.tensor(rgb, dtype=torch.float32).cuda()
    proj = torch.tensor(proj, dtype=torch.float32).cuda()

    disparity = torch.zeros(rgb.shape[0], rgb.shape[1]).cuda()
    disparity_mask = torch.ones(rgb.shape[0], rgb.shape[1]).cuda()

    with custma.Timer("cuda time: {:.6f}s"):
        ex2, ey2, exy, ex2_mean, ey2_mean, cost_volume = cuda_cost_volume_backward(
            rgb[:, :, 0], proj, kernel_size, softargmax_beta, cost_volume_threshold
        )
    with custma.Timer("torch time: {:.6f}s"):
        EX2, EY2, EXY, EX2_MEAN, EY2_MEAN, torch_cost_volume, torch_camera_image_grad = torch_cost_volume_backward(
            rgb[:, :, 0], proj, kernel_size, softargmax_beta, cost_volume_threshold
        )

    EX2 = EX2.squeeze()
    EY2 = EY2.squeeze()
    EX2_MEAN = EX2_MEAN.squeeze()
    EY2_MEAN = EY2_MEAN.squeeze()
    
    import ipdb; ipdb.set_trace()
    # print(cuda_camera_image_grad)
    print(torch_camera_image_grad)
    # print(disparity)
