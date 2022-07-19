import os
from re import M

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

    ex2 = torch.zeros_like(camera_image)
    ey2 = torch.zeros_like(projector_image)
    exy = torch.zeros(
        [camera_image.shape[0], camera_image.shape[1], projector_image.shape[1]],
        device=camera_image.device,
    )
    ex2_mean = torch.zeros_like(ex2)
    ey2_mean = torch.zeros_like(ey2)
    ex2_grad = torch.zeros_like(ex2)
    exy_grad = torch.zeros_like(exy)

    with custma.Timer("cuda forward time: {:.6f}s"):
        # cost_volume = custma.stereo_matching(
        #     camera_image.contiguous(), projector_image.contiguous(), 0, kernel_size
        # )
        ex2, ey2, exy, cost_volume = _C.stereo_matching_forward(
            camera_image.contiguous(), projector_image.contiguous(), 0, kernel_size
        )
    with custma.Timer("cuda backward time: {:.6f}s"):
        # cost_volume.backward(torch.ones_like(cost_volume))
        camera_grad = camera_image.grad
        camera_patch_grad = None
        camera_grad, ex2_grad, exy_grad = _C.stereo_matching_backward(
            torch.ones_like(cost_volume),
            camera_image.contiguous(),
            projector_image.contiguous(),
            ex2,
            ey2,
            exy,
            kernel_size,
        )
        # camera_grad = _C.exy_grad_to_image(torch.ones_like(exy), camera_image.contiguous(), projector_image.contiguous(), ey2_mean, kernel_size)
        # camera_grad = _C.ex2_grad_to_image(torch.ones_like(ex2), camera_image.contiguous(), ex2_mean, kernel_size)

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

    return (
        cost_volume,
        camera_grad,
        camera_patch_grad,
        ex2,
        ey2,
        exy,
        ex2_grad,
        exy_grad,
        ex2_mean,
        ey2_mean,
    )


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

        projector_img_patches_ = (
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
        projector_img_patches = projector_img_patches_.contiguous().reshape(H, W, -1)
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
        factor = torch.bmm(EX2, EY2) + eps
        factor.retain_grad()
        cost_volume = (EXY + eps) / (torch.sqrt(factor))

    # import ipdb; ipdb.set_trace()
    # camera_img_patches_mean.retain_grad()
    EX2.retain_grad()
    EXY.retain_grad()
    with custma.Timer("torch backward time: {:.6f}s"):
        cost_volume.backward(torch.ones_like(cost_volume))
        camera_image_grad = camera_image.grad
        # camera_image_grad = torch.autograd.grad(
        #     EX2,
        #     camera_image,
        #     torch.ones_like(EX2),
        #     create_graph=True,
        #     retain_graph=True,
        #     only_inputs=True
        # )[0]

    # print("Cost Volume shape:", cost_volume.shape)
    # # Detach To Calculate Cost Volume Mask
    # cost_volume_max, _ = torch.max(cost_volume.detach().contiguous().reshape(H * W, -1), dim=-1)
    # cost_volume_max = cost_volume_max.reshape(H, W)
    # cost_volume_mask = torch.where(cost_volume_max > cost_volume_threshold, torch.ones_like(cost_volume_max), torch.zeros_like(cost_volume_max))

    # cv2.imwrite(os.path.join(os.path.dirname(__file__), "temp_torch.png"), np.array(cost_volume_mask.cpu()) * 255)

    return (
        EX2,
        EY2,
        EXY,
        factor.grad,
        camera_img_patches_mean,
        projector_img_patches_mean,
        cost_volume,
        camera_img_patches_.grad,
        camera_image_grad,
        camera_img_patches,
        projector_img_patches,
    )


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

    torch.cuda.empty_cache()
    with custma.Timer("cuda time: {:.6f}s"):
        (
            cuda_cost_volume,
            cuda_img_grad,
            cuda_img_patch_grad,
            ex2,
            ey2,
            exy,
            ex2_grad,
            exy_grad,
            ex2_mean,
            ey2_mean,
        ) = cuda_cost_volume_backward(
            rgb[:, :, 0], proj, kernel_size, softargmax_beta, cost_volume_threshold
        )

    torch.cuda.empty_cache()
    with custma.Timer("torch time: {:.6f}s"):
        (
            EX2,
            EY2,
            EXY,
            factor_grad,
            EX2_MEAN,
            EY2_MEAN,
            torch_cost_volume,
            torch_img_patch_grad,
            torch_img_grad,
            camera_img_patches,
            projector_img_patches,
        ) = torch_cost_volume_backward(
            rgb[:, :, 0], proj, kernel_size, softargmax_beta, cost_volume_threshold
        )

    EX2_GRAD = EX2.grad.squeeze()
    EXY_GRAD = EXY.grad.squeeze()
    EX2 = EX2.squeeze()
    EY2 = EY2.squeeze()
    EXY = EXY.squeeze()
    ex2 = ex2.squeeze()
    ey2 = ey2.squeeze()
    ex2_grad = ex2_grad.squeeze()
    EX2_MEAN = EX2_MEAN.squeeze()
    EY2_MEAN = EY2_MEAN.squeeze()

    print(f"ex2 error: {(ex2 - EX2).abs().max()}")
    print(f"ey2 error: {(ey2 - EY2).abs().max()}")
    print(f"exy error: {(exy - EXY).abs().max()}")
    # print(f"ex2_mean error: {(ex2_mean - EX2_MEAN).abs().max()}")
    # print(f"ey2_mean error: {(ey2_mean - EY2_MEAN).abs().max()}")
    print(f"cost_volume error: {(cuda_cost_volume - torch_cost_volume).abs().max()}")
    print(f"ex2_grad error: {(ex2_grad - EX2_GRAD).abs().max()}")
    print(f"exy_grad error: {(exy_grad - EXY_GRAD).abs().max()}")

    cuda_img_patch_grad = torch.bmm(
        exy_grad, projector_img_patches.reshape(H, W, -1)
    ) + 2 * torch.bmm(
        ex2_grad.reshape(H * W, 1, 1), camera_img_patches.reshape(H * W, 1, 225)
    ).reshape(
        H, W, 225
    )
    print(f"img_grad error: {(cuda_img_grad - torch_img_grad).abs().max()}")
    # print(f"img_grad error: {(cuda_img_grad - torch_img_patch_grad.reshape(H, W, -1)).abs().max()}")
    print(
        f"img_patch_grad error: {(cuda_img_patch_grad - torch_img_patch_grad.reshape(H, W, 225)).abs().max()}"
    )
    print("cuda_img_grad: \n", cuda_img_grad[270:275, 210:212])
    print("torch_img_grad: \n", torch_img_grad[270:275, 210:212])

