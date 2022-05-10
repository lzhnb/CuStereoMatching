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

    # cost_volume = custma.stereo_matching_forward(camera_image_patches.contiguous(), projector_image_patches.contiguous(), 200, kernel_size)
    cost_volume = custma.stereo_matching_forward(camera_image.contiguous(), projector_image.contiguous(), 200, kernel_size)

    print("Cost Volume shape:", cost_volume.shape)
    # Detach To Calculate Cost Volume Mask
    cost_volume_max, _ = torch.max(cost_volume.detach().contiguous().reshape(H * (W - D), -1), dim=-1)
    cost_volume_max = cost_volume_max.reshape(H, (W - D))
    cost_volume_mask = torch.where(cost_volume_max > cost_volume_threshold, torch.ones_like(cost_volume_max), torch.zeros_like(cost_volume_max))

    cv2.imwrite("temp.png", np.array(cost_volume_mask.cpu()) * 255)
    torch.cuda.empty_cache()
    import ipdb; ipdb.set_trace()

    pass
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
    rgb = np.array(cv2.imread("test_chair_0003_rgb_20220504_3.png")) / 255
    proj = np.array(cv2.imread("SpecklePattern_PreRender_D51/4/0.png", 0)) / 255

    rgb = torch.tensor(rgb, dtype=torch.float32).cuda()
    proj = torch.tensor(proj, dtype=torch.float32).cuda()

    disparity = torch.zeros(rgb.shape[0], rgb.shape[1]).cuda()
    disparity_mask = torch.ones(rgb.shape[0], rgb.shape[1]).cuda()

    disparity = differentiable_disparity_calculating(rgb[:,:, 0], proj, kernel_size, softargmax_beta, cost_volume_threshold)
    # print(disparity)


