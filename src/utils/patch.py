import numpy as np
import math
from torchvision.transforms import transforms
from PIL import Image


def compute(patch):
    weight, height = patch.size
    m = weight
    res = 0
    patch = np.array(patch).astype(np.int64)
    diff_horizontal = np.sum(np.abs(patch[:, :-1, :] - patch[:, 1:, :]))
    diff_vertical = np.sum(np.abs(patch[:-1, :, :] - patch[1:, :, :]))
    diff_diagonal = np.sum(np.abs(patch[:-1, :-1, :] - patch[1:, 1:, :]))
    diff_diagonal += np.sum(np.abs(patch[1:, :-1, :] - patch[:-1, 1:, :]))
    res = diff_horizontal + diff_vertical + diff_diagonal
    return res.sum()


def patch_img(img, patch_size, height, deterministic=False):
    """
    从图像中选取复杂度最低的 patch。
    
    Args:
        img: PIL Image
        patch_size: patch 边长
        height: 目标高度（resize 后）
        deterministic: 是否使用确定性网格切分（用于推理）
    
    Returns:
        复杂度最低的 patch (PIL Image)
    """
    img_width, img_height = img.size
    num_patch = (height // patch_size) * (height // patch_size)
    patch_list = []
    min_len = min(img_height, img_width)
    rz = transforms.Resize((height, height))
    if min_len < patch_size:
        img = rz(img)
    
    if deterministic:
        # 确定性网格切分：将图像划分成网格，固定位置提取 patch
        img_resized = rz(img)
        grid_size = height // patch_size  # e.g., 256 // 32 = 8
        for row in range(grid_size):
            for col in range(grid_size):
                left = col * patch_size
                top = row * patch_size
                patch = img_resized.crop((left, top, left + patch_size, top + patch_size))
                patch_list.append(patch)
    else:
        # 随机裁剪（用于训练数据增强）
        rp = transforms.RandomCrop(patch_size)
        for i in range(num_patch):
            patch_list.append(rp(img))
    
    patch_list.sort(key=lambda x: compute(x), reverse=False)
    new_img = patch_list[0]

    return new_img


def patch_img_deterministic(img, patch_size, height):
    """
    确定性版本的 patch_img，用于推理。
    等价于 patch_img(img, patch_size, height, deterministic=True)
    """
    return patch_img(img, patch_size, height, deterministic=True)
