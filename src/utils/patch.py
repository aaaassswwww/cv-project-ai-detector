import numpy as np
import math
from torchvision import transforms
from PIL import Image


def _brightness_variance(patch: Image.Image) -> float:
    arr = np.asarray(patch.convert('L'), dtype=np.float32)
    return float(arr.var())


def compute(patch):
    """计算 patch 的纹理复杂度（所有方向的像素差分和）"""
    patch = np.array(patch).astype(np.int64)
    diff_horizontal = np.sum(np.abs(patch[:, :-1, :] - patch[:, 1:, :]))
    diff_vertical = np.sum(np.abs(patch[:-1, :, :] - patch[1:, :, :]))
    diff_diagonal = np.sum(np.abs(patch[:-1, :-1, :] - patch[1:, 1:, :]))
    diff_diagonal += np.sum(np.abs(patch[1:, :-1, :] - patch[:-1, 1:, :]))
    # 已经是标量，直接返回（不要再调用 .sum()）
    return float(diff_horizontal + diff_vertical + diff_diagonal)


def patch_img(img, patch_size, height, deterministic=False, var_thresh: float = 0.0, topk: int = 1):
    """
    从图像中选取复杂度最低的 top-k patch。
    
    Args:
        img: PIL Image
        patch_size: patch 边长
        height: 目标高度（resize 后）
        deterministic: 是否使用确定性网格切分（用于推理）
        var_thresh: 亮度方差阈值，低于此值的纯色 patch 会被过滤
        topk: 返回前 k 个最简单的 patch
    
    Returns:
        单个 PIL Image（topk=1）或 list of PIL Images（topk>1）
    """
    # 统一 resize 到目标尺寸，确保所有 patch 提取逻辑一致
    rz = transforms.Resize((height, height))
    img_resized = rz(img)
    
    patch_list = []
    grid_size = height // patch_size  # e.g., 256 // 32 = 8
    num_patch = grid_size * grid_size
    
    if deterministic:
        # 确定性网格切分：将图像划分成网格，固定位置提取 patch
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
            patch_list.append(rp(img_resized))

    # 先按亮度方差过滤纯色/低信息 patch
    if var_thresh > 0:
        filtered = [p for p in patch_list if _brightness_variance(p) >= var_thresh]
        if filtered:  # 只在有符合条件的 patch 时才过滤，否则保留全部
            patch_list = filtered

    # 按复杂度从低到高排序
    patch_list.sort(key=lambda x: compute(x), reverse=False)

    # 取 top-k 最简单 patch，至少返回 1 个
    topk = max(1, min(topk, len(patch_list)))
    top_patches = patch_list[:topk]

    # 返回单个 patch 或 list（保持一致性）
    return top_patches[0] if topk == 1 else top_patches


def patch_img_deterministic(img, patch_size, height, var_thresh: float = 0.0, topk: int = 1):
    """
    确定性版本的 patch_img，用于推理。
    等价于 patch_img(img, patch_size, height, deterministic=True)
    """
    return patch_img(img, patch_size, height, deterministic=True, var_thresh=var_thresh, topk=topk)
