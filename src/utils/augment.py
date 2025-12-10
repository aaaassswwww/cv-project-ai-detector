import io
import random
from typing import Tuple

import torch
from PIL import Image
from torchvision import transforms


class RandomJPEGCompression:
    """Apply random JPEG compression to a PIL image."""

    def __init__(self, p: float = 0.2, quality_range: Tuple[int, int] = (30, 95)):
        self.p = p
        self.quality_range = quality_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        buffer = io.BytesIO()
        quality = random.randint(*self.quality_range)
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        pil = Image.open(buffer).convert("RGB")
        out = pil.copy()   # 把图像数据复制到内存，不再依赖 buffer
        buffer.close()
        return out


class RandomGaussianBlurProb:
    def __init__(self, p: float = 0.15, kernel_size: int = 3, sigma_range: Tuple[float, float] = (0.1, 2.0)):
        self.p = p
        self.kernel_size = kernel_size
        self.sigma_range = sigma_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        sigma = random.uniform(*self.sigma_range)
        blur = transforms.GaussianBlur(kernel_size=self.kernel_size, sigma=sigma)
        return blur(img)


class RandomResample:
    """Downsample then upsample to original size."""

    def __init__(self, p: float = 0.15, scale_range: Tuple[float, float] = (0.5, 0.9), resample=Image.BICUBIC):
        self.p = p
        self.scale_range = scale_range
        self.resample = resample

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        w, h = img.size
        scale = random.uniform(*self.scale_range)
        down_w = max(1, int(w * scale))
        down_h = max(1, int(h * scale))
        downsampled = img.resize((down_w, down_h), self.resample)
        return downsampled.resize((w, h), self.resample)


class RandomGaussianNoise:
    def __init__(self, p: float = 0.1, sigma_range: Tuple[float, float] = (0.005, 0.02)):
        self.p = p
        self.sigma_range = sigma_range

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return tensor
        sigma = random.uniform(*self.sigma_range)
        noise = torch.randn_like(tensor) * sigma
        return (tensor + noise).clamp(0.0, 1.0)


class RandomFreqPerturbation:
    """Scale high-frequency spectrum components."""

    def __init__(self, p: float = 0.1, scale_range: Tuple[float, float] = (0.7, 1.3), radius: float = 0.25):
        self.p = p
        self.scale_range = scale_range
        self.radius = radius

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return tensor
        if tensor.dim() != 3:
            return tensor
        c, h, w = tensor.shape
        device = tensor.device
        dtype = tensor.dtype

        freq = torch.fft.fftn(tensor, dim=(-2, -1))

        fy = torch.fft.fftfreq(h, d=1.0, device=device)
        fx = torch.fft.fftfreq(w, d=1.0, device=device)
        grid_y, grid_x = torch.meshgrid(fy, fx, indexing="ij")
        radius_map = torch.sqrt(grid_x**2 + grid_y**2)

        scale = random.uniform(*self.scale_range)
        # 使用 freq 的 dtype (complex) 而非 tensor 的 dtype (float)
        mask = torch.ones((h, w), device=device, dtype=freq.dtype)
        mask = mask + (radius_map >= self.radius) * (scale - 1.0)
        mask = mask.unsqueeze(0)

        freq = freq * mask
        perturbed = torch.fft.ifftn(freq, dim=(-2, -1)).real
        return perturbed.clamp(0.0, 1.0)
