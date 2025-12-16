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
        # 使用 float dtype 避免 bool 转 complex 的问题
        mask = torch.ones((h, w), device=device, dtype=torch.float32)
        # 明确将 bool mask 转换为 float
        high_freq_mask = (radius_map >= self.radius).to(torch.float32)
        mask = mask + high_freq_mask * (scale - 1.0)
        mask = mask.unsqueeze(0)

        freq = freq * mask
        perturbed = torch.fft.ifftn(freq, dim=(-2, -1)).real
        return perturbed.clamp(0.0, 1.0)


class RandomFDA:
    """
    FDA (Fourier Domain Adaptation) augmentation for low-frequency domain mixing.
    
    Replaces low-frequency amplitude of source image with target image's amplitude,
    while keeping source phase intact. Only affects central low-frequency region
    (controlled by beta_range) to preserve high-frequency forensic features.
    
    Args:
        beta_range: Range of beta values (e.g., (0.01, 0.05) means 1-5% of image size)
                   Controls the size of low-frequency region to replace
        p: Probability of applying FDA (default: 0.3)
    
    Example:
        beta=0.05, image 384x384 -> replace central 38x38 region in frequency domain
        This affects global color/lighting/texture style while preserving local forensic traces
    """
    
    def __init__(self, beta_range: Tuple[float, float] = (0.01, 0.05), p: float = 0.3):
        self.beta_range = beta_range
        self.p = p

    def __call__(self, src: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        """
        Apply FDA augmentation.
        
        Args:
            src: Source tensor in [0,1] range, shape (C, H, W)
            trg: Target tensor in [0,1] range, shape (C, H, W)
        
        Returns:
            Augmented tensor in [0,1] range with src's phase and mixed amplitude
        """
        # Skip if no target or random check fails
        if trg is None or random.random() > self.p:
            return src
        
        # Skip if shapes don't match
        if src.shape != trg.shape:
            return src

        # Perform 2D FFT on H and W dimensions
        fft_src = torch.fft.fftn(src, dim=(-2, -1))
        fft_trg = torch.fft.fftn(trg, dim=(-2, -1))

        # Shift zero-frequency component to center (low-freq becomes central)
        fft_src = torch.fft.fftshift(fft_src, dim=(-2, -1))
        fft_trg = torch.fft.fftshift(fft_trg, dim=(-2, -1))

        # Extract amplitude and phase
        amp_src = torch.abs(fft_src)
        pha_src = torch.angle(fft_src)
        amp_trg = torch.abs(fft_trg)

        # Calculate replacement region size
        _, h, w = src.shape
        beta = random.uniform(self.beta_range[0], self.beta_range[1])
        b_h = max(1, int(h * beta))
        b_w = max(1, int(w * beta))

        # Calculate center region bounds
        c_h, c_w = h // 2, w // 2
        h1, h2 = c_h - b_h, c_h + b_h
        w1, w2 = c_w - b_w, c_w + b_w

        # Replace center (low-frequency) amplitude with target's amplitude
        amp_src[:, h1:h2, w1:w2] = amp_trg[:, h1:h2, w1:w2]

        # Reconstruct complex spectrum with mixed amplitude and source phase
        fft_new = torch.polar(amp_src, pha_src)

        # Inverse shift and inverse FFT
        fft_new = torch.fft.ifftshift(fft_new, dim=(-2, -1))
        out = torch.fft.ifftn(fft_new, dim=(-2, -1)).real

        # Clamp to valid range [0, 1]
        return torch.clamp(out, 0.0, 1.0)
