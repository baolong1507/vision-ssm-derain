import torch

def psnr(pred, target, eps=1e-8):
    mse = torch.mean((pred - target) ** 2)
    return 10.0 * torch.log10(1.0 / (mse + eps))
