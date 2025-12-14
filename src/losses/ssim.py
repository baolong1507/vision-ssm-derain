import torch
import torch.nn.functional as F

def _gaussian(window_size, sigma):
    gauss = torch.tensor([-(x - window_size//2)**2 / float(2*sigma**2) for x in range(window_size)])
    gauss = torch.exp(gauss)
    return gauss / gauss.sum()

def _create_window(window_size=11, sigma=1.5, channel=3, device="cpu"):
    _1d = _gaussian(window_size, sigma).to(device=device).unsqueeze(1)
    _2d = (_1d @ _1d.t()).unsqueeze(0).unsqueeze(0)
    window = _2d.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(pred, target, window_size=11, sigma=1.5, C1=0.01**2, C2=0.03**2):
    device = pred.device
    channel = pred.size(1)
    window = _create_window(window_size, sigma, channel, device)

    mu1 = F.conv2d(pred, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size//2, groups=channel) - mu12

    ssim_map = ((2*mu12 + C1) * (2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
