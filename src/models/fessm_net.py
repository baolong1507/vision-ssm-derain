import torch
import torch.nn as nn

from .unet_baseline import ConvBlock
from .blocks.freq import FreqEnhance
from .blocks.ssm2d import SSM2DBlock


class FESSMNet(nn.Module):
    def __init__(
        self,
        in_ch=3,
        out_ch=3,
        base_ch=48,
        freq_ch=16,
        ssm_mode="convscan",
        use_freq=True,
        use_ssm_bottleneck=True,
        use_ssm_decoder=True,
    ):
        super().__init__()

        self.enc1 = ConvBlock(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.pool3 = nn.MaxPool2d(2)

        bot_layers = [ConvBlock(base_ch * 4, base_ch * 8)]
        if use_ssm_bottleneck:
            bot_layers.append(SSM2DBlock(base_ch * 8, mode=ssm_mode))
        bot_layers.append(ConvBlock(base_ch * 8, base_ch * 8))
        self.bot = nn.Sequential(*bot_layers)

        self.fe1 = FreqEnhance(base_ch, freq_ch) if use_freq else nn.Identity()
        self.fe2 = FreqEnhance(base_ch * 2, freq_ch) if use_freq else nn.Identity()
        self.fe3 = FreqEnhance(base_ch * 4, freq_ch) if use_freq else nn.Identity()

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        dec3_layers = [ConvBlock(base_ch * 8, base_ch * 4)]
        if use_ssm_decoder:
            dec3_layers.append(SSM2DBlock(base_ch * 4, mode=ssm_mode))
        self.dec3 = nn.Sequential(*dec3_layers)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        dec2_layers = [ConvBlock(base_ch * 4, base_ch * 2)]
        if use_ssm_decoder:
            dec2_layers.append(SSM2DBlock(base_ch * 2, mode=ssm_mode))
        self.dec2 = nn.Sequential(*dec2_layers)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        dec1_layers = [ConvBlock(base_ch * 2, base_ch)]
        if use_ssm_decoder:
            dec1_layers.append(SSM2DBlock(base_ch, mode=ssm_mode))
        self.dec1 = nn.Sequential(*dec1_layers)

        self.out = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bot(self.pool3(e3))

        e1 = self.fe1(e1)
        e2 = self.fe2(e2)
        e3 = self.fe3(e3)

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        y = self.out(d1)
        return torch.sigmoid(y)