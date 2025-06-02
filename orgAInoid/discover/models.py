"""DISCOVER core network modules in PyTorch
------------------------------------------------
Encoder, Decoder (generator) and PatchGAN Discriminator
are implemented as plain `torch.nn.Module` classes.

All architecture hyper‑parameters are exposed through the
constructor so you can tweak input size, latent dimensions,
or channel widths from a single place.  The helper
`build_models()` returns ready‑to‑train instances that share
those settings.

Usage
:::::
>>> from models import build_models
>>> enc, dec, disc = build_models(img_size=224, latent_dim=256)
>>> z = enc(torch.randn(4, 3, 224, 224))  # [B, latent_dim]
>>> x_hat = dec(z)                         # [B, 3, 224, 224]
>>> logits = disc(x_hat)                  # [B, 1, H/16, W/16]

Note
::::
‣ The decoder outputs in [-1, 1] via Tanh so remember to normalise
your images the same way when computing reconstruction losses.
‣ PatchGAN output size depends on the input resolution; that’s OK—you
can average over spatial dims for a scalar real/fake score.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

def sn_conv(c_in, c_out, k=3, s=1, p=1):
    return nn.utils.spectral_norm(nn.Conv2d(c_in, c_out, k, s, p, bias=False))

def sn_conv_t(c_in, c_out, k=4, s=2, p=1):
    return nn.utils.spectral_norm(nn.ConvTranspose2d(c_in, c_out, k, s, p, bias=False))

def sn_linear(f_in, f_out):
    return nn.utils.spectral_norm(nn.Linear(f_in, f_out, bias=True))


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ResidualBlockSN(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, stride: int = 1):
        super().__init__()
        self.conv1 = sn_conv(ch_in, ch_out, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = sn_conv(ch_out, ch_out, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.shortcut = (
            nn.Sequential(sn_conv(ch_in, ch_out, 1, stride, 0), nn.BatchNorm2d(ch_out))
            if stride != 1 or ch_in != ch_out else nn.Identity()
        )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + self.shortcut(x), inplace=True)
        return out


class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=350, base_channels=128):
        super().__init__()
        self.stem = nn.Sequential(
            sn_conv(in_channels, base_channels, 7, 2, 3),
            nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        c = base_channels
        self.layer1 = ResidualBlockSN(c, c)
        self.layer2 = ResidualBlockSN(c, c * 2, stride=2)
        self.layer3 = ResidualBlockSN(c * 2, c * 4, stride=2)
        self.layer4 = ResidualBlockSN(c * 4, c * 8, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = sn_linear(c * 8, latent_dim)
        self.swish = Swish()
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.swish(self.fc(x))


class Decoder(nn.Module):
    def __init__(self, out_channels=3, latent_dim=350, base_channels=128, img_size=224):
        super().__init__()
        init_res = img_size // 32
        self.fc = sn_linear(latent_dim, base_channels * 8 * init_res * init_res)
        c = base_channels
        self.up1 = nn.Sequential(sn_conv_t(c * 8, c * 4), nn.BatchNorm2d(c * 4), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(sn_conv_t(c * 4, c * 2), nn.BatchNorm2d(c * 2), nn.ReLU(inplace=True))
        self.up3 = nn.Sequential(sn_conv_t(c * 2, c), nn.BatchNorm2d(c), nn.ReLU(inplace=True))
        self.up4 = nn.Sequential(sn_conv_t(c, c // 2), nn.BatchNorm2d(c // 2), nn.ReLU(inplace=True))
        self.final = nn.Sequential(sn_conv_t(c // 2, out_channels), nn.Tanh())
        self.init_res = init_res
    def forward(self, z):
        b = z.size(0)
        x = self.fc(z).view(b, -1, self.init_res, self.init_res)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return self.final(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=128):
        super().__init__()
        c = base_channels
        layers = [
            sn_conv(in_channels, c, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            sn_conv(c, c * 2, 4, 2, 1), nn.BatchNorm2d(c * 2), nn.LeakyReLU(0.2, inplace=True),
            sn_conv(c * 2, c * 4, 4, 2, 1), nn.BatchNorm2d(c * 4), nn.LeakyReLU(0.2, inplace=True),
            sn_conv(c * 4, c * 8, 4, 1, 1), nn.BatchNorm2d(c * 8), nn.LeakyReLU(0.2, inplace=True),
            sn_conv(c * 8, 1, 4, 1, 1),
        ]
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

# ─────────────── factory (kept API‑compatible) ────────────────────

def build_models(*,
                 input_channels=3,
                 img_size=224,
                 latent_dim=350,
                 base_channels=128) -> Tuple[Encoder, Decoder, Discriminator]:
    enc = Encoder(input_channels, latent_dim, base_channels)
    dec = Decoder(input_channels, latent_dim, base_channels, img_size)
    disc = Discriminator(input_channels, base_channels)
    return enc, dec, disc
