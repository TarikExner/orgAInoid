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

from typing import Tuple, Union

def sn_conv(c_in, c_out, k=3, s=1, p=1):
    return nn.utils.spectral_norm(nn.Conv2d(c_in, c_out, k, s, p, bias=False))

def sn_conv_t(c_in, c_out, k=4, s=2, p=1, op=0):
    return nn.utils.spectral_norm(nn.ConvTranspose2d(c_in, c_out, k, s, p, output_padding= op, bias=False))

def sn_linear(f_in, f_out):
    return nn.utils.spectral_norm(nn.Linear(f_in, f_out, bias=True))

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ResBlockDown(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        # Main path: BN → ReLU → Conv(stride=2, 3×3) → BN → ReLU → Conv(stride=1, 3×3)
        self.bn1   = nn.BatchNorm2d(ch_in)
        self.conv1 = sn_conv(ch_in, ch_out, k=3, s=2, p=1)
        self.bn2   = nn.BatchNorm2d(ch_out)
        self.conv2 = sn_conv(ch_out, ch_out, k=3, s=1, p=1)
        # Skip path: 1×1 Conv(stride=2)
        self.skip  = sn_conv(ch_in, ch_out, k=1, s=2, p=0)

    def forward(self, x):
        out = self.bn1(x)
        out = F.relu(out, inplace=True)
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        sc  = self.skip(x)
        return out + sc

class ResBlockUp(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        # Main path: BN → ReLU → ConvTranspose(stride=2, 3×3) → BN → ReLU → ConvTranspose(stride=1, 3×3)
        self.bn1   = nn.BatchNorm2d(ch_in)
        self.conv1 = sn_conv_t(ch_in, ch_out, k=3, s=2, p=1, op=1)
        self.bn2   = nn.BatchNorm2d(ch_out)
        self.conv2 = sn_conv_t(ch_out, ch_out, k=3, s=1, p=1, op=0)
        # Skip path: 1×1 ConvTranspose(stride=2)
        self.skip  = sn_conv_t(ch_in, ch_out, k=1, s=2, p=0, op=1)

    def forward(self, x):
        out = self.bn1(x)
        out = F.relu(out, inplace=True)
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        sc  = self.skip(x)
        return out + sc

class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=350, base_channels=128, img_size=224):
        super().__init__()
        assert img_size % 16 == 0, "img_size must be divisible by 16"
        self.img_size      = img_size
        self.init_res      = img_size // 16
        c = base_channels

        # 1) Stem: single 3×3 conv (stride=1, padding=1) → ReLU
        self.stem = nn.Sequential(
            sn_conv(in_channels, c, k=3, s=1, p=1),
            nn.ReLU(inplace=True)
        )

        # 2) Four ResBlockDown blocks (each halves spatial)
        self.layer1 = ResBlockDown(c, c)
        self.layer2 = ResBlockDown(c, c * 2)
        self.layer3 = ResBlockDown(c * 2, c * 4)
        self.layer4 = ResBlockDown(c * 4, c * 8)

        self.fc = nn.Linear(c * 8 * self.init_res * self.init_res, latent_dim)
        self.swish = nn.SiLU()  # Swish activation

    def forward(self, x):
        # x: (B, in_channels, img_size, img_size)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # flatten exactly (B, 8c, init_res, init_res) → (B, 8c*init_res^2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.swish(x)

class Decoder(nn.Module):
    def __init__(self, out_channels=3, latent_dim=350, base_channels=128, img_size=224):
        super().__init__()
        assert img_size % 16 == 0, "img_size must be divisible by 16"
        self.img_size      = img_size
        self.init_res      = img_size // 16
        c = base_channels

        # 1) Project latent_dim → (c*8 * init_res * init_res), then reshape to (B, c*8, init_res, init_res)
        self.fc = nn.Linear(latent_dim, c * 8 * self.init_res * self.init_res)

        # 2) Four ResBlockUp blocks (each doubles spatial)
        self.up1 = ResBlockUp(c * 8, c * 8)
        self.up2 = ResBlockUp(c * 8, c * 4)
        self.up3 = ResBlockUp(c * 4, c * 2)
        self.up4 = ResBlockUp(c * 2, c)

        # 3) Final 3×3 conv (stride=1, padding=1) → BN → Sigmoid to get (out_channels, img_size, img_size)
        self.final_conv = sn_conv(c, out_channels, k=3, s=1, p=1)
        self.final_bn   = nn.BatchNorm2d(out_channels)

    def forward(self, z):
        # z: (B, latent_dim)
        b = z.size(0)

        x = self.fc(z).view(b, -1, self.init_res, self.init_res)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        x = self.final_conv(x)
        x = self.final_bn(x)
        return x

class PatchGANDiscriminator(nn.Module):
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

class MLPDiscriminator(nn.Module):
    def __init__(self, latent_dim=350):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048,   1),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

    def forward(self, z):
        # z: [B, 350]
        return self.net(z).view(-1)  # [B]

class Disentangler(nn.Module):
    """
    Disentangler for 3×224×224 input, matching Supplementary Table 4 (but adapted
    so that the final feature map is always 1024×4×4 before flattening).

    Args:
      latent_dim:  Number of latent dimensions (350 in the paper).
      salient_idx: Sequence of 14 indices in [0..latent_dim−1] for the “salient” head.

    Architecture:
      1) Conv2D(3→64, 3×3, stride=1, pad=1), SN, no BN, ReLU
      2) ResBlockDown(64→512)    → (B,512,H/2,W/2)
      3) ResBlockDown(512→512)   → (B,512,H/4,W/4)
      4) ResBlockDown(512→1024)  → (B,1024,H/8,W/8)
      5) ResBlockDown(1024→1024) → (B,1024,H/16,W/16)
      6) AdaptiveAvgPool2d((4,4)) → (B,1024,4,4)
      7) Flatten → (B, 1024*4*4 = 16 384)
      8) Swish (x * sigmoid(x)) → (B, 16 384)
      9) Linear(16 384→350) → logits_350, then Softmax → (B,350)
     10) Salient head: take z[:, salient_idx] (B,14) → SNLinear(14→1) → Sigmoid → (B,1)

    In forward(x,z) we return (logits_350, head_out).  Here x∈ℝ^{B×3×224×224},
    and z∈ℝ^{B×latent_dim} is only used by the “salient” head (step 10).
    """
    def __init__(self, latent_dim: int = 350, salient_idx: range = range(14)):
        super().__init__()
        self.latent_dim = latent_dim
        self.salient_idx = list(salient_idx)

        # 1) Initial Conv: 3→64, 3×3, stride=1, pad=1, SN, no BN, ReLU
        self.initial_conv = sn_conv(3, 64, k=3, s=1, p=1)

        # 2) Four ResBlockDown layers (64→512→512→1024→1024)
        self.res1 = ResBlockDown(64,   512)   # -> (B,512,112,112)
        self.res2 = ResBlockDown(512,  512)   # -> (B,512, 56, 56)
        self.res3 = ResBlockDown(512,  1024)  # -> (B,1024,28, 28)
        self.res4 = ResBlockDown(1024, 1024)  # -> (B,1024,14, 14)

        # 3) Force final map to (4×4) regardless of input:
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # → (B,1024,4,4)

        # 4) Flatten (B,1024,4,4) → (B, 16 384)
        self.flatten = nn.Flatten()

        # 5) Swish on 16 384 dims
        #    We'll apply `x * sigmoid(x)` inside forward.

        # 6) Dense(16 384→350), no SN, no BN
        self.fc_logits = nn.Linear(1024 * 4 * 4, latent_dim)

        # 7) Salient head: SNLinear(14→1)
        self.head = sn_linear(len(self.salient_idx), 1)

    def forward(self, x: torch.Tensor, z: torch.Tensor):
        """
        Args:
          x: (B, 3, 224, 224) — diff‐image, already normalized.
          z: (B, latent_dim)  — latent vector from encoder, used only by the salient head.

        Returns:
          logits_350: FloatTensor (B, 350), after Softmax
          head_out:   FloatTensor (B, 1), after Sigmoid on the salient slice
        """
        # 1) Initial conv + ReLU  → (B, 64, 224, 224)
        out = self.initial_conv(x)
        out = F.relu(out, inplace=True)

        # 2) Four ResBlockDowns → eventually (B,1024,14,14)
        out = self.res1(out)     # → (B, 512, 112, 112)
        out = self.res2(out)     # → (B, 512,  56,  56)
        out = self.res3(out)     # → (B,1024,  28,  28)
        out = self.res4(out)     # → (B,1024,  14,  14)

        # 3) Adaptive average pool to (4×4)
        out = self.adaptive_pool(out)  # → (B,1024, 4, 4)

        # 4) Flatten → (B, 1024*4*4 = 16384)
        out = self.flatten(out)       # → (B,16384)

        # 5) Swish: x * sigmoid(x)
        out = out * torch.sigmoid(out)  # → (B,16384)

        # 6) Dense(16384→350) → Softmax over dim=1
        logits_350 = self.fc_logits(out)        # → (B,350)
        logits_350 = F.softmax(logits_350, dim=1)

        # 7) Salient head: pick 14 dims from z, apply SNLinear(14→1), then Sigmoid
        salient = z[:, self.salient_idx]       # → (B,14)
        head_logit = self.head(salient)        # → (B, 1)
        head_out   = torch.sigmoid(head_logit) # → (B, 1)

        return logits_350, head_out

def build_models(*,
                 input_channels=3,
                 img_size=224,
                 latent_dim=350,
                 patch_gan: bool = False,
                 base_channels=128) -> Tuple[Encoder, Decoder, Union[PatchGANDiscriminator, MLPDiscriminator], Disentangler]:
    enc = Encoder(input_channels, latent_dim, base_channels)
    dec = Decoder(input_channels, latent_dim, base_channels, img_size)
    if patch_gan:
        disc = PatchGANDiscriminator(input_channels, base_channels)
    else:
        disc = MLPDiscriminator(latent_dim)
    dis = Disentangler(latent_dim)
    return enc, dec, disc, dis
