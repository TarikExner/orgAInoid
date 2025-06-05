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
    def __init__(self, latent_dim=350, salient_idx: range = range(14)):
        """
        Disentangler, as per Table 4 (“Supplementary Table 4: Disentangler architecture”).

        Args:
          latent_dim:  Number of latent dimensions (350 in the paper).
          salient_idx: Sequence of 14 integers (indices in [0..latent_dim−1]) indicating
                       which subset of the latent vector to feed into the “salient” head.

        Architecture:
          1) Conv2D(3→64, 3×3, stride=1, padding=1), Spectral Norm, no BN, ReLU
          2) ResBlockDown(64→512)               → [B,512,32,32]
          3) ResBlockDown(512→512)              → [B,512,16,16]
          4) ResBlockDown(512→1024)             → [B,1024, 8, 8]
          5) ResBlockDown(1024→1024)            → [B,1024, 4, 4]
          6) Flatten (B, 1024,4,4) → (B, 16 384)
          7) Swish on 16 384
          8) Dense(16 384→350), no BN, no SN, then Softmax → this is the “350‐way output”
          9) Separate “salient” head: z[:, salient_idx] → (B,14) → Dense(14→1, SN) → Sigmoid.

        Returns:
          logits_350:  FloatTensor of shape (B,350), after Softmax
          head_out:   FloatTensor of shape (B, 1), after Sigmoid on (B,14)→(B,1).
        """
        super().__init__()

        # ─── 1) First Conv: 3→64, 3×3, stride=1, pad=1, SN, ReLU ─────────────
        self.initial_conv = sn_conv(3, 64, k=3, s=1, p=1)

        # ─── 2) Four ResBlockDown blocks with the exact channel transitions ─────
        #    (64→512), (512→512), (512→1024), (1024→1024), each with stride=2 on the first conv
        self.res1 = ResBlockDown(64, 512)   # → [B,512, 32, 32]
        self.res2 = ResBlockDown(512, 512)   # → [B,512, 16, 16]
        self.res3 = ResBlockDown(512, 1024)  # → [B,1024, 8,  8]
        self.res4 = ResBlockDown(1024, 1024)  # → [B,1024, 4,  4]

        # ─── 3) Flatten the (1024,4,4) → length = 1024*4*4 = 16 384 ──────────────
        self.flatten = nn.Flatten()

        # ─── 4) Swish on that 16 384‐dim vector ─────────────────────────────────
        #     (we’ll implement in forward as x * sigmoid(x))
        #     Then Dense(16 384→350), followed by Softmax
        self.fc_logits = nn.Linear(1024 * 4 * 4, latent_dim)  # no SN here, no BN

        # ─── 5) Separate Salient‐Head (14→1, with SN + Sigmoid) ────────────────
        self.salient_idx = list(salient_idx)
        self.head        = sn_linear(len(self.salient_idx), 1)

    def forward(self, x, z):
        """
        Args:
          x: (B, 3, 224, 224) – image in DenseNet/VGG‐normalized space
          z: (B, 350)       – latent vector from Encoder

        Returns:
          (logits_350, head_out):
            logits_350: (B,350) after Softmax
            head_out:   (B,1)  after Sigmoid on the 14‐dim salient slice
        """
        # 1) Initial Conv + ReLU
        out = self.initial_conv(x)    # → [B, 64, 224, 224]
        out = F.relu(out, inplace=True)

        # 2) Four ResBlockDowns
        out = self.res1(out)          # → [B, 512, 32,  32]
        out = self.res2(out)          # → [B, 512, 16,  16]
        out = self.res3(out)          # → [B,1024, 8,   8]
        out = self.res4(out)          # → [B,1024, 4,   4]

        # 3) Flatten (1024×4×4 → 16 384)
        out = self.flatten(out)       # → [B, 1024*4*4 = 16384]

        # 4) Swish on 16 384 dims:  x * sigmoid(x)
        out = out * torch.sigmoid(out)  # → [B,16384]

        # 5) Dense(16384→350) to get raw 350 logits, then Softmax
        logits_350 = self.fc_logits(out)       # → [B, 350]
        logits_350 = F.softmax(logits_350, dim=1)  # Softmax over the 350 dims

        # 6) Salient head: pick only the 14 specified latent dims from z
        salient = z[:, self.salient_idx]       # → [B, 14]
        head_logit = self.head(salient)        # → [B, 1]   (SN‐Linear(14→1))
        head_out   = torch.sigmoid(head_logit) # → [B, 1]

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
