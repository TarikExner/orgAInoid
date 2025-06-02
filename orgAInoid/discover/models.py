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


class Encoder(nn.Module):
    """Simple ResNet‑like encoder → latent vector."""

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 350,
        base_channels: int = 64) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )  # → H/4, W/4

        # ─── four simple residual blocks ─────────────────────────────────
        def res_block(ch_in: int, ch_out: int, stride: int = 1):
            down = stride != 1 or ch_in != ch_out
            layers = [
                nn.Conv2d(ch_in, ch_out, 3, stride, 1, bias=False),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch_out, ch_out, 3, 1, 1, bias=False),
                nn.BatchNorm2d(ch_out),
            ]
            shortcut = (
                nn.Sequential(
                    nn.Conv2d(ch_in, ch_out, 1, stride, bias=False), nn.BatchNorm2d(ch_out)
                )
                if down
                else nn.Identity()
            )
            return nn.Sequential(*layers, shortcut, nn.ReLU(inplace=True))

        c = base_channels
        self.layer1 = res_block(c, c)  # H/4
        self.layer2 = res_block(c, c * 2, stride=2)  # H/8
        self.layer3 = res_block(c * 2, c * 4, stride=2)  # H/16
        self.layer4 = res_block(c * 4, c * 8, stride=2)  # H/32

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c * 8, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        z = self.fc(x)
        return z  # [B, latent_dim]


class Decoder(nn.Module):
    """Mirror of the encoder using transposed‑convs."""

    def __init__(
        self,
        out_channels: int = 3,
        latent_dim: int = 350,
        base_channels: int = 64,
        img_size: int = 224,
    ) -> None:
        super().__init__()
        self.init_res = img_size // 32
        self.fc = nn.Linear(latent_dim, base_channels * 8 * self.init_res * self.init_res)

        def up_block(ch_in: int, ch_out: int):
            return nn.Sequential(
                nn.ConvTranspose2d(ch_in, ch_out, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
            )

        c = base_channels
        self.up1 = up_block(c * 8, c * 4)  # H/16
        self.up2 = up_block(c * 4, c * 2)  # H/8
        self.up3 = up_block(c * 2, c)  # H/4
        self.up4 = up_block(c, c // 2)  # H/2
        self.final = nn.Sequential(
            nn.ConvTranspose2d(c // 2, out_channels, 4, 2, 1),
            nn.Tanh(),
        )  # H

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(z.size(0), -1, self.init_res, self.init_res)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.final(x)
        return x  # [B, C, H, W]


class Discriminator(nn.Module):
    """70×70 PatchGAN discriminator (hinge loss ready)."""

    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()

        def conv(c_in: int, c_out: int, stride: int = 2):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 4, stride, 1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.LeakyReLU(0.2, inplace=True),
            )

        c = base_channels
        layers = [
            nn.Conv2d(in_channels, c, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            conv(c, c * 2),
            conv(c * 2, c * 4),
            conv(c * 4, c * 8, stride=1),
            nn.Conv2d(c * 8, 1, 4, 1, 0),  # no sigmoid; use hinge / BCE in loss
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def build_models(
    *,
    input_channels: int = 3,
    img_size: int = 224,
    latent_dim: int = 350,
    base_channels: int = 64,
) -> tuple[Encoder, Decoder, Discriminator]:
    """Factory that returns encoder, decoder, discriminator
    that share the same hyper‑parameters.
    """
    enc = Encoder(input_channels, latent_dim, base_channels)
    dec = Decoder(input_channels, latent_dim, base_channels, img_size)
    disc = Discriminator(input_channels, base_channels)
    return enc, dec, disc
