import torch
import torch.nn as nn
import math

import torch.nn.functional as F

import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=1, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches_per_image = (img_size // patch_size) ** 2
        self.num_images = 5  # Number of images in the input sequence

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, num_images, in_channels, img_size, img_size)
        Returns:
            patches: Tensor of shape (batch_size, num_images * num_patches_per_image, embed_dim)
        """
        batch_size, num_images, in_channels, img_size, _ = x.shape
        x = x.view(batch_size * num_images, in_channels, img_size, img_size)  # (B*num_images, C, H, W)
        x = self.proj(x)  # (B*num_images, embed_dim, H/patch, W/patch)
        x = x.flatten(2)  # (B*num_images, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B*num_images, num_patches, embed_dim)
        x = x.reshape(batch_size, num_images * self.num_patches_per_image, -1)  # (B, num_images*num_patches, embed_dim)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_length):
        super(PositionalEncoding, self).__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_length, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_length, embed_dim)
        Returns:
            x + pos_embed: Tensor of shape (batch_size, seq_length, embed_dim)
        """
        return x + self.pos_embed[:, :x.size(1), :]

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_length, batch_size, embed_dim)
        Returns:
            x: Tensor of shape (seq_length, batch_size, embed_dim)
        """
        # Self-attention block
        attn_output, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.dropout1(attn_output)
        
        # MLP block
        mlp_output = self.mlp(self.norm2(x))
        x = x + mlp_output
        return x

class VisionTransformerPredictor(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=1,
        num_input_images=5,
        num_predicted_images=5,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        mlp_dim=3072,
        dropout=0.1
    ):
        super(VisionTransformerPredictor, self).__init__()
        self.num_input_images = num_input_images
        self.num_predicted_images = num_predicted_images
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        num_patches = self.patch_embed.num_patches_per_image * num_input_images
        
        self.pos_embed = PositionalEncoding(embed_dim, max_seq_length=num_patches)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer Encoder
        self.transformer = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Prediction Head
        # For predicting multiple images, we'll output a flattened tensor representing all predicted images
        self.pred_head = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_predicted_images * img_size * img_size)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, num_input_images, in_channels, img_size, img_size)
        Returns:
            predicted_images: Tensor of shape (batch_size, num_predicted_images, 1, img_size, img_size)
        """
        batch_size = x.size(0)

        # Patch Embedding
        x = self.patch_embed(x)  # (B, num_input_images*num_patches, embed_dim)

        # CLS token prepending
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 1 + num_input_images*num_patches, embed_dim)

        # Positional Encoding
        x = self.pos_embed(x)  # positional embedding includes the cls token position
        x = self.dropout(x)

        # Prepare input for Transformer: (seq_length, batch_size, embed_dim)
        x = x.transpose(0, 1)

        # Transformer Encoder
        for layer in self.transformer:
            x = layer(x)

        x = self.norm(x)  # (seq_length, batch_size, embed_dim)

        # Extract the CLS token representation for predictions
        cls_output = x[0]  # (batch_size, embed_dim)

        # Prediction Head
        x = self.pred_head(cls_output)  # (batch_size, num_predicted_images * img_size * img_size)
        x = x.view(batch_size, self.num_predicted_images, 1, self.patch_embed.img_size, self.patch_embed.img_size)

        return x


class ImageTransformerV2(nn.Module):
    def __init__(self, img_size, patch_size, subseries_length, embed_dim, num_heads, num_layers, dropout, prediction_length=10):
        super(ImageTransformerV2, self).__init__()
        self.img_size = img_size  # (height, width)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.prediction_length = prediction_length  # Number of future images to predict

        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.patch_dim = patch_size * patch_size

        # Patch Embedding
        self.patch_embedding = nn.Conv2d(in_channels=1, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional Encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, subseries_length * self.num_patches, embed_dim))

        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers, enable_nested_tensor=True)

        # Transformer Decoder for multiple timepoints
        self.decoder = nn.Linear(embed_dim, self.patch_dim * prediction_length)

    def forward(self, x):
        # Step 1: Patch Embedding
        # Input x: [batch_size, subseries_length, height, width]
        batch_size, subseries_length, height, width = x.size()
    
        # Add a channel dimension to match the expected input of the Conv2d layer
        x = x.unsqueeze(2)  # Now x.shape: [batch_size, subseries_length, 1, height, width]
    
        # Flatten the subseries_length into the batch dimension to process patches
        x = x.view(-1, 1, height, width)  # Flatten subseries_length into the batch dimension: [batch_size * subseries_length, 1, height, width]
    
        # Apply patch embedding
        x = self.patch_embedding(x)  # [batch_size * subseries_length, embed_dim, height/patch_size, width/patch_size]
    
        # Flatten patches
        x = x.flatten(2)  # [batch_size * subseries_length, embed_dim, num_patches]
        
        # Transpose to match the transformer input expectations
        x = x.transpose(1, 2)  # [batch_size * subseries_length, num_patches, embed_dim]
    
        # Reshape back to consider subseries length as a separate dimension
        x = x.reshape(batch_size, subseries_length * self.num_patches, self.embed_dim)  # [batch_size, subseries_length * num_patches, embed_dim]
    
        # Step 2: Positional Encoding
        dynamic_pos_encoder = self.pos_encoder[:, :subseries_length * self.num_patches, :]
        x = x + dynamic_pos_encoder  # Adds positional information to the patch embeddings
        
        # Step 3: Transformer Encoder
        # Transformer expects input of shape [seq_length, batch_size, embed_dim]
        x = x.transpose(0, 1)  # [seq_length, batch_size, embed_dim]
        x = self.transformer_encoder(x)  # [seq_length, batch_size, embed_dim]
        x = x.transpose(0, 1)  # [batch_size, seq_length, embed_dim]
        
        # Step 4: Decoding
        # Decode each patch embedding back to its original patch size but for multiple future predictions
        x = self.decoder(x)  # [batch_size, seq_length, patch_size * patch_size * prediction_length]
        
        # Reshape the output to form the predicted patches for each future image
        x = x.view(batch_size, -1, self.patch_size, self.patch_size)  # [batch_size, seq_length * prediction_length, patch_size, patch_size]
    
        # Focus only on the last set of patches for the last image in the predicted sequence
        x = x[:, -self.num_patches * self.prediction_length:, :, :]  # [batch_size, num_patches * prediction_length, patch_size, patch_size]
    
        # Step 6: Reconstruct the predicted images from patches
        num_patches_per_dim_height = self.img_size[0] // self.patch_size
        num_patches_per_dim_width = self.img_size[1] // self.patch_size
        
        # Reshape considering the full sequence of predicted images
        x = x.reshape(batch_size * self.prediction_length, num_patches_per_dim_height, num_patches_per_dim_width, self.patch_size, self.patch_size)
    
        # Permute and reshape to form the final predicted images
        x = x.permute(0, 1, 3, 2, 4).contiguous().view(batch_size, self.prediction_length, self.img_size[0], self.img_size[1])
    
        return x
