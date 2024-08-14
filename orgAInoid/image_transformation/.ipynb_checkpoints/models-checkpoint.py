import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels=1, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.embed_dim = embed_dim

        # Linear projection of flattened patches
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [batch_size, seq_length, height, width]
        batch_size, seq_length, height, width = x.shape

        # Reshape to treat each image in the sequence individually
        x = x.reshape(batch_size * seq_length, 1, height, width)  # (batch_size * seq_length, 1, height, width)

        # Apply convolution to each patch
        x = self.proj(x)  # (batch_size * seq_length, embed_dim, num_patches_height, num_patches_width)
        
        # Flatten the spatial dimensions
        x = x.flatten(2)  # (batch_size * seq_length, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch_size * seq_length, num_patches, embed_dim)

        # Reshape back to the sequence form
        x = x.reshape(batch_size, seq_length * self.num_patches, self.embed_dim)  # (batch_size, seq_length * num_patches, embed_dim)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class ImageTransformer(nn.Module):
    def __init__(self, img_size, patch_size, subseries_length, embed_dim, num_heads, num_layers, dropout=0.1):
        super(ImageTransformer, self).__init__()

        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels=1, embed_dim=embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        
        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=512, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers, enable_nested_tensor=True)
        
        # Decoder to project back to the image space
        self.decoder = nn.Linear(embed_dim, patch_size * patch_size)
        
        self.img_size = img_size  # img_size should be a tuple (height, width)
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

    def forward(self, x):
        # Step 1: Patch Embedding
        # Input x: [batch_size, subseries_length, height, width]
        batch_size = x.size(0)

        x = self.patch_embedding(x)  # [batch_size, subseries_length * num_patches, embed_dim]

        # Step 2: Positional Encoding
        x = self.pos_encoder(x)  # Adds positional information to the patch embeddings
        
        # Step 3: Transformer Encoder
        # Transformer expects input of shape (seq_length, batch_size, embed_dim)
        x = x.transpose(0, 1)  # [seq_length, batch_size, embed_dim]
        x = self.transformer_encoder(x)  # [seq_length, batch_size, embed_dim]
        x = x.transpose(0, 1)  # [batch_size, seq_length, embed_dim]
        
        # Step 4: Decoding
        # Decode each patch embedding back to its original patch size
        x = self.decoder(x)  # [batch_size, seq_length, patch_size * patch_size]
        
        x = x.view(batch_size, -1, self.patch_size, self.patch_size)  # [5, 320, 16, 16]

        # Focus only on the last set of patches (n+1 image)
        x = x[:, -self.num_patches:, :, :]  # [batch_size, num_patches, patch_size, patch_size]
    
        # Step 6: Reconstruct the n+1 image from patches
        num_patches_per_dim_height = self.img_size[0] // self.patch_size
        num_patches_per_dim_width = self.img_size[1] // self.patch_size
    
        x = x.view(batch_size, num_patches_per_dim_height, num_patches_per_dim_width, self.patch_size, self.patch_size)
    
        # Permute and reshape to form the final predicted image
        x = x.permute(0, 1, 3, 2, 4).contiguous().view(batch_size, 1, self.img_size[0], self.img_size[1])
    
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
