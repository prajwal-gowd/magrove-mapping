import torch
import torch.nn as nn
import math


class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height, width):
        super(PositionalEncoding2D, self).__init__()
        # Simplified 1D absolute positional encoding for flattened 2D patches
        self.pos_embedding = nn.Parameter(torch.randn(1, height * width, d_model))

    def forward(self, x):
        """
        x: [B, N, C]
        """
        return x + self.pos_embedding


class TransformerModule(nn.Module):
    """
    Vision Transformer module that processes CNN spatial features
    and returns spatially-enriched attention-refined features.

    Unlike the original that only returned a CLS token (discarding
    all spatial information), this version returns the full set of
    transformed spatial tokens reshaped back to spatial dimensions.
    Each spatial position is enriched with global context from
    self-attention across all positions.
    """

    def __init__(self, d_model=2048, num_heads=8, num_layers=2,
                 dim_feedforward=512, dropout=0.1, feat_h=7, feat_w=7):
        super(TransformerModule, self).__init__()

        self.d_model = d_model
        self.feat_h = feat_h
        self.feat_w = feat_w
        self.pos_encoder = PositionalEncoding2D(d_model, feat_h, feat_w)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # A simple learnable CLS token to aggregate info, similar to ViT
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x):
        """
        Args:
            x (Tensor): CNN spatial features [B, C, H, W]
        Returns:
            Tensor: Spatially-refined features [B, C, H, W]
                    Each spatial position is enriched with global context
                    from self-attention across all positions.
        """
        B, C, H, W = x.shape

        # Flatten spatial dimensions: [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
        x = x.view(B, C, -1).permute(0, 2, 1)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Expand CLS token to match batch size
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, C]

        # Concatenate CLS token and spatial features
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 1 + H*W, C]

        # Pass through transformer
        x = self.transformer_encoder(x)

        # Extract spatial tokens (skip CLS), reshape back to spatial dims
        spatial_out = x[:, 1:, :]                                    # [B, H*W, C]
        spatial_out = spatial_out.permute(0, 2, 1).view(B, C, H, W)  # [B, C, H, W]

        return spatial_out


if __name__ == "__main__":
    # Quick test
    model = TransformerModule(
        d_model=2048, num_heads=8, num_layers=2,
        dim_feedforward=512, dropout=0.1, feat_h=7, feat_w=7,
    )
    dummy_input = torch.randn(2, 2048, 7, 7)
    out = model(dummy_input)
    print("Transformer output shape:", out.shape)  # Expected: [2, 2048, 7, 7]
