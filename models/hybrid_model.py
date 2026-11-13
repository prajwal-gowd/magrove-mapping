import torch
import torch.nn as nn
from .resnet_backbone import ResNetBackbone
from .transformer_encoder import TransformerModule
import config


class DecoderBlock(nn.Module):
    """Double-conv block used in UNet-style decoder stages."""

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class HybridTCCFNet(nn.Module):
    """
    Improved HybridTCCFNet with UNet-style skip connections.

    Key improvements over v1:
    1. Skip connections from ResNet layers 1-3 to decoder stages,
       preserving fine-grained spatial detail for boundary segmentation.
    2. Transformer returns spatially-refined features (not just CLS token),
       so attention patterns are preserved in the feature fusion.
    3. 1x1 bottleneck convolution reduces the fused 4096-channel features
       before decoding, cutting parameters by ~14M.

    Architecture:
        Encoder: ResNet-50 backbone -> multi-scale features
        Transformer: Self-attention on deepest features (7x7)
        Fusion: Concatenate CNN + Transformer spatial features
        Decoder: UNet-style progressive upsampling with skip connections
    """

    def __init__(self, num_classes=1, backbone_freeze=False, pretrained_backbone=True):
        super(HybridTCCFNet, self).__init__()

        self.cnn = ResNetBackbone(pretrained=pretrained_backbone)

        if backbone_freeze:
            for param in self.cnn.parameters():
                param.requires_grad = False

        # ResNet50 output channels = 2048
        self.transformer = TransformerModule(
            d_model=2048,
            num_heads=config.TRANSFORMER_HEADS,
            num_layers=config.TRANSFORMER_LAYERS,
            dim_feedforward=config.TRANSFORMER_DIM_FEEDFORWARD,
            dropout=config.DROPOUT,
            feat_h=7,  # Expected ResNet50 spatial output size for 224x224 input
            feat_w=7,
        )

        dec_drop = getattr(config, "DECODER_DROPOUT", 0.2)

        # Bottleneck: reduce fused channels (4096 -> 1024) with 1x1 conv
        self.bottleneck = nn.Sequential(
            nn.Conv2d(4096, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        # ---- UNet Decoder with Skip Connections ----
        # Stage 3: 7x7 -> 14x14, skip from layer3 (1024ch)
        self.up3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec3 = DecoderBlock(512 + 1024, 512, dropout=dec_drop)

        # Stage 2: 14x14 -> 28x28, skip from layer2 (512ch)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = DecoderBlock(256 + 512, 256, dropout=dec_drop)

        # Stage 1: 28x28 -> 56x56, skip from layer1 (256ch)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = DecoderBlock(128 + 256, 128, dropout=dec_drop)

        # Stage 0: 56x56 -> 112x112 (no skip connection)
        self.up0 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec0 = DecoderBlock(64, 64)

        # Final: 112x112 -> 224x224
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def freeze_backbone(self):
        """Freeze CNN backbone parameters (for initial decoder-only training)."""
        for param in self.cnn.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze CNN backbone parameters for full fine-tuning."""
        for param in self.cnn.parameters():
            param.requires_grad = True

    def get_param_groups(self, backbone_lr, decoder_lr):
        """
        Return parameter groups with differential learning rates.
        Lower LR for pretrained backbone, higher for decoder/transformer.
        """
        backbone_params = list(self.cnn.parameters())
        backbone_ids = set(id(p) for p in backbone_params)
        other_params = [p for p in self.parameters() if id(p) not in backbone_ids]
        return [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": other_params, "lr": decoder_lr},
        ]

    def forward(self, x):
        """
        Forward pass for the hybrid segmentation model.
        Args:
            x: Input images [B, 3, 224, 224]
        Returns:
            Spatial probability logits [B, 1, 224, 224]
        """
        # ---- Encoder ----
        # x4: deepest features [B, 2048, 7, 7]
        # skips: [skip1(256,56,56), skip2(512,28,28), skip3(1024,14,14)]
        x4, _, skips = self.cnn(x)
        skip1, skip2, skip3 = skips

        # ---- Transformer ----
        # Spatially-refined features via self-attention [B, 2048, 7, 7]
        transformer_features = self.transformer(x4)

        # ---- Feature Fusion ----
        fused = torch.cat([x4, transformer_features], dim=1)  # [B, 4096, 7, 7]
        fused = self.bottleneck(fused)                         # [B, 1024, 7, 7]

        # ---- Decoder with Skip Connections ----
        d3 = self.up3(fused)                        # [B, 512, 14, 14]
        d3 = torch.cat([d3, skip3], dim=1)          # [B, 1536, 14, 14]
        d3 = self.dec3(d3)                          # [B, 512, 14, 14]

        d2 = self.up2(d3)                           # [B, 256, 28, 28]
        d2 = torch.cat([d2, skip2], dim=1)          # [B, 768, 28, 28]
        d2 = self.dec2(d2)                          # [B, 256, 28, 28]

        d1 = self.up1(d2)                           # [B, 128, 56, 56]
        d1 = torch.cat([d1, skip1], dim=1)          # [B, 384, 56, 56]
        d1 = self.dec1(d1)                          # [B, 128, 56, 56]

        d0 = self.up0(d1)                           # [B, 64, 112, 112]
        d0 = self.dec0(d0)                          # [B, 64, 112, 112]

        out = self.final_up(d0)                     # [B, 32, 224, 224]
        logits = self.head(out)                     # [B, 1, 224, 224]

        return logits


def run_dummy_forward():
    print("Running dummy forward pass to test execution...")
    model = HybridTCCFNet(num_classes=config.NUM_CLASSES)
    model.eval()

    # Create a random input tensor simulating a batch of 2 RGB images of size 224x224
    dummy_input = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        output = model(dummy_input)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Output shape: {output.shape} (Expected: [2, {config.NUM_CLASSES}, 224, 224])")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("Dummy forward pass completed successfully.")


if __name__ == "__main__":
    run_dummy_forward()
