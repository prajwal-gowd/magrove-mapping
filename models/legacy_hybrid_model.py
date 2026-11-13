"""Legacy TCCFNet architecture used by the checked-in baseline checkpoint.

The current training model includes UNet skip connections, while the available
``checkpoints/best_model.pth`` was trained with the original sequential ResNet
backbone and decoder. Keep this class alongside the current model so the
existing checkpoint remains usable for inference.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

import config


class LegacyResNetBackbone(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = resnet50(weights=weights)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.avgpool = resnet.avgpool

    def forward(self, x):
        spatial = self.features(x)
        pooled = torch.flatten(self.avgpool(spatial), 1)
        return spatial, pooled


class LegacyPositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height, width):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, height * width, d_model))

    def forward(self, x):
        return x + self.pos_embedding


class LegacyTransformerModule(nn.Module):
    def __init__(self, d_model=2048, num_heads=8, num_layers=2,
                 dim_feedforward=512, dropout=0.1, feat_h=7, feat_w=7):
        super().__init__()
        self.d_model = d_model
        self.pos_encoder = LegacyPositionalEncoding2D(d_model, feat_h, feat_w)
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
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x):
        batch, channels, height, width = x.shape
        x = x.reshape(batch, channels, -1).permute(0, 2, 1)
        x = self.pos_encoder(x)
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer_encoder(x)
        return x[:, 0, :]


class LegacyHybridTCCFNet(nn.Module):
    """Original TCCFNet matching the state-dict layout of best_model.pth."""

    def __init__(self, num_classes=1, backbone_freeze=False,
                 pretrained_backbone=False):
        super().__init__()
        self.cnn = LegacyResNetBackbone(pretrained=pretrained_backbone)
        if backbone_freeze:
            for param in self.cnn.parameters():
                param.requires_grad = False

        self.transformer = LegacyTransformerModule(
            d_model=2048,
            num_heads=config.TRANSFORMER_HEADS,
            num_layers=config.TRANSFORMER_LAYERS,
            dim_feedforward=config.TRANSFORMER_DIM_FEEDFORWARD,
            dropout=config.DROPOUT,
            feat_h=7,
            feat_w=7,
        )

        dropout = getattr(config, "DECODER_DROPOUT", 0.0)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4096, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, x):
        spatial, _ = self.cnn(x)
        context = self.transformer(spatial).unsqueeze(-1).unsqueeze(-1)
        context = context.expand(-1, -1, spatial.shape[-2], spatial.shape[-1])
        fused = torch.cat((spatial, context), dim=1)
        return self.decoder(fused)
