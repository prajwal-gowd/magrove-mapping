import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNetBackbone(nn.Module):
    """
    ResNet-50 backbone with multi-scale feature extraction for UNet-style
    skip connections.

    Returns spatial features, global features, and intermediate skip features
    from layer1, layer2, and layer3.
    """

    def __init__(self, pretrained=True):
        super(ResNetBackbone, self).__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = resnet50(weights=weights)

        # Expose individual stages for skip connections
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.layer1 = resnet.layer1  # -> [B, 256, 56, 56]
        self.layer2 = resnet.layer2  # -> [B, 512, 28, 28]
        self.layer3 = resnet.layer3  # -> [B, 1024, 14, 14]
        self.layer4 = resnet.layer4  # -> [B, 2048, 7, 7]
        self.avgpool = resnet.avgpool

    def forward(self, x):
        """
        Forward pass with multi-scale feature extraction.
        Args:
            x (Tensor): Input images of shape [B, 3, H, W]
        Returns:
            spatial_features (Tensor): Deepest feature maps [B, 2048, 7, 7]
            global_features (Tensor): Global average pooled features [B, 2048]
            skips (list[Tensor]): Intermediate features for skip connections
                - skip1: [B, 256, 56, 56] from layer1
                - skip2: [B, 512, 28, 28] from layer2
                - skip3: [B, 1024, 14, 14] from layer3
        """
        x = self.stem(x)       # [B, 64, 56, 56]
        s1 = self.layer1(x)    # [B, 256, 56, 56]
        s2 = self.layer2(s1)   # [B, 512, 28, 28]
        s3 = self.layer3(s2)   # [B, 1024, 14, 14]
        s4 = self.layer4(s3)   # [B, 2048, 7, 7]

        global_features = self.avgpool(s4)
        global_features = torch.flatten(global_features, 1)  # [B, 2048]

        return s4, global_features, [s1, s2, s3]


if __name__ == "__main__":
    # Quick test
    model = ResNetBackbone()
    dummy_input = torch.randn(2, 3, 224, 224)
    spatial, global_feat, skips = model(dummy_input)
    print("Spatial features shape:", spatial.shape)    # [2, 2048, 7, 7]
    print("Global features shape:", global_feat.shape) # [2, 2048]
    for i, s in enumerate(skips):
        print(f"Skip {i+1} shape:", s.shape)
