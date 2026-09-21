import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNetBackbone(nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        # Load a pretrained ResNet50
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # We want to extract spatial feature maps, so we remove the Global Average Pooling and FC layer.
        # resnet essentially consists of:
        # conv1, bn1, relu, maxpool
        # layer1, layer2, layer3, layer4
        # avgpool, fc
        
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        
        # We also keep the global average pooling to extract global CNN features
        self.avgpool = resnet.avgpool

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (Tensor): Input images of shape [B, 3, H, W]
        Returns:
            spatial_features (Tensor): Feature maps of shape [B, 2048, H', W']
            global_features (Tensor): Flattened global features of shape [B, 2048]
        """
        spatial_features = self.features(x)
        
        global_features = self.avgpool(spatial_features)
        global_features = torch.flatten(global_features, 1) # [B, 2048]
        
        return spatial_features, global_features

if __name__ == "__main__":
    # Quick test
    model = ResNetBackbone()
    dummy_input = torch.randn(2, 3, 224, 224)
    spatial, global_feat = model(dummy_input)
    print("Spatial features shape:", spatial.shape) # Expected: [2, 2048, 7, 7]
    print("Global features shape:", global_feat.shape) # Expected: [2, 2048]
