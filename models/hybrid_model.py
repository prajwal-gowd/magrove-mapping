import torch
import torch.nn as nn
from models.resnet_backbone import ResNetBackbone
from models.transformer_encoder import TransformerModule
import config

class HybridTCCFNet(nn.Module):
    def __init__(self, num_classes=1, backbone_freeze=False):
        super(HybridTCCFNet, self).__init__()
        
        self.cnn = ResNetBackbone()
        
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
            feat_h=7, # Expected ResNet50 spatial output size for 224x224 input
            feat_w=7
        )
        
        # Segmentation Decoder: from [B, 4096, 7, 7] to [B, num_classes, 224, 224]
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4096, 512, kernel_size=4, stride=2, padding=1), # -> 14x14
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # -> 28x28
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # -> 56x56
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # -> 112x112
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # -> 224x224
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, kernel_size=1) # -> 224x224
        )

    def forward(self, x):
        """
        Forward pass for the hybrid segmentation model.
        Args:
            x: Input images [B, 3, 224, 224]
        Returns:
            Spatial probability logits [B, 1, 224, 224]
        """
        B = x.size(0)
        
        # CNN Feature Extraction
        # spatial_features: [B, 2048, 7, 7]
        spatial_features, _ = self.cnn(x)
        
        # Transformer Processing
        # global_transformer_features: [B, 2048]
        global_transformer_features = self.transformer(spatial_features)
        
        # Feature Fusion
        # Broadcast transformer features to spatial dims: [B, 2048, 7, 7]
        global_transformer_broadcast = global_transformer_features.unsqueeze(2).unsqueeze(3).expand(-1, -1, 7, 7)
        
        fused_features = torch.cat((spatial_features, global_transformer_broadcast), dim=1) # [B, 4096, 7, 7]
        
        # Final Segmentation Decoding
        logits = self.decoder(fused_features) # [B, 1, 224, 224]
        
        return logits

def run_dummy_forward():
    print("Running dummy forward pass to test execution...")
    model = HybridTCCFNet(num_classes=config.NUM_CLASSES)
    model.eval()
    
    # Create a random input tensor simulating a batch of 2 RGB images of size 224x224
    dummy_input = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        output = model(dummy_input)
        
    print(f"Output shape: {output.shape} (Expected: [2, {config.NUM_CLASSES}, 224, 224])")
    print("Dummy forward pass completed successfully.")
    
if __name__ == "__main__":
    run_dummy_forward()
