import timm
import torch
import torch.nn as nn

class HybridMultiModalModel(nn.Module):
    """
    Fusion model combining EfficientNet-B0 vision backbone with clinical metadata MLP.
    Designed for edge deployment with minimal VRAM footprint.
    Uses timm library for stable pre-trained weights.
    """
    def __init__(self, num_metadata_features):
        super().__init__()
        
        # Vision backbone: EfficientNet-B0 (pre-trained via timm for better stability)
        self.vision_backbone = timm.create_model('efficientnet_b0', pretrained=True)
        
        # Extract features (1280 for B0) and remove the original classifier
        num_vision_features = self.vision_backbone.classifier.in_features
        self.vision_backbone.reset_classifier(0)
        
        # Tabular MLP for clinical metadata
        self.metadata_mlp = nn.Sequential(
            nn.Linear(num_metadata_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Fusion head: 1280 (vision) + 32 (metadata) = 1312
        self.decision_head = nn.Sequential(
            nn.Linear(num_vision_features + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 1)
        )
    
    def forward(self, images, metadata):
        # Extract visual features
        vision_features = self.vision_backbone(images)  # (batch, 1280)
        
        # Extract metadata features
        metadata_features = self.metadata_mlp(metadata)  # (batch, 32)
        
        # Concatenate
        fused = torch.cat([vision_features, metadata_features], dim=1)  # (batch, 1312)
        
        # Final prediction (logits)
        output = self.decision_head(fused)  # (batch, 1)
        
        return output.squeeze(1)
