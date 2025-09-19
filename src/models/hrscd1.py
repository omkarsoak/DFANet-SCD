######## HRSCD 1: PCC with UNet ############
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class HRSCD1(nn.Module):
    def __init__(self, architecture='unet', encoder='resnet34', input_channels=3, num_semantic_classes=4, num_cd_classes=3):
        super().__init__()
        # Semantic segmentation models for each timestamp
        if architecture.lower() == 'unet':
            self.sem_model = smp.Unet(
                encoder_name=encoder,
                encoder_weights="imagenet",
                in_channels=input_channels,
                classes=num_semantic_classes,
            )

        self.change_head = nn.Sequential(
            nn.Conv2d(num_semantic_classes*2, 64, kernel_size=3, padding=1),  # Update input channels
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_cd_classes, kernel_size=1),  # Update output channels
            nn.Dropout(0.3)
        )

    def forward(self, x1, x2):
        # Get semantic features for both timestamps
        sem_feat1 = self.sem_model(x1)
        sem_feat2 = self.sem_model(x2)

        # Concatenate semantic features
        combined_feat = torch.cat([sem_feat1, sem_feat2], dim=1)

        # Get change detection output
        change_out = self.change_head(combined_feat)

        return sem_feat1, sem_feat2, change_out
