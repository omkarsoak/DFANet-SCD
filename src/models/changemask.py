########### ChangeMask ############
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder

class Squeeze(nn.Module):
    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        return x.squeeze(dim=self.dim)

class SpatioTemporalInteraction(nn.Sequential):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation=1,
                 type='conv3d'):
        if type == 'conv3d':
            padding = dilation * (kernel_size - 1) // 2
            super(SpatioTemporalInteraction, self).__init__(
                nn.Conv3d(in_channels, out_channels, [2, kernel_size, kernel_size], stride=1,
                          dilation=(1, dilation, dilation),
                          padding=(0, padding, padding),
                          bias=False),
                Squeeze(dim=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            )
        elif type == 'conv1plus2d':
            super(SpatioTemporalInteraction, self).__init__(
                nn.Conv3d(in_channels, out_channels, (2, 1, 1), stride=1,
                          padding=(0, 0, 0),
                          bias=False),
                Squeeze(dim=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, kernel_size, 1,
                          kernel_size // 2) if kernel_size > 1 else nn.Identity(),
                nn.BatchNorm2d(out_channels) if kernel_size > 1 else nn.Identity(),
                nn.ReLU(True) if kernel_size > 1 else nn.Identity(),
            )

class TemporalSymmetricTransformer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation=1,
                 interaction_type='conv3d',
                 symmetric_fusion='add'):
        super(TemporalSymmetricTransformer, self).__init__()

        if isinstance(in_channels, list) or isinstance(in_channels, tuple):
            self.t = nn.ModuleList([
                SpatioTemporalInteraction(inc, outc, kernel_size, dilation=dilation, type=interaction_type)
                for inc, outc in zip(in_channels, out_channels)
            ])
        else:
            self.t = SpatioTemporalInteraction(in_channels, out_channels, kernel_size, dilation=dilation,
                                               type=interaction_type)

        if symmetric_fusion == 'add':
            self.symmetric_fusion = lambda x, y: x + y
        elif symmetric_fusion == 'mul':
            self.symmetric_fusion = lambda x, y: x * y
        elif symmetric_fusion == None:
            self.symmetric_fusion = None

    def forward(self, features1, features2):
        if isinstance(features1, list):
            d12_features = [op(torch.stack([f1, f2], dim=2)) for op, f1, f2 in
                            zip(self.t, features1, features2)]
            if self.symmetric_fusion:
                d21_features = [op(torch.stack([f2, f1], dim=2)) for op, f1, f2 in
                                zip(self.t, features1, features2)]
                change_features = [self.symmetric_fusion(d12, d21) for d12, d21 in zip(d12_features, d21_features)]
            else:
                change_features = d12_features
        else:
            if self.symmetric_fusion:
                change_features = self.symmetric_fusion(self.t(torch.stack([features1, features2], dim=2)),
                                                        self.t(torch.stack([features2, features1], dim=2)))
            else:
                change_features = self.t(torch.stack([features1, features2], dim=2))
            change_features = change_features.squeeze(dim=2)
        return change_features

class ChangeMask(nn.Module):
    """
    ChangeMask model adapted to align with other models in the codebase

    Args:
        input_nbr (int): Number of input channels per image
        sem_classes (int): Number of semantic segmentation classes
        cd_classes (int): Number of change detection classes (usually binary for ChangeMask)
        with_softmax (bool): Whether to apply softmax to outputs
    """
    def __init__(self, input_nbr=3, sem_classes=4, cd_classes=2, with_softmax=True):
        super(ChangeMask, self).__init__()

        self.input_nbr = input_nbr
        self.sem_classes = sem_classes
        self.cd_classes = cd_classes
        self.with_softmax = with_softmax

        # Use EfficientNet-B0 as encoder
        self.encoder = smp.encoders.get_encoder('efficientnet-b0', weights='imagenet')
        out_channels = self.encoder.out_channels

        # Semantic decoder for both timesteps (shared weights)
        self.semantic_decoder = UnetDecoder(
            encoder_channels=out_channels,
            decoder_channels=[256, 128, 64, 32, 16],
        )

        # Change detection decoder
        self.change_decoder = UnetDecoder(
            encoder_channels=out_channels,
            decoder_channels=[256, 128, 64, 32, 16],
        )

        # Temporal transformer for change detection
        self.temporal_transformer = TemporalSymmetricTransformer(
            out_channels, out_channels,
            3, interaction_type='conv3d', symmetric_fusion='add',
        )

        # Output layers
        self.sem_head = nn.Conv2d(16, sem_classes, 1)

        # For change detection, output cd_classes or 1 (binary) based on how you're handling change
        if cd_classes == 2:
            self.change_head = nn.Conv2d(16, 1, 1)  # Binary change detection
        else:
            self.change_head = nn.Conv2d(16, cd_classes, 1)  # Multi-class change detection

        # Optional softmax for output probabilities
        if with_softmax:
            self.softmax = nn.Softmax(dim=1)
            self.sigmoid = nn.Sigmoid()

    def encode_single_image(self, x):
        """Extract features from a single image"""
        features = self.encoder(x)
        return features

    def forward(self, x1, x2):
        """Forward method that matches your other models' interface"""
        # Extract features for both images
        t1_features = self.encode_single_image(x1)
        t2_features = self.encode_single_image(x2)

        # Process semantic features for both timesteps
        sem_out1 = self.sem_head(self.semantic_decoder(*t1_features))
        sem_out2 = self.sem_head(self.semantic_decoder(*t2_features))

        # Process change features
        temporal_features = self.temporal_transformer(t1_features, t2_features)
        change_out = self.change_head(self.change_decoder(*temporal_features))

        # Apply softmax/sigmoid if requested
        if self.with_softmax:
            sem_out1 = self.softmax(sem_out1)
            sem_out2 = self.softmax(sem_out2)

            # For change detection, use sigmoid for binary or softmax for multi-class
            if change_out.shape[1] == 1:
                change_out = self.sigmoid(change_out)
            else:
                change_out = self.softmax(change_out)

        return sem_out1, sem_out2, change_out