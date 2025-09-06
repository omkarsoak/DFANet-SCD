######### Model definition #########
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d
import torchvision.models as models

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for channel attention"""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SpatialAttention(nn.Module):
    """Spatial attention module"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channel, reduction=16, spatial_kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = SEBlock(channel, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

##########
# input x
# Mc(F) = self.channel_attention(x)
# Ms(F) = self.spatial_attention(Mc(F))
# z = Mc(F) * Ms(F) 
# return z

class DenseFeatureFusion(nn.Module):
    """Dense feature fusion module with residual connections"""
    def __init__(self, in_channels, growth_rate=32, num_layers=4):
        super(DenseFeatureFusion, self).__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            layer = nn.Sequential(
                nn.BatchNorm2d(in_channels + i * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
            )
            self.layers.append(layer)

        self.out_channels = in_channels + num_layers * growth_rate
        self.conv1x1 = nn.Conv2d(self.out_channels, in_channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        features = [x]

        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)

        out = self.conv1x1(torch.cat(features, 1))
        out = self.norm(out)
        out = self.relu(out + x)  # Residual connection

        return out
    
class BasicBlock_ss_DFANet(nn.Module):
    """Enhanced basic block with squeeze-excitation and residual connection"""
    def __init__(self, inplanes, planes=None, subsamp=1, use_attention=True):
        super(BasicBlock_ss_DFANet, self).__init__()
        if planes is None:
            planes = inplanes * subsamp

        # First conv block
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(inplace=True)

        # Second conv block
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        # Downsampling
        self.subsamp = subsamp
        self.doit = planes != inplanes

        if self.doit:
            self.couple = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.bnc = nn.BatchNorm2d(planes)

        # Attention mechanism
        self.use_attention = use_attention
        if use_attention:
            self.cbam = CBAM(planes)

        # Dropout for regularization
        self.dropout = nn.Dropout2d(p=0.1)
        # self.dropout = nn.Dropout2d(p=0.3)

    def forward(self, x):
        if self.doit:
            residual = self.couple(x)
            residual = self.bnc(residual)
        else:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        if self.subsamp > 1:
            out = F.max_pool2d(out, kernel_size=self.subsamp, stride=self.subsamp)
            residual = F.max_pool2d(residual, kernel_size=self.subsamp, stride=self.subsamp)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_attention:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out

class BasicBlock_us_DFANet(nn.Module):
    """Enhanced upsampling block"""
    def __init__(self, inplanes, upsamp=1, use_attention=True):
        super(BasicBlock_us_DFANet, self).__init__()
        planes = int(inplanes / upsamp)

        # Transposed convolution for upsampling
        self.conv1 = nn.ConvTranspose2d(inplanes, planes, kernel_size=3, padding=1,
                                        stride=upsamp, output_padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(inplace=True)

        # Regular convolution
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        # Skip connection
        self.upsamp = upsamp
        self.couple = nn.ConvTranspose2d(inplanes, planes, kernel_size=3, padding=1,
                                         stride=upsamp, output_padding=1, bias=False)
        self.bnc = nn.BatchNorm2d(planes)

        # Attention
        self.use_attention = use_attention
        if use_attention:
            self.cbam = CBAM(planes)

        # Dropout
        self.dropout = nn.Dropout2d(p=0.1)
        # self.dropout = nn.Dropout2d(p=0.3)

    def forward(self, x):
        residual = self.couple(x)
        residual = self.bnc(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_attention:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out

class EncoderDFANet(nn.Module):
    """Enhanced encoder with deep supervision + attention + dense fusion"""
    def __init__(self, input_nbr, base_depth=16, use_attention=True):
        super(EncoderDFANet, self).__init__()
        self.input_nbr = input_nbr
        self.use_attention = use_attention

        cur_depth = input_nbr
        self.base_depth = base_depth

        # Encoding stage 1
        self.encres1_1 = BasicBlock_ss_DFANet(cur_depth, planes=base_depth, use_attention=use_attention)
        cur_depth = base_depth
        d1 = base_depth
        self.encres1_2 = BasicBlock_ss_DFANet(cur_depth, use_attention=use_attention)
        self.encres1_3 = BasicBlock_ss_DFANet(cur_depth, subsamp=2, use_attention=use_attention)
        cur_depth *= 2

        # Encoding stage 2
        self.encres2_1 = BasicBlock_ss_DFANet(cur_depth, use_attention=use_attention)
        d2 = cur_depth
        self.encres2_2 = BasicBlock_ss_DFANet(cur_depth, use_attention=use_attention)
        self.encres2_3 = BasicBlock_ss_DFANet(cur_depth, subsamp=2, use_attention=use_attention)
        cur_depth *= 2

        # Encoding stage 3
        self.encres3_1 = BasicBlock_ss_DFANet(cur_depth, use_attention=use_attention)
        d3 = cur_depth
        self.encres3_2 = BasicBlock_ss_DFANet(cur_depth, use_attention=use_attention)
        self.encres3_3 = BasicBlock_ss_DFANet(cur_depth, subsamp=2, use_attention=use_attention)
        cur_depth *= 2

        # Encoding stage 4
        self.encres4_1 = BasicBlock_ss_DFANet(cur_depth, use_attention=use_attention)
        d4 = cur_depth
        self.encres4_2 = BasicBlock_ss_DFANet(cur_depth, use_attention=use_attention)
        self.encres4_3 = BasicBlock_ss_DFANet(cur_depth, subsamp=2, use_attention=use_attention)
        cur_depth *= 2

        # Encoding stage 5
        self.encres5_1 = BasicBlock_ss_DFANet(cur_depth, use_attention=use_attention)
        d5 = cur_depth
        self.encres5_2 = BasicBlock_ss_DFANet(cur_depth, use_attention=use_attention)
        self.encres5_3 = BasicBlock_ss_DFANet(cur_depth, subsamp=2, use_attention=use_attention)
        cur_depth *= 2
        
        # use dense feature fusion
        self.dense_fusion = DenseFeatureFusion(cur_depth, growth_rate=32, num_layers=4)
        
        # Store final depths for the decoder
        self.depths = [input_nbr, d1, d2, d3, d4, d5, cur_depth]

        # Optional parameter initialization
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        s1_1 = x.size()
        x = self.encres1_1(x)
        x1 = self.encres1_2(x)
        x = self.encres1_3(x1)

        s2_1 = x.size()
        x = self.encres2_1(x)
        x2 = self.encres2_2(x)
        x = self.encres2_3(x2)

        s3_1 = x.size()
        x = self.encres3_1(x)
        x3 = self.encres3_2(x)
        x = self.encres3_3(x3)

        s4_1 = x.size()
        x = self.encres4_1(x)
        x4 = self.encres4_2(x)
        x = self.encres4_3(x4)

        s5_1 = x.size()
        x = self.encres5_1(x)
        x5 = self.encres5_2(x)
        x = self.encres5_3(x5)

        # # Now apply dense fusion at the bottleneck
        x = self.dense_fusion(x)

        sizes = [s1_1, s2_1, s3_1, s4_1, s5_1]
        outputs = [x1, x2, x3, x4, x5, x]

        return outputs, sizes

class DecoderDFANet(nn.Module):
    """Enhanced decoder with multi-scale fusion and attention"""
    def __init__(self, label_nbr, depths, CD=False, use_attention=True):
        super(DecoderDFANet, self).__init__()
        self.use_attention = use_attention

        cur_depth = depths[6]

        # Decoding stage 5
        self.decres5_1 = BasicBlock_ss_DFANet(cur_depth + CD * depths[6], planes=cur_depth, use_attention=use_attention)
        self.decres5_2 = BasicBlock_ss_DFANet(cur_depth, use_attention=use_attention)
        self.decres5_3 = BasicBlock_us_DFANet(cur_depth, upsamp=2, use_attention=use_attention)
        cur_depth = depths[5]

        # Decoding stage 4
        self.decres4_1 = BasicBlock_ss_DFANet(cur_depth + depths[5] + CD * depths[5], planes=cur_depth, use_attention=use_attention)
        self.decres4_2 = BasicBlock_ss_DFANet(cur_depth, use_attention=use_attention)
        self.decres4_3 = BasicBlock_us_DFANet(cur_depth, upsamp=2, use_attention=use_attention)
        cur_depth = depths[4]

        # Decoding stage 3
        self.decres3_1 = BasicBlock_ss_DFANet(cur_depth + depths[4] + CD * depths[4], planes=cur_depth, use_attention=use_attention)
        self.decres3_2 = BasicBlock_ss_DFANet(cur_depth, use_attention=use_attention)
        self.decres3_3 = BasicBlock_us_DFANet(cur_depth, upsamp=2, use_attention=use_attention)
        cur_depth = depths[3]

        # Decoding stage 2
        self.decres2_1 = BasicBlock_ss_DFANet(cur_depth + depths[3] + CD * depths[3], planes=cur_depth, use_attention=use_attention)
        self.decres2_2 = BasicBlock_ss_DFANet(cur_depth, use_attention=use_attention)
        self.decres2_3 = BasicBlock_us_DFANet(cur_depth, upsamp=2, use_attention=use_attention)
        cur_depth = depths[2]

        # Decoding stage 1
        self.decres1_1 = BasicBlock_ss_DFANet(cur_depth + depths[2] + CD * depths[2], planes=cur_depth, use_attention=use_attention)
        self.decres1_2 = BasicBlock_ss_DFANet(cur_depth, use_attention=use_attention)
        self.decres1_3 = BasicBlock_us_DFANet(cur_depth, upsamp=2, use_attention=use_attention)
        cur_depth = depths[1]

        # Decoding stage 0
        self.decres0_1 = BasicBlock_ss_DFANet(cur_depth + depths[1] + CD * depths[1], planes=cur_depth, use_attention=use_attention)
        self.decres0_2 = BasicBlock_ss_DFANet(cur_depth, use_attention=use_attention)
        self.decres0_3 = BasicBlock_ss_DFANet(cur_depth, use_attention=use_attention)

        # Deep supervision paths
        self.aux_head4 = nn.Sequential(
            nn.Conv2d(depths[5], depths[5] // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(depths[5] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(depths[5] // 2, label_nbr, kernel_size=1)
        )

        self.aux_head3 = nn.Sequential(
            nn.Conv2d(depths[4], depths[4] // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(depths[4] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(depths[4] // 2, label_nbr, kernel_size=1)
        )

        self.aux_head2 = nn.Sequential(
            nn.Conv2d(depths[3], depths[3] // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(depths[3] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(depths[3] // 2, label_nbr, kernel_size=1)
        )

        # Final output layer
        self.coupling = nn.Conv2d(cur_depth, label_nbr, kernel_size=1)

        # Optional: Initialize parameters
        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, outputs, sizes, return_aux=False):
        # Bottleneck layer
        x = self.decres5_1(outputs[5])
        x = self.decres5_2(x)
        x = self.decres5_3(x)
        s5_1 = sizes[4]
        s5_2 = x.size()
        pad5 = ReplicationPad2d((0, s5_1[3] - s5_2[3], 0, s5_1[2] - s5_2[2]))
        x = pad5(x)

        # Stage 4 decoding
        x = self.decres4_1(torch.cat((x, outputs[4]), 1))
        x = self.decres4_2(x)
        aux4 = self.aux_head4(x)  # Deep supervision
        x = self.decres4_3(x)
        s4_1 = sizes[3]
        s4_2 = x.size()
        pad4 = ReplicationPad2d((0, s4_1[3] - s4_2[3], 0, s4_1[2] - s4_2[2]))
        x = pad4(x)

        # Stage 3 decoding
        x = self.decres3_1(torch.cat((x, outputs[3]), 1))
        x = self.decres3_2(x)
        aux3 = self.aux_head3(x)  # Deep supervision
        x = self.decres3_3(x)
        s3_1 = sizes[2]
        s3_2 = x.size()
        pad3 = ReplicationPad2d((0, s3_1[3] - s3_2[3], 0, s3_1[2] - s3_2[2]))
        x = pad3(x)

        # Stage 2 decoding
        x = self.decres2_1(torch.cat((x, outputs[2]), 1))
        x = self.decres2_2(x)
        aux2 = self.aux_head2(x)  # Deep supervision
        x = self.decres2_3(x)
        s2_1 = sizes[1]
        s2_2 = x.size()
        pad2 = ReplicationPad2d((0, s2_1[3] - s2_2[3], 0, s2_1[2] - s2_2[2]))
        x = pad2(x)

        # Stage 1 decoding
        x = self.decres1_1(torch.cat((x, outputs[1]), 1))
        x = self.decres1_2(x)
        x = self.decres1_3(x)
        s1_1 = sizes[0]
        s1_2 = x.size()
        pad1 = ReplicationPad2d((0, s1_1[3] - s1_2[3], 0, s1_1[2] - s1_2[2]))
        x = pad1(x)

        # Final stage
        x = self.decres0_1(torch.cat((x, outputs[0]), 1))
        x = self.decres0_2(x)
        x = self.decres0_3(x)

        # Final output
        x = self.coupling(x)

        if return_aux:
            # Resize auxiliary outputs to match the input size
            aux4 = F.interpolate(aux4, size=s1_1[2:], mode='bilinear', align_corners=False)
            aux3 = F.interpolate(aux3, size=s1_1[2:], mode='bilinear', align_corners=False)
            aux2 = F.interpolate(aux2, size=s1_1[2:], mode='bilinear', align_corners=False)
            return x, aux4, aux3, aux2

        return x

class DifferenceFusion(nn.Module):
    """Enhanced fusion module for feature differences"""
    def __init__(self, in_channels, fusion_type='weighted'):
        super(DifferenceFusion, self).__init__()
        self.fusion_type = fusion_type

        if fusion_type == 'weighted':
            self.weight_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
            self.norm = nn.BatchNorm2d(in_channels)
            self.relu = nn.ReLU(inplace=True)
        elif fusion_type == 'attention':
            self.channel_attention = SEBlock(in_channels)
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
            self.norm = nn.BatchNorm2d(in_channels)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, feat1, feat2):
        diff = torch.abs(feat1 - feat2)

        if self.fusion_type == 'weighted':
            # Weighted fusion that considers both inputs
            combined = torch.cat([feat1, feat2], dim=1)
            weights = self.weight_conv(combined)
            weights = self.norm(weights)
            weights = self.relu(weights)
            return diff * weights

        elif self.fusion_type == 'attention':
            # Apply channel attention to the difference features
            diff = self.channel_attention(diff)
            diff = self.conv(diff)
            diff = self.norm(diff)
            return self.relu(diff)

        else:  # Default: direct
            return diff

class DFANet(nn.Module):
    """
    Enhanced multitask learning model for semantic change detection:
    - Improved backbone with attention mechanisms
    - Enhanced feature fusion with difference features
    - Deep supervision for better gradient flow
    - Regularization techniques to prevent overfitting
    """
    def __init__(self, input_nbr=3, sem_classes=4, cd_classes=13, with_softmax=True, base_depth=16):
        super(DFANet, self).__init__()

        self.input_nbr = input_nbr
        self.sem_classes = sem_classes
        self.cd_classes = cd_classes
        self.with_softmax = with_softmax

        # Terrain Classification (Semantic Segmentation)
        self.SSEncoder = EncoderDFANet(self.input_nbr, base_depth=base_depth)
        self.SSDecoder = DecoderDFANet(self.sem_classes, self.SSEncoder.depths)

        # Change Detection with improved feature differencing
        self.CDEncoder = EncoderDFANet(2 * self.input_nbr, base_depth=base_depth)
        self.CDDecoder = DecoderDFANet(self.cd_classes, self.CDEncoder.depths, CD=True)

        # Enhanced difference feature fusion
        self.diff_fusion = nn.ModuleList()
        for i, depth in enumerate(self.SSEncoder.depths[1:]):
            self.diff_fusion.append(DifferenceFusion(depth, fusion_type='attention'))

        # Optional: Multi-scale prediction refinement
        self.refine_conv = nn.Conv2d(self.cd_classes + self.sem_classes * 2, self.cd_classes, kernel_size=3, padding=1)
        self.refine_norm = nn.BatchNorm2d(self.cd_classes)
        self.refine_relu = nn.ReLU(inplace=True)

        # Optional softmax for output probabilities
        if with_softmax:
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, training=False):
        # Terrain Classification (Semantic Segmentation) - Image 1
        outputs_1, sizes_1 = self.SSEncoder(x1)
        sem_out1 = self.SSDecoder(outputs_1, sizes_1)

        # Terrain Classification (Semantic Segmentation) - Image 2
        outputs_2, sizes_2 = self.SSEncoder(x2)
        sem_out2 = self.SSDecoder(outputs_2, sizes_2)

        # Change Detection with feature differencing
        outputs_cd, sizes_cd = self.CDEncoder(torch.cat((x1, x2), 1))

        # Enhanced feature differencing with attention-based fusion
        for i in range(len(outputs_cd)):
            if i < len(self.diff_fusion):
                diff_features = self.diff_fusion[i](outputs_1[i], outputs_2[i])
                outputs_cd[i] = torch.cat((outputs_cd[i], diff_features), 1)
            else:
                # For any extra features not covered by diff_fusion
                outputs_cd[i] = torch.cat((outputs_cd[i], torch.abs(outputs_1[i] - outputs_2[i])), 1)

        # Get change detection output
        if training:
            change_out, aux4, aux3, aux2 = self.CDDecoder(outputs_cd, sizes_cd, return_aux=True)
        else:
            change_out = self.CDDecoder(outputs_cd, sizes_cd)

        # Optional refinement using semantic outputs
        if sizes_1[0][2:] != change_out.shape[2:]:
            # Ensure all outputs have the same spatial dimensions
            sem_out1 = F.interpolate(sem_out1, size=change_out.shape[2:], mode='bilinear', align_corners=False)
            sem_out2 = F.interpolate(sem_out2, size=change_out.shape[2:], mode='bilinear', align_corners=False)

        # Refine change detection results using semantic segmentation results
        refined = torch.cat([change_out, sem_out1, sem_out2], dim=1)
        refined = self.refine_conv(refined)
        refined = self.refine_norm(refined)
        refined = self.refine_relu(refined)

        # Combine with original change detection
        change_out = change_out + refined  # Residual connection

        # Apply softmax if requested
        if self.with_softmax:
            sem_out1 = self.softmax(sem_out1)
            sem_out2 = self.softmax(sem_out2)
            change_out = self.softmax(change_out)

        if training:
            aux_outputs = {
                'aux4': self.softmax(aux4) if self.with_softmax else aux4,
                'aux3': self.softmax(aux3) if self.with_softmax else aux3,
                'aux2': self.softmax(aux2) if self.with_softmax else aux2
            }
            return sem_out1, sem_out2, change_out, aux_outputs

        return sem_out1, sem_out2, change_out
