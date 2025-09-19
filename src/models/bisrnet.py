################# BiSRNet #################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class FCN(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):
        super(FCN, self).__init__()
        resnet = models.resnet34(pretrained=pretrained)
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        newconv1.weight.data[:, 0:3, :, :].copy_(resnet.conv1.weight.data[:, 0:3, :, :])
        if in_channels > 3:
            newconv1.weight.data[:, 3:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels-3, :, :])

        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Adjust stride for deeper layers to maintain resolution
        for n, m in self.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)

        # Add head layer for feature extraction
        self.head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Save input size for later upsampling
        input_shape = x.shape[-2:]

        # Encoder pathway with features
        x0 = self.layer0(x)  # 1/2
        x0p = self.maxpool(x0)  # 1/4
        x1 = self.layer1(x0p)  # 1/4
        x2 = self.layer2(x1)  # 1/8
        x3 = self.layer3(x2)  # 1/16 (but with adjusted stride, remains 1/8)
        x4 = self.layer4(x3)  # 1/32 (but with adjusted stride, remains 1/8)

        # Head
        features = self.head(x4)

        return features, input_shape

class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class SR(nn.Module):
    '''Spatial reasoning module'''
    def __init__(self, in_dim):
        super(SR, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = x + self.gamma*out

        return out

class CotSR(nn.Module):
    def __init__(self, in_dim):
        super(CotSR, self).__init__()
        self.chanel_in = in_dim

        self.query_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.query_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        m_batchsize, C, height, width = x1.size()

        q1 = self.query_conv1(x1).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        k1 = self.key_conv1(x1).view(m_batchsize, -1, width*height)
        v1 = self.value_conv1(x1).view(m_batchsize, -1, width*height)

        q2 = self.query_conv2(x2).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        k2 = self.key_conv2(x2).view(m_batchsize, -1, width*height)
        v2 = self.value_conv2(x2).view(m_batchsize, -1, width*height)

        energy1 = torch.bmm(q1, k2)
        attention1 = self.softmax(energy1)
        out1 = torch.bmm(v2, attention1.permute(0, 2, 1))
        out1 = out1.view(m_batchsize, C, height, width)

        energy2 = torch.bmm(q2, k1)
        attention2 = self.softmax(energy2)
        out2 = torch.bmm(v1, attention2.permute(0, 2, 1))
        out2 = out2.view(m_batchsize, C, height, width)

        out1 = x1 + self.gamma1*out1
        out2 = x2 + self.gamma2*out2

        return out1, out2

class BiSRNet(nn.Module):
    def __init__(self, input_nbr=3, sem_classes=7, cd_classes=31):
        super(BiSRNet, self).__init__()
        self.input_nbr = input_nbr
        self.sem_classes = sem_classes
        self.cd_classes = cd_classes

        # Feature extraction network using Resnet34
        self.FCN = FCN(input_nbr, pretrained=True)

        # Spatial reasoning modules
        self.SiamSR = SR(128)
        self.CotSR = CotSR(128)

        # Change detection branch
        self.resCD = self._make_layer(ResBlock, 256, 128, 6, stride=1)

        # Output classifiers
        self.classifier1 = nn.Conv2d(128, sem_classes, kernel_size=1)
        self.classifier2 = nn.Conv2d(128, sem_classes, kernel_size=1)
        self.classifierCD = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, cd_classes, kernel_size=1)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes) )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def base_forward(self, x):
        features, input_shape = self.FCN(x)
        features = self.SiamSR(features)
        return features, input_shape

    def forward(self, x1, x2):
        # Extract features using the shared encoder
        x1_feat, input_shape = self.base_forward(x1)
        x2_feat, _ = self.base_forward(x2)

        # Change detection - concatenate features
        x_cd = torch.cat([x1_feat, x2_feat], 1)
        x_cd = self.resCD(x_cd)

        # Apply CotSR for feature refinement
        x1_refine, x2_refine = self.CotSR(x1_feat, x2_feat)

        # Generate outputs
        lcm1_logits = self.classifier1(x1_refine)
        lcm2_logits = self.classifier2(x2_refine)
        change_logits = self.classifierCD(x_cd)

        # Upsample all outputs to match input size
        change_logits = F.interpolate(change_logits, size=input_shape, mode='bilinear', align_corners=False)
        lcm1_logits = F.interpolate(lcm1_logits, size=input_shape, mode='bilinear', align_corners=False)
        lcm2_logits = F.interpolate(lcm2_logits, size=input_shape, mode='bilinear', align_corners=False)

        return lcm1_logits, lcm2_logits, change_logits