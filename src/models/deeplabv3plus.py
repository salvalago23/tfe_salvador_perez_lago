"""
DeepLabV3+ para segmentación de imágenes médicas

Referencia: Chen et al., "Encoder-Decoder with Atrous Separable Convolution 
for Semantic Image Segmentation"
https://arxiv.org/abs/1802.02611

Características principales:
- Atrous Spatial Pyramid Pooling (ASPP) para capturar contexto multi-escala
- Decoder con características de bajo nivel para bordes precisos
- Backbone configurable (ResNet simplificado por defecto)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """Bloque Conv + BatchNorm + ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=1, dilation=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SeparableConv2d(nn.Module):
    """Convolución separable en profundidad (Depthwise Separable Convolution)"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=1, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                                   padding, dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class ASPPConv(nn.Module):
    """Convolución atrous para ASPP"""
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, 
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class ASPPPooling(nn.Module):
    """Global Average Pooling para ASPP"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        size = x.shape[2:]
        x = self.pool(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling
    
    Captura información contextual a múltiples escalas usando convoluciones
    atrous con diferentes tasas de dilatación.
    """
    def __init__(self, in_channels, out_channels=256, atrous_rates=[6, 12, 18]):
        super().__init__()
        
        modules = []
        
        # 1x1 convolution
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # Atrous convolutions
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        
        # Global average pooling
        modules.append(ASPPPooling(in_channels, out_channels))
        
        self.convs = nn.ModuleList(modules)
        
        # Project concatenated features
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class ResidualBlock(nn.Module):
    """Bloque residual simplificado para el backbone"""
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class SimpleBackbone(nn.Module):
    """
    Backbone simplificado tipo ResNet para DeepLabV3+
    
    Genera features de bajo nivel (stride 4) y alto nivel (stride 16)
    """
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, features[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # Layer 1 (stride 4) - Low-level features
        self.layer1 = self._make_layer(features[0], features[0], blocks=2, stride=1)
        
        # Layer 2 (stride 8)
        self.layer2 = self._make_layer(features[0], features[1], blocks=2, stride=2)
        
        # Layer 3 (stride 16)
        self.layer3 = self._make_layer(features[1], features[2], blocks=2, stride=2)
        
        # Layer 4 (stride 16, con dilatación) - High-level features
        self.layer4 = self._make_layer(features[2], features[3], blocks=2, stride=1, dilation=2)
        
        self.low_level_channels = features[0]
        self.high_level_channels = features[3]
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1, dilation=1):
        layers = [ResidualBlock(in_channels, out_channels, stride, dilation)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1, dilation))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        low_level = self.layer1(x)  # stride 4
        x = self.layer2(low_level)
        x = self.layer3(x)
        high_level = self.layer4(x)  # stride 16
        return low_level, high_level


class DeepLabV3PlusDecoder(nn.Module):
    """
    Decoder de DeepLabV3+
    
    Combina features de alto nivel (del ASPP) con features de bajo nivel
    del backbone para obtener segmentaciones con bordes precisos.
    """
    def __init__(self, low_level_channels, aspp_channels=256, out_channels=1):
        super().__init__()
        
        # Reduce low-level feature channels
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Final convolutions
        self.final_conv = nn.Sequential(
            SeparableConv2d(aspp_channels + 48, 256, 3, padding=1),
            SeparableConv2d(256, 256, 3, padding=1),
            nn.Conv2d(256, out_channels, 1)
        )
    
    def forward(self, low_level_features, aspp_features):
        # Reduce low-level features
        low_level = self.low_level_conv(low_level_features)
        
        # Upsample ASPP features to match low-level size
        aspp_up = F.interpolate(aspp_features, size=low_level.shape[2:],
                                mode='bilinear', align_corners=False)
        
        # Concatenate
        concat = torch.cat([low_level, aspp_up], dim=1)
        
        # Final prediction
        return self.final_conv(concat)


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ para segmentación semántica
    
    Combina:
    - Backbone para extracción de características multi-escala
    - ASPP para captura de contexto global
    - Decoder para recuperar detalles de bordes
    
    Args:
        in_channels: Canales de entrada (default: 3 para RGB)
        out_channels: Canales de salida (default: 1 para segmentación binaria)
        features: Lista de canales para el backbone (default: [64, 128, 256, 512])
        aspp_channels: Canales de salida del ASPP (default: 256)
        atrous_rates: Tasas de dilatación para ASPP (default: [6, 12, 18])
    """
    
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
                 aspp_channels=256, atrous_rates=[6, 12, 18]):
        super().__init__()
        
        # Backbone
        self.backbone = SimpleBackbone(in_channels, features)
        
        # ASPP
        self.aspp = ASPP(self.backbone.high_level_channels, aspp_channels, atrous_rates)
        
        # Decoder
        self.decoder = DeepLabV3PlusDecoder(
            self.backbone.low_level_channels, 
            aspp_channels, 
            out_channels
        )
    
    def forward(self, x):
        input_size = x.shape[2:]
        
        # Extract features
        low_level, high_level = self.backbone(x)
        
        # ASPP
        aspp_out = self.aspp(high_level)
        
        # Decoder
        out = self.decoder(low_level, aspp_out)
        
        # Upsample to original size
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        
        return out
