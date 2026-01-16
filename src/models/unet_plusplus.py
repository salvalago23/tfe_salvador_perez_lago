"""
U-Net++ (Nested U-Net) para segmentación de imágenes médicas

Referencia: Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image Segmentation"
https://arxiv.org/abs/1807.10165

Características principales:
- Conexiones densas anidadas entre encoder y decoder
- Mejora la fusión de características multi-escala
- Supervisión profunda opcional para mejor entrenamiento
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Bloque convolucional doble con BatchNorm y ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UNetPlusPlus(nn.Module):
    """
    Versión simplificada y más eficiente de U-Net++ para 4 niveles
    
    Esta implementación es más directa y sigue exactamente la estructura
    del paper original para L=4 niveles.
    """
    
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
                 deep_supervision=False):
        super().__init__()
        
        self.deep_supervision = deep_supervision
        f = features  # Alias para código más legible
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Encoder (backbone)
        self.conv0_0 = ConvBlock(in_channels, f[0])
        self.conv1_0 = ConvBlock(f[0], f[1])
        self.conv2_0 = ConvBlock(f[1], f[2])
        self.conv3_0 = ConvBlock(f[2], f[3])
        self.conv4_0 = ConvBlock(f[3], f[3] * 2)  # Bottleneck
        
        # Decoder nivel 0 (más superficial)
        self.up0_1 = nn.ConvTranspose2d(f[1], f[0], 2, 2)
        self.conv0_1 = ConvBlock(f[0] * 2, f[0])
        
        self.up0_2 = nn.ConvTranspose2d(f[1], f[0], 2, 2)
        self.conv0_2 = ConvBlock(f[0] * 3, f[0])
        
        self.up0_3 = nn.ConvTranspose2d(f[1], f[0], 2, 2)
        self.conv0_3 = ConvBlock(f[0] * 4, f[0])
        
        self.up0_4 = nn.ConvTranspose2d(f[1], f[0], 2, 2)
        self.conv0_4 = ConvBlock(f[0] * 5, f[0])
        
        # Decoder nivel 1
        self.up1_1 = nn.ConvTranspose2d(f[2], f[1], 2, 2)
        self.conv1_1 = ConvBlock(f[1] * 2, f[1])
        
        self.up1_2 = nn.ConvTranspose2d(f[2], f[1], 2, 2)
        self.conv1_2 = ConvBlock(f[1] * 3, f[1])
        
        self.up1_3 = nn.ConvTranspose2d(f[2], f[1], 2, 2)
        self.conv1_3 = ConvBlock(f[1] * 4, f[1])
        
        # Decoder nivel 2
        self.up2_1 = nn.ConvTranspose2d(f[3], f[2], 2, 2)
        self.conv2_1 = ConvBlock(f[2] * 2, f[2])
        
        self.up2_2 = nn.ConvTranspose2d(f[3], f[2], 2, 2)
        self.conv2_2 = ConvBlock(f[2] * 3, f[2])
        
        # Decoder nivel 3
        self.up3_1 = nn.ConvTranspose2d(f[3] * 2, f[3], 2, 2)
        self.conv3_1 = ConvBlock(f[3] * 2, f[3])
        
        # Output layers
        if deep_supervision:
            self.output1 = nn.Conv2d(f[0], out_channels, 1)
            self.output2 = nn.Conv2d(f[0], out_channels, 1)
            self.output3 = nn.Conv2d(f[0], out_channels, 1)
            self.output4 = nn.Conv2d(f[0], out_channels, 1)
        else:
            self.final_conv = nn.Conv2d(f[0], out_channels, 1)
    
    def forward(self, x):
        # Encoder
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        
        # Decoder - Primera diagonal
        x3_1 = self.conv3_1(torch.cat([x3_0, self._up(self.up3_1(x4_0), x3_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self._up(self.up2_1(x3_0), x2_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self._up(self.up1_1(x2_0), x1_0)], 1))
        x0_1 = self.conv0_1(torch.cat([x0_0, self._up(self.up0_1(x1_0), x0_0)], 1))
        
        # Decoder - Segunda diagonal
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self._up(self.up2_2(x3_1), x2_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self._up(self.up1_2(x2_1), x1_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self._up(self.up0_2(x1_1), x0_0)], 1))
        
        # Decoder - Tercera diagonal
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self._up(self.up1_3(x2_2), x1_0)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self._up(self.up0_3(x1_2), x0_0)], 1))
        
        # Decoder - Cuarta diagonal (final)
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self._up(self.up0_4(x1_3), x0_0)], 1))
        
        # Output
        if self.deep_supervision and self.training:
            out1 = self.output1(x0_1)
            out2 = self.output2(x0_2)
            out3 = self.output3(x0_3)
            out4 = self.output4(x0_4)
            return [out1, out2, out3, out4]
        else:
            if self.deep_supervision:
                return self.output4(x0_4)
            else:
                return self.final_conv(x0_4)
    
    def _up(self, x, target):
        """Ajusta el tamaño de x para que coincida con target"""
        if x.shape[2:] != target.shape[2:]:
            x = F.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=False)
        return x
