"""
TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation

Referencia: Chen et al., "TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation"
https://arxiv.org/abs/2102.04306

Características principales:
- Combina CNN (ResNet) como encoder con Vision Transformer
- Captura dependencias globales mediante self-attention
- Decoder tipo U-Net para recuperar resolución espacial
- Estado del arte en segmentación médica
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple


class PatchEmbedding(nn.Module):
    """
    Convierte feature maps en secuencia de patches para el Transformer
    """
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 1):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        # x: (B, C, H, W)
        x = self.projection(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, embed_dim)
        x = self.norm(x)
        return x, H, W


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention para Vision Transformer
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """
    MLP block para Transformer
    """
    def __init__(self, in_features: int, hidden_features: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    Bloque Transformer con Pre-Norm
    """
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CNNEncoder(nn.Module):
    """
    Encoder CNN simple (similar a ResNet simplificado)
    Extrae características multi-escala
    """
    def __init__(self, in_channels: int = 3, base_features: int = 64):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Stages (similar a ResNet)
        self.layer1 = self._make_layer(base_features, base_features, 2)
        self.layer2 = self._make_layer(base_features, base_features * 2, 2, stride=2)
        self.layer3 = self._make_layer(base_features * 2, base_features * 4, 2, stride=2)
        self.layer4 = self._make_layer(base_features * 4, base_features * 8, 2, stride=2)
        
        self.out_channels = [base_features, base_features, base_features * 2, 
                            base_features * 4, base_features * 8]
    
    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1):
        layers = []
        
        # Primera capa con posible downsampling
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # Capas adicionales
        for _ in range(1, blocks):
            layers.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        
        x = self.stem(x)
        features.append(x)  # 1/4
        
        x = self.layer1(x)
        features.append(x)  # 1/4
        
        x = self.layer2(x)
        features.append(x)  # 1/8
        
        x = self.layer3(x)
        features.append(x)  # 1/16
        
        x = self.layer4(x)
        features.append(x)  # 1/32
        
        return features


class DecoderBlock(nn.Module):
    """
    Bloque de decoder con skip connection
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.up(x)
        
        if skip is not None:
            # Ajustar tamaño si es necesario
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        
        x = self.conv(x)
        return x


class TransUNet(nn.Module):
    """
    TransUNet: Híbrido CNN-Transformer para segmentación de imágenes médicas
    
    Arquitectura:
    1. CNN Encoder extrae características multi-escala
    2. Transformer procesa las características de mayor nivel
    3. Decoder tipo U-Net reconstruye la máscara de segmentación
    
    Args:
        in_channels: Número de canales de entrada (default: 3)
        out_channels: Número de canales de salida (default: 1)
        img_size: Tamaño de la imagen de entrada (default: 256)
        base_features: Características base del encoder CNN (default: 64)
        embed_dim: Dimensión del embedding del Transformer (default: 768)
        num_heads: Número de cabezas de atención (default: 12)
        num_layers: Número de bloques Transformer (default: 12)
        mlp_ratio: Ratio de expansión del MLP (default: 4.0)
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        img_size: int = 256,
        base_features: int = 32,#64,
        embed_dim: int = 256,#768,
        num_heads: int = 8,#12,
        num_layers: int = 4,#12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.img_size = img_size
        self.embed_dim = embed_dim
        
        # CNN Encoder
        self.encoder = CNNEncoder(in_channels, base_features)
        encoder_out_channels = self.encoder.out_channels[-1]  # 512
        
        # Patch Embedding (convierte features del encoder a secuencia)
        self.patch_embed = PatchEmbedding(encoder_out_channels, embed_dim, patch_size=1)
        
        # Calcular número de patches
        self.num_patches = (img_size // 32) ** 2
        
        # Position Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer Blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # Proyección de vuelta a espacio de características
        self.proj_back = nn.Conv2d(embed_dim, encoder_out_channels, kernel_size=1)
        
        # Decoder
        decoder_channels = [base_features * 8, base_features * 4, base_features * 2, 
                          base_features, base_features]
        
        self.decoder_blocks = nn.ModuleList()
        
        # Decoder block 1: 1/32 -> 1/16
        self.decoder_blocks.append(DecoderBlock(decoder_channels[0], base_features * 4, decoder_channels[1]))
        # Decoder block 2: 1/16 -> 1/8
        self.decoder_blocks.append(DecoderBlock(decoder_channels[1], base_features * 2, decoder_channels[2]))
        # Decoder block 3: 1/8 -> 1/4
        self.decoder_blocks.append(DecoderBlock(decoder_channels[2], base_features, decoder_channels[3]))
        # Decoder block 4: 1/4 -> 1/2
        self.decoder_blocks.append(DecoderBlock(decoder_channels[3], 0, decoder_channels[4]))
        
        # Final upsampling y clasificación
        self.final_up = nn.ConvTranspose2d(decoder_channels[4], decoder_channels[4], kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(decoder_channels[4], out_channels, kernel_size=1)
        
        # Inicialización
        self._init_weights()
    
    def _init_weights(self):
        # Inicializar position embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Inicializar otros pesos
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        # CNN Encoder - extraer características multi-escala
        encoder_features = self.encoder(x)
        # encoder_features: [feat_1/4, feat_1/4, feat_1/8, feat_1/16, feat_1/32]
        
        # Último feature map para Transformer
        x = encoder_features[-1]  # (B, 512, H/32, W/32)
        
        # Patch Embedding
        x, H, W = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Añadir position embedding
        x = x + self.pos_embed[:, :x.shape[1], :]
        x = self.pos_drop(x)
        
        # Transformer Blocks
        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm(x)
        
        # Reshape de vuelta a feature map
        x = x.transpose(1, 2).reshape(B, self.embed_dim, H, W)
        
        # Proyección de vuelta a espacio de características del encoder
        x = self.proj_back(x)
        
        # Decoder con skip connections
        x = self.decoder_blocks[0](x, encoder_features[3])  # skip from 1/16
        x = self.decoder_blocks[1](x, encoder_features[2])  # skip from 1/8
        x = self.decoder_blocks[2](x, encoder_features[1])  # skip from 1/4
        x = self.decoder_blocks[3](x, None)  # no skip
        
        # Final upsampling
        x = self.final_up(x)
        x = self.final_conv(x)
        
        return x