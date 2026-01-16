import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List

class DiceLoss(nn.Module):
    """Dice Loss para segmentación"""
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Asegurar que las predicciones estén entre 0 y 1
        pred = torch.sigmoid(pred)
        
        # Clampear target para estabilidad numérica
        pred = torch.clamp(pred, min=1e-7, max=1-1e-7)
        
        # Aplanar
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        
        # Calcular intersección y unión
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        # Dice coefficient
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice

class CombinedLoss(nn.Module):
    """Combina múltiples funciones de pérdida"""
    
    def __init__(self, 
                 losses: List[str] = ['bce', 'dice'],
                 weights: Optional[List[float]] = None,
                 **kwargs):
        super().__init__()
        
        self.losses = nn.ModuleDict()
        
        # Configurar pérdidas
        for loss_name in losses:
            if loss_name == 'bce':
                self.losses[loss_name] = nn.BCEWithLogitsLoss()
            elif loss_name == 'dice':
                self.losses[loss_name] = DiceLoss(**kwargs.get('dice_kwargs', {}))
        
        # Pesos (por defecto iguales)
        if weights is None:
            self.weights = {name: 1.0/len(losses) for name in losses}
        else:
            self.weights = dict(zip(losses, weights))
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        
        for name, loss_fn in self.losses.items():
            loss = loss_fn(pred, target)
            total_loss += self.weights[name] * loss
        
        return total_loss

class IoULoss(nn.Module):
    """Intersection over Union Loss"""
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        
        # Aplanar
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        
        # Calcular intersección y unión
        intersection = (pred_flat * target_flat).sum()
        total = pred_flat.sum() + target_flat.sum()
        union = total - intersection
        
        # IoU
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return 1 - iou