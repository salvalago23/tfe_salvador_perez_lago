import torch
import numpy as np
from typing import Tuple, List, Optional

def dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Calcula el Dice Score"""
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum()
    
    return (2. * intersection / (union + 1e-6)).item()

def iou_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Calcula el Intersection over Union"""
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection
    
    return (intersection / (union + 1e-6)).item()

def precision_recall_f1(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Tuple[float, float, float]:
    """Calcula precisión, recall y F1-score"""
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    
    # Verdaderos positivos, falsos positivos, falsos negativos
    tp = (pred_bin * target).sum()
    fp = (pred_bin * (1 - target)).sum()
    fn = ((1 - pred_bin) * target).sum()
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    return precision.item(), recall.item(), f1.item()

class SegmentationMetrics:
    """Clase para calcular múltiples métricas"""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        self.dice_scores = []
        self.iou_scores = []
        self.precisions = []
        self.recalls = []
        self.f1_scores = []
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        dice = dice_score(pred, target, self.threshold)
        iou = iou_score(pred, target, self.threshold)
        precision, recall, f1 = precision_recall_f1(pred, target, self.threshold)
        
        self.dice_scores.append(dice)
        self.iou_scores.append(iou)
        self.precisions.append(precision)
        self.recalls.append(recall)
        self.f1_scores.append(f1)
    
    def compute(self) -> dict:
        return {
            'dice': np.mean(self.dice_scores) if self.dice_scores else 0.0,
            'iou': np.mean(self.iou_scores) if self.iou_scores else 0.0,
            'precision': np.mean(self.precisions) if self.precisions else 0.0,
            'recall': np.mean(self.recalls) if self.recalls else 0.0,
            'f1': np.mean(self.f1_scores) if self.f1_scores else 0.0,
        }