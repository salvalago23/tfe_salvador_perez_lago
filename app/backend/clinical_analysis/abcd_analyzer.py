"""
ABCD Clinical Analyzer
Módulo para extraer características clínicas según criterios dermatológicos
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List
from scipy.spatial import distance
from sklearn.cluster import KMeans


@dataclass
class ABCDFeatures:
    """Almacena todas las características ABCD extraídas"""
    # A - Asymmetry
    asymmetry_score: float
    asymmetry_x: float
    asymmetry_y: float
    
    # B - Border
    border_irregularity: float
    compactness: float
    perimeter_variation: float
    
    # C - Color
    num_colors: int
    color_variance: float
    has_blue_gray: bool
    has_white: bool
    has_red: bool
    dominant_colors: List = field(default_factory=list)
    
    # D - Diameter
    diameter_mm: float = 0.0
    major_axis: float = 0.0
    minor_axis: float = 0.0
    area_mm2: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario para fácil exportación"""
        return {
            'asymmetry_score': self.asymmetry_score,
            'asymmetry_x': self.asymmetry_x,
            'asymmetry_y': self.asymmetry_y,
            'border_irregularity': self.border_irregularity,
            'compactness': self.compactness,
            'perimeter_variation': self.perimeter_variation,
            'num_colors': self.num_colors,
            'color_variance': self.color_variance,
            'has_blue_gray': self.has_blue_gray,
            'has_white': self.has_white,
            'has_red': self.has_red,
            'diameter_mm': self.diameter_mm,
            'major_axis': self.major_axis,
            'minor_axis': self.minor_axis,
            'area_mm2': self.area_mm2
        }


class ABCDAnalyzer:
    """
    Analizador completo de características ABCD
    """
    
    def __init__(self, pixels_per_mm: float = 10.0):
        """
        Args:
            pixels_per_mm: Factor de conversión píxeles a milímetros
                          (por defecto 10, ajustar según calibración del dataset)
        """
        self.pixels_per_mm = pixels_per_mm
    
    def analyze(self, image: np.ndarray, mask: np.ndarray) -> ABCDFeatures:
        """
        Realiza análisis completo ABCD
        
        Args:
            image: Imagen RGB original (H, W, 3)
            mask: Máscara binaria de segmentación (H, W)
        
        Returns:
            ABCDFeatures con todas las métricas calculadas
        """
        # Asegurar que la máscara es binaria
        mask_binary = (mask > 0.5).astype(np.uint8)
        
        # Calcular cada criterio
        asymmetry = self._calculate_asymmetry(mask_binary)
        border = self._calculate_border(mask_binary)
        color = self._calculate_color(image, mask_binary)
        diameter = self._calculate_diameter(mask_binary)
        
        return ABCDFeatures(
            asymmetry_score=asymmetry['score'],
            asymmetry_x=asymmetry['x_axis'],
            asymmetry_y=asymmetry['y_axis'],
            border_irregularity=border['irregularity'],
            compactness=border['compactness'],
            perimeter_variation=border['perimeter_variation'],
            num_colors=color['num_colors'],
            color_variance=color['variance'],
            has_blue_gray=color['has_blue_gray'],
            has_white=color['has_white'],
            has_red=color['has_red'],
            dominant_colors=color['dominant_colors'],
            diameter_mm=diameter['diameter_mm'],
            major_axis=diameter['major_axis'],
            minor_axis=diameter['minor_axis'],
            area_mm2=diameter['area_mm2']
        )
    
    def _calculate_asymmetry(self, mask: np.ndarray) -> Dict:
        """
        A - Asymmetry: Calcula asimetría en dos ejes
        """
        # Encontrar contornos y momentos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return {'score': 0, 'x_axis': 0, 'y_axis': 0}
        
        cnt = max(contours, key=cv2.contourArea)
        M = cv2.moments(cnt)
        
        if M['m00'] == 0:
            return {'score': 0, 'x_axis': 0, 'y_axis': 0}
        
        # Centroide
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        # Asimetría en eje X (vertical)
        left_half = mask[:, :cx]
        right_half = mask[:, cx:]
        right_half_flipped = np.fliplr(right_half)
        
        # Hacer las mitades del mismo tamaño
        min_width = min(left_half.shape[1], right_half_flipped.shape[1])
        left_half = left_half[:, -min_width:]
        right_half_flipped = right_half_flipped[:, :min_width]
        
        # XOR para encontrar diferencias
        diff_x = np.logical_xor(left_half, right_half_flipped)
        asymmetry_x = np.sum(diff_x) / np.sum(mask) if np.sum(mask) > 0 else 0
        
        # Asimetría en eje Y (horizontal)
        top_half = mask[:cy, :]
        bottom_half = mask[cy:, :]
        bottom_half_flipped = np.flipud(bottom_half)
        
        min_height = min(top_half.shape[0], bottom_half_flipped.shape[0])
        top_half = top_half[-min_height:, :]
        bottom_half_flipped = bottom_half_flipped[:min_height, :]
        
        diff_y = np.logical_xor(top_half, bottom_half_flipped)
        asymmetry_y = np.sum(diff_y) / np.sum(mask) if np.sum(mask) > 0 else 0
        
        # Score total (promedio de ambos ejes)
        asymmetry_score = (asymmetry_x + asymmetry_y) / 2
        
        return {
            'score': float(asymmetry_score),
            'x_axis': float(asymmetry_x),
            'y_axis': float(asymmetry_y)
        }
    
    def _calculate_border(self, mask: np.ndarray) -> Dict:
        """
        B - Border: Analiza irregularidad de bordes
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return {'irregularity': 0, 'compactness': 0, 'perimeter_variation': 0}
        
        cnt = max(contours, key=cv2.contourArea)
        
        # Área y perímetro
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        if area == 0 or perimeter == 0:
            return {'irregularity': 0, 'compactness': 0, 'perimeter_variation': 0}
        
        # Compacidad (círculo perfecto = 1.0)
        compactness = (4 * np.pi * area) / (perimeter ** 2)
        
        # Calcular variación de distancias desde centroide
        M = cv2.moments(cnt)
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        centroid = np.array([cx, cy])
        
        points = cnt.reshape(-1, 2)
        distances = np.array([distance.euclidean(p, centroid) for p in points])
        
        # Irregularidad como coeficiente de variación
        irregularity = np.std(distances) / np.mean(distances) if np.mean(distances) > 0 else 0
        
        # Variación del perímetro (usando aproximación poligonal)
        epsilon = 0.01 * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        perimeter_variation = len(approx) / len(cnt)
        
        return {
            'irregularity': float(irregularity),
            'compactness': float(compactness),
            'perimeter_variation': float(perimeter_variation)
        }
    
    def _calculate_color(self, image: np.ndarray, mask: np.ndarray) -> Dict:
        """
        C - Color: Analiza variedad y distribución de colores
        """
        # Extraer píxeles dentro de la máscara
        lesion_pixels = image[mask > 0]
        
        if len(lesion_pixels) == 0:
            return {
                'num_colors': 0,
                'variance': 0,
                'has_blue_gray': False,
                'has_white': False,
                'has_red': False,
                'dominant_colors': []
            }
        
        # Convertir a HSV para mejor análisis
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lesion_hsv = image_hsv[mask > 0]
        
        # Clustering de colores (3-6 colores típicamente)
        n_clusters = min(6, len(lesion_pixels))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(lesion_hsv)
        
        # Número efectivo de colores (clusters con >5% de píxeles)
        labels, counts = np.unique(kmeans.labels_, return_counts=True)
        significant_colors = np.sum(counts > (0.05 * len(lesion_pixels)))
        
        # Varianza de color
        color_variance = np.mean(np.var(lesion_hsv, axis=0))
        
        # Detectar colores de riesgo en RGB
        lesion_rgb = lesion_pixels
        
        # Azul-gris: B > R y B > G
        blue_gray_pixels = (lesion_rgb[:, 2] > lesion_rgb[:, 0]) & \
                           (lesion_rgb[:, 2] > lesion_rgb[:, 1])
        has_blue_gray = np.sum(blue_gray_pixels) > (0.1 * len(lesion_pixels))
        
        # Blanco: R, G, B todos > 200
        white_pixels = np.all(lesion_rgb > 200, axis=1)
        has_white = np.sum(white_pixels) > (0.1 * len(lesion_pixels))
        
        # Rojo: R > 150 y R > G y R > B
        red_pixels = (lesion_rgb[:, 0] > 150) & \
                     (lesion_rgb[:, 0] > lesion_rgb[:, 1]) & \
                     (lesion_rgb[:, 0] > lesion_rgb[:, 2])
        has_red = np.sum(red_pixels) > (0.1 * len(lesion_pixels))
        
        # Colores dominantes (centroides en RGB)
        dominant_colors = kmeans.cluster_centers_.tolist()
        
        return {
            'num_colors': int(significant_colors),
            'variance': float(color_variance),
            'has_blue_gray': bool(has_blue_gray),
            'has_white': bool(has_white),
            'has_red': bool(has_red),
            'dominant_colors': dominant_colors
        }
    
    def _calculate_diameter(self, mask: np.ndarray) -> Dict:
        """
        D - Diameter: Calcula dimensiones de la lesión
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return {
                'diameter_mm': 0,
                'major_axis': 0,
                'minor_axis': 0,
                'area_mm2': 0
            }
        
        cnt = max(contours, key=cv2.contourArea)
        
        # Ajustar elipse
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (center, axes, angle) = ellipse
            major_axis_px = max(axes)
            minor_axis_px = min(axes)
        else:
            # Si no hay suficientes puntos, usar bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            major_axis_px = max(w, h)
            minor_axis_px = min(w, h)
        
        # Área
        area_px = cv2.contourArea(cnt)
        
        # Convertir a mm
        major_axis_mm = major_axis_px / self.pixels_per_mm
        minor_axis_mm = minor_axis_px / self.pixels_per_mm
        area_mm2 = area_px / (self.pixels_per_mm ** 2)
        
        return {
            'diameter_mm': float(major_axis_mm),
            'major_axis': float(major_axis_mm),
            'minor_axis': float(minor_axis_mm),
            'area_mm2': float(area_mm2)
        }
    
    
    def visualize_analysis(self, image: np.ndarray, mask: np.ndarray, 
                          features: ABCDFeatures) -> np.ndarray:
        """
        Crea visualización de los resultados del análisis ABCD
        
        Returns:
            Imagen con visualizaciones superpuestas
        """
        vis = image.copy()
        
        # Superponer máscara con transparencia
        overlay = vis.copy()
        overlay[mask > 0] = [255, 0, 0]  # Rojo
        vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
        
        # Dibujar contorno
        contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                       cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
        
        # Añadir texto con métricas
        y_offset = 30
        metrics_text = [
            f"Asymmetry: {features.asymmetry_score:.3f}",
            f"Border Irreg: {features.border_irregularity:.3f}",
            f"Colors: {features.num_colors}",
            f"Diameter: {features.diameter_mm:.2f}mm"
        ]
        
        for text in metrics_text:
            cv2.putText(vis, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30
        
        return vis