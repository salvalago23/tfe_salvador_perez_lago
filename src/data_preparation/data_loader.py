import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from .dataset import ISIC2018Dataset
from typing import Optional, Tuple

class ISICDataLoader:
    """
    DataLoader personalizado para el dataset ISIC 2018
    """
    
    def __init__(self, 
                 base_path: str = "data",
                 batch_size: int = 8,
                 target_size: Tuple[int, int] = (256, 256),
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 use_class_balancing: bool = True,
                 debug: bool = False):
        """
        Args:
            base_path: Ruta base de datos
            batch_size: Tamaño del batch
            target_size: Tamaño objetivo de imágenes
            num_workers: Número de workers para DataLoader
            pin_memory: Si usar pin memory (mejor para GPU)
            use_class_balancing: Si balancear clases (solo train)
            debug: Modo debug
        """
        self.base_path = base_path
        self.batch_size = batch_size
        self.target_size = target_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_class_balancing = use_class_balancing
        self.debug = debug
        
        # Crear datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Crear dataloaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
    def create_datasets(self, 
                       use_processed: bool = True,
                       augment_train: bool = True):
        """
        Crea datasets para train, val y test
        
        Args:
            use_processed: Si usar datos preprocesados (.npy) o raw (.jpg)
            augment_train: Si aplicar aumentación al train
        """
        print("Creando datasets...")
        
        # Determinar rutas según si usamos procesados o raw
        if use_processed:
            data_subdir = "processed"
            file_extension = "npy"
        else:
            data_subdir = "raw"
            file_extension = "jpg"
        
        # Dataset de entrenamiento
        train_images_dir = f"{self.base_path}/{data_subdir}/train/images"
        train_masks_dir = f"{self.base_path}/{data_subdir}/train/masks"
        
        self.train_dataset = ISIC2018Dataset(
            images_dir=train_images_dir,
            masks_dir=train_masks_dir,
            phase='train',
            target_size=self.target_size,
            augment=augment_train,
            debug=self.debug
        )
        
        print(f"  Train: {len(self.train_dataset)} imágenes")
        
        # Dataset de validación
        val_images_dir = f"{self.base_path}/{data_subdir}/val/images"
        val_masks_dir = f"{self.base_path}/{data_subdir}/val/masks"
        
        self.val_dataset = ISIC2018Dataset(
            images_dir=val_images_dir,
            masks_dir=val_masks_dir,
            phase='val',
            target_size=self.target_size,
            augment=False,  # Sin aumentación en validación
            debug=self.debug
        )
        
        print(f"  Val: {len(self.val_dataset)} imágenes")
        
        # Dataset de test
        test_images_dir = f"{self.base_path}/{data_subdir}/test/images"
        test_masks_dir = f"{self.base_path}/{data_subdir}/test/masks"
        
        self.test_dataset = ISIC2018Dataset(
            images_dir=test_images_dir,
            masks_dir=test_masks_dir,
            phase='test',
            target_size=self.target_size,
            augment=False,
            debug=self.debug
        )
        print(f"  Test: {len(self.test_dataset)} imágenes")
    
    def _create_weighted_sampler(self, dataset: ISIC2018Dataset) -> WeightedRandomSampler:
        """
        Crea un sampler ponderado para balancear clases
        
        Args:
            dataset: Dataset para calcular pesos
            
        Returns:
            WeightedRandomSampler para balanceo
        """
        print("Creando weighted sampler...")
        
        # Calcular pesos para cada muestra basado en cobertura de máscara
        sample_weights = []
        
        for i in range(len(dataset)):
            try:
                # Obtener máscara (forma 1xHxW)
                mask = dataset[i]['mask']
                
                # Calcular peso: muestras con más foreground tienen mayor peso
                foreground_ratio = mask.mean().item()
                
                # Peso inverso: dar más peso a muestras con poco foreground
                # (porque hay más background que foreground en general)
                if foreground_ratio > 0:
                    weight = 1.0 / (foreground_ratio + 0.1)  # +0.1 para evitar infinito
                else:
                    weight = 0.5  # Peso para muestras sin foreground
                
                sample_weights.append(weight)
                
            except Exception as e:
                # En caso de error, peso neutral
                sample_weights.append(1.0)
        
        # Convertir a tensor
        sample_weights = torch.tensor(sample_weights, dtype=torch.float32)
        
        # Normalizar pesos
        sample_weights = sample_weights / sample_weights.sum()
        
        # Crear sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        print(f"  Sampler creado con {len(sample_weights)} pesos")
        
        return sampler
    
    def create_dataloaders(self, shuffle_train: bool = True, 
                           generator: Optional[torch.Generator] = None,
                           worker_init_fn: Optional[callable] = None):
        """
        Crea DataLoaders a partir de los datasets
        
        Args:
            shuffle_train: Si mezclar datos de entrenamiento
            generator: Generator de PyTorch para reproducibilidad
            worker_init_fn: Función de inicialización de workers para reproducibilidad
        """
        print("Creando DataLoaders...")
        
        if not self.train_dataset:
            raise ValueError("Primero crea los datasets con create_datasets()")
        
        # Sampler para train (balanceado o no)
        train_sampler = None
        if self.use_class_balancing and shuffle_train:
            train_sampler = self._create_weighted_sampler(self.train_dataset)
            shuffle_train = False  # El sampler maneja el shuffling
        
        # DataLoader de entrenamiento
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle_train,
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,  # Útil para batch normalization
            persistent_workers=True if self.num_workers > 0 else False,
            generator=generator,
            worker_init_fn=worker_init_fn
        )
        
        print(f"  Train Loader: {len(self.train_loader)} batches")
        
        # DataLoader de validación (sin shuffling)
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            generator=generator,
            worker_init_fn=worker_init_fn
        )
        
        print(f"  Val Loader: {len(self.val_loader)} batches")
        
        # DataLoader de test (si existe)
        if self.test_dataset:
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=True if self.num_workers > 0 else False,
                generator=generator,
                worker_init_fn=worker_init_fn
            )
            print(f"  Test Loader: {len(self.test_loader)} batches")
    
    def get_class_weights(self) -> Optional[torch.Tensor]:
        """
        Obtiene pesos de clase para pérdida ponderada
        
        Returns:
            Tensor con pesos [background_weight, foreground_weight]
        """
        if self.train_dataset:
            return self.train_dataset.get_class_weights()
        return None
    
    def visualize_batch(self, num_samples: int = 4, phase: str = 'train'):
        """
        Visualiza un batch de datos
        
        Args:
            num_samples: Número de muestras a visualizar
            phase: 'train', 'val', o 'test'
        """
        import matplotlib.pyplot as plt
        
        # Obtener DataLoader correspondiente
        if phase == 'train' and self.train_loader:
            dataloader = self.train_loader
        elif phase == 'val' and self.val_loader:
            dataloader = self.val_loader
        elif phase == 'test' and self.test_loader:
            dataloader = self.test_loader
        else:
            print(f"DataLoader para fase '{phase}' no disponible")
            return
        
        # Obtener un batch
        batch = next(iter(dataloader))
        images = batch['image']
        masks = batch['mask']
        
        # Configurar figura
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Batch de {phase.upper()} (tamaño: {images.shape})', fontsize=16, y=1.02)
        
        for i in range(min(num_samples, images.shape[0])):
            # Desnormalizar imagen
            img_np = images[i].cpu().numpy()
            img_np = np.transpose(img_np, (1, 2, 0))
            
            # Desnormalizar usando stats de ISIC 2018
            mean = np.array([0.7082, 0.5819, 0.5359])
            std = np.array([0.1574, 0.1657, 0.1808])
            img_np = img_np * std + mean
            img_np = np.clip(img_np, 0, 1)
            
            # Máscara
            mask_np = masks[i, 0].cpu().numpy() if masks is not None else None
            
            # Imagen original
            axes[i, 0].imshow(img_np)
            axes[i, 0].set_title(f"Imagen {batch['image_id'][i][:10]}...")
            axes[i, 0].axis('off')
            
            # Máscara (si existe)
            if mask_np is not None:
                axes[i, 1].imshow(mask_np, cmap='gray', vmin=0, vmax=1)
                axes[i, 1].set_title(f"Máscara (cov: {mask_np.mean():.2%})")
                axes[i, 1].axis('off')
                
                # Superposición
                overlay = img_np.copy()
                overlay[mask_np > 0.5, 1] = 1.0  # Verde para lesión
                axes[i, 2].imshow(overlay)
                axes[i, 2].set_title("Superposición")
                axes[i, 2].axis('off')
            else:
                axes[i, 1].text(0.5, 0.5, "Sin máscara", 
                              ha='center', va='center', fontsize=12)
                axes[i, 1].axis('off')
                axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Estadísticas del batch
        print(f"\nEstadísticas del batch ({phase}):")
        print(f"  Batch shape: {images.shape}")
        print(f"  Rango imagen: [{images.min():.3f}, {images.max():.3f}]")
        if masks is not None:
            print(f"  Rango máscara: [{masks.min():.3f}, {masks.max():.3f}]")
            print(f"  Cobertura media batch: {masks.mean():.3%}")