## Prerrequisitos
- Git LFS (para poder descargar los modelos)

## Descarga del proyecto
- git lfs install
- git clone https://github.com/salvalago23/tfe_salvador_perez_lago.git

# Instalación (en el directorio raíz del proyecto)
- conda env create -f environment.yml
- conda activate tfm_segmentacion_env

- python3 download_dataset.py


## ARCHIVOS DESCARGADOS (ISIC 2018)
El dataset completo ocupa ~13 GB

- ISIC2018 Training Input (10.46 GB) → 2594 imágenes training
- ISIC2018 Training GroundTruth (33 MB) → 2594 máscaras training
- ISIC2018 Validation Input (228 MB) → 100 imágenes validation
- ISIC2018 Validation GroundTruth (1 MB) → 100 máscaras validation
- ISIC2018 Test Input (2.26 GB) → 1000 imágenes test
- ISIC2018 Test GroundTruth (11 MB) → 1000 máscaras test


### Todo el EDA, entrenamiento de modelos y análisis de los resultados de los mismos se pueden encontrar en los 3 Jupyter Notebook del directorio ./notebooks

### Para ejecutar la app, basta con moverse al directorio ./app y ejecutar el archivo bash run.sh


## Estructura del Proyecto

```
tfm_segmentacion/
├── README.md                      # Documentación principal del proyecto
├── environment.yml                # Dependencias del entorno conda
├── download_dataset.py            # Script de descarga del dataset ISIC 2018
│
├── app/                          # Aplicación web de segmentación
│   ├── README.md                 # Documentación específica de la app
│   ├── run.sh                    # Script de ejecución del servidor
│   ├── backend/                  # Backend FastAPI
│   │   ├── main.py              # Servidor con API REST
│   │   ├── clinical_analysis/   # Análisis clínico ABCD
│   │   └── data/                # Base de datos y resultados
│   └── frontend/                # Interfaz web
│       └── index.html           # SPA con visualización de resultados
│
├── data/                         # Datos del proyecto
│   ├── raw/                     # Datos originales ISIC 2018
│   │   ├── isic_2018_train/    # 2594 imágenes + máscaras
│   │   ├── isic_2018_val/      # 100 imágenes + máscaras
│   │   └── isic_2018_test/     # 1000 imágenes + máscaras
│   └── processed/               # Datos preprocesados (.npy)
│       ├── train/               # Imágenes y máscaras normalizadas
│       ├── val/
│       └── test/
│
├── notebooks/                    # Jupyter Notebooks experimentales
│   ├── 01_eda.ipynb             # Análisis exploratorio de datos
│   ├── 02_model_train.ipynb     # Entrenamiento de modelos
│   ├── 03_model_eval.ipynb      # Evaluación y comparación
│   └── experiments/             # Resultados experimentales
│       ├── eda_results/         # Estadísticas del dataset
│       ├── notebook_experiments/ # Modelos entrenados
│       │   ├── u-net_básica/
│       │   ├── attention_u-net/
│       │   ├── residual_u-net/
│       │   ├── u-net++/
│       │   ├── deeplabv3+/
│       │   └── transunet/
│       └── comparative_analysis/ # Comparación de rendimiento
│
└── src/                          # Código fuente modular
    ├── data_preparation/         # Gestión de datos
    │   ├── data_loader.py       # DataLoader con balanceo de clases
    │   └── dataset.py           # Dataset ISIC con augmentación
    ├── models/                   # Arquitecturas de segmentación
    │   ├── __init__.py
    │   ├── unet.py              # U-Net básica
    │   ├── attention_unet.py    # U-Net con mecanismos de atención
    │   ├── residual_unet.py     # U-Net con bloques residuales
    │   ├── unet_plusplus.py     # U-Net++ (nested U-Net)
    │   ├── deeplabv3plus.py     # DeepLabV3+ con atrous convolutions
    │   ├── transunet.py         # TransUNet (transformers + U-Net)
    │   ├── losses.py            # Funciones de pérdida (Dice, BCE, combinadas)
    │   └── metrics.py           # Métricas de evaluación (Dice, IoU, F1)
    └── train_eval_scripts/       # Scripts de entrenamiento y evaluación
        ├── train_model.py        # Pipeline de entrenamiento
        └── evaluate_model.py     # Evaluación de modelos guardados
```

## Descripción de Componentes

### Raíz del Proyecto

#### `download_dataset.py`
Script automatizado para descargar y organizar el dataset ISIC 2018 Challenge (Task 1). Descarga ~13 GB de imágenes dermatoscópicas y máscaras de segmentación ground truth para training (2594), validation (100) y test (1000).

#### `environment.yml`
Define el entorno conda con todas las dependencias necesarias: PyTorch, FastAPI, albumentations, OpenCV, etc.

### `src/` - Código Fuente

#### `src/data_preparation/`
- **`dataset.py`**: Implementa `ISIC2018Dataset` con soporte para:
  - Carga de imágenes raw (.jpg/.png) o procesadas (.npy)
  - Augmentación en tiempo real usando Albumentations
  - Normalización con estadísticas ISIC específicas
  - Manejo de casos edge (imágenes corruptas, máscaras faltantes)

- **`data_loader.py`**: Clase `ISICDataLoader` que gestiona:
  - Creación de datasets para train/val/test
  - Balanceo de clases mediante WeightedRandomSampler
  - Configuración optimizada de DataLoaders (num_workers, pin_memory)

#### `src/models/`
Implementaciones de 6 arquitecturas:
- **U-Net básica**: Encoder-decoder clásico con skip connections
- **Attention U-Net**: Attention gates para enfoque selectivo
- **Residual U-Net**: Bloques residuales para entrenamiento profundo
- **U-Net++**: Nested skip connections con deep supervision
- **DeepLabV3+**: Atrous spatial pyramid pooling para multi-escala
- **TransUNet**: Hybrid transformers + CNN para contexto global

**Módulos auxiliares**:
- `losses.py`: DiceLoss, CombinedLoss (BCE + Dice)
- `metrics.py`: Dice Score, IoU, Precision/Recall/F1

### `notebooks/` - Análisis Experimental

- **`01_eda.ipynb`**: Análisis exploratorio (distribuciones, estadísticas, visualizaciones)
- **`02_model_train.ipynb`**: Experimentación con hiperparámetros y arquitecturas
- **`03_model_eval.ipynb`**: Comparación cuantitativa y cualitativa de modelos

Los modelos entrenados se guardan en `notebooks/experiments/notebook_experiments/`, cada uno con:
- `best_model.pth`: Checkpoint del mejor modelo
- `training_history.json`: Métricas por época
- Configuración utilizada

### `app/` - Aplicación Web

Ver [app/README.md](app/README.md) para detalles completos.

Aplicación web full-stack para uso clínico:
- Backend FastAPI con inferencia en tiempo real
- Frontend con drag & drop y visualización interactiva
- Gestión de pacientes y seguimiento longitudinal
- Análisis automático de características ABCD dermatológicas

### `data/` - Dataset

- **`raw/`**: Imágenes originales ISIC 2018 en formato .jpg/.png
- **`processed/`**: Versión preprocesada en .npy (normalización + resize) para carga más rápida durante entrenamiento
