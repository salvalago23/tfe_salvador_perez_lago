# Aplicación Web de Segmentación de Lesiones Cutáneas

Una aplicación web para la segmentación automática de lesiones cutáneas usando modelos de aprendizaje profundo.

## Características

- **Carga de Imágenes**: Sube imágenes dermatoscópicas para análisis mediante arrastrar y soltar o selección de archivos
- **Múltiples Modelos**: Elige entre varios modelos de segmentación entrenados (U-Net, Attention U-Net, ResidualU-Net, U-Net++, DeepLabV3+, TransUNet)
- **Inferencia en Tiempo Real**: Obtén resultados de segmentación con mediciones de tiempo de inferencia
- **Gestión de Pacientes**: Crea perfiles de pacientes para hacer seguimiento de la evolución de lesiones a lo largo del tiempo
- **Historial de Resultados**: Almacena y compara análisis previos para cada paciente
- **Visualización**: Visualiza la imagen original, la máscara de segmentación y la superposición

## Uso

```bash
chmod +x run.sh
./run.sh
```

Luego abre http://localhost:8000 en tu navegador.


## Endpoints de la API

### Salud e Información
- `GET /api/health` - Verificación de salud e información del dispositivo
- `GET /api/models` - Lista de modelos disponibles

### Pacientes
- `POST /api/patients` - Crear nuevo paciente
- `GET /api/patients` - Listar todos los pacientes
- `GET /api/patients/{id}` - Obtener detalles del paciente
- `DELETE /api/patients/{id}` - Eliminar paciente
- `GET /api/patients/{id}/results` - Obtener historial de análisis del paciente

### Predicción
- `POST /api/predict` - Ejecutar segmentación en imagen cargada
  - Parámetros: `file` (imagen), `model` (clave del modelo), `patient_id`
  - Devuelve: Imagen original, máscara de segmentación, superposición, tiempo de inferencia, cobertura de la lesión

## Estructura del Proyecto

```
app/
├── README.md              # Documentación de la aplicación
├── run.sh                 # Script de ejecución del servidor
├── backend/               # Backend FastAPI
│   ├── main.py           # Servidor principal con endpoints de la API
│   ├── clinical_analysis/ # Módulo de análisis clínico
│   │   ├── __init__.py
│   │   └── abcd_analyzer.py  # Analizador de criterios ABCD dermatológicos
│   └── data/             # Datos persistentes
│       ├── patients.json # Base de datos de pacientes
│       └── results/      # Resultados de análisis guardados
└── frontend/             # Interfaz de usuario
    └── index.html        # Aplicación web de página única
```

### Descripción de Componentes

#### `run.sh`
Script bash que configura el entorno y ejecuta el servidor FastAPI. Establece el PYTHONPATH correcto y lanza uvicorn en el puerto 8000.

#### `backend/main.py`
Servidor principal de la aplicación que incluye:
- Gestión de modelos de segmentación (U-Net, Attention U-Net, ResidualU-Net, U-Net++, DeepLabV3+, TransUNet)
- Endpoints REST API para predicción, gestión de pacientes y consultas
- Preprocesamiento de imágenes y postprocesamiento de máscaras
- Integración con analizador ABCD para extracción de características clínicas
- Sistema de almacenamiento de resultados con historial por paciente

#### `backend/clinical_analysis/abcd_analyzer.py`
Módulo especializado en análisis dermatológico que extrae características según criterios ABCD:
- **A (Asymmetry)**: Calcula asimetría en ejes X e Y
- **B (Border)**: Evalúa irregularidad del borde y compacidad
- **C (Color)**: Detecta número de colores, varianza y presencia de colores clínicamente relevantes
- **D (Diameter)**: Mide diámetro, ejes mayor/menor y área de la lesión

#### `backend/data/`
Almacenamiento persistente:
- `patients.json`: Base de datos JSON con información de pacientes
- `results/`: Directorio con imágenes y máscaras de análisis previos

#### `frontend/index.html`
Interfaz web completa que incluye:
- Sistema de carga de imágenes (drag & drop o selección)
- Selector de modelos de segmentación
- Gestión de pacientes (crear, listar, eliminar)
- Visualización de resultados (imagen original, máscara, overlay)
- Historial de análisis por paciente
- Dashboard con métricas y características clínicas
- Diseño responsive y moderno con estilos CSS integrados

