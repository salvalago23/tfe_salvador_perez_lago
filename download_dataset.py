import os
import requests
import zipfile
import shutil
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# URLs para descargar
urls = [
    "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Training_Input.zip",
    "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Training_GroundTruth.zip",
    "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Validation_Input.zip",
    "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Validation_GroundTruth.zip",
    "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Test_Input.zip",
    "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Test_GroundTruth.zip"
]

def download_file(url, dest_folder="data"):
    """Descargar un archivo con barra de progreso"""
    # Crear la carpeta de destino si no existe
    Path(dest_folder).mkdir(parents=True, exist_ok=True)
    
    # Extraer el nombre del archivo de la URL
    filename = url.split("/")[-1]
    filepath = os.path.join(dest_folder, filename)
    
    # Comprobar si el archivo ya existe
    if os.path.exists(filepath):
        print(f"{filename} already exists, skipping download...")
        return filepath
    
    print(f"Downloading {filename}...")
    
    try:
        # Descargar en streaming
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Obtener el tamaño total del archivo
        total_size = int(response.headers.get('content-length', 0))
        
        # Descargar con barra de progreso
        with open(filepath, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                pbar.update(size)
        
        print(f"{filename} downloaded successfully\n")
        return filepath
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {filename}: {e}\n")
        return None

def unzip_file(zip_path, dest_folder="data"):
    """Descomprimir un archivo y mostrar el progreso"""
    if not os.path.exists(zip_path):
        print(f"{zip_path} not found, skipping unzip")
        return
    
    filename = os.path.basename(zip_path)
    print(f"Unzipping {filename}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Obtener la lista de archivos en el archivo comprimido
            file_list = zip_ref.namelist()
            
            # Extraer con barra de progreso
            for file in tqdm(file_list, desc=f"Extracting {filename}"):
                zip_ref.extract(file, dest_folder)
        
        print(f"{filename} extracted successfully\n")
        return True
        
    except zipfile.BadZipFile as e:
        print(f"Error unzipping {filename}: {e}\n")
        return False

def organize_dataset(base_folder="data"):
    """Organizar el dataset en la estructura de carpetas raw"""
    print("\n" + "=" * 50)
    print("Organizing dataset into structured folders...\n")
    
    # Definir el mapeo de origen a destino
    mappings = [
        ("ISIC2018_Task1-2_Training_Input", "raw/isic_2018_train/images"),
        ("ISIC2018_Task1_Training_GroundTruth", "raw/isic_2018_train/masks"),
        ("ISIC2018_Task1-2_Test_Input", "raw/isic_2018_test/images"),
        ("ISIC2018_Task1_Test_GroundTruth", "raw/isic_2018_test/masks"),
        ("ISIC2018_Task1-2_Validation_Input", "raw/isic_2018_val/images"),
        ("ISIC2018_Task1_Validation_GroundTruth", "raw/isic_2018_val/masks"),
    ]
    
    # Crear la estructura de carpetas raw
    raw_folders = [
        "raw/isic_2018_train/images",
        "raw/isic_2018_train/masks",
        "raw/isic_2018_test/images",
        "raw/isic_2018_test/masks",
        "raw/isic_2018_val/images",
        "raw/isic_2018_val/masks",
    ]
    
    for folder in raw_folders:
        Path(os.path.join(base_folder, folder)).mkdir(parents=True, exist_ok=True)
    
    print("Created folder structure\n")
    
    # Mover archivos según los mapeos
    for source_folder, dest_folder in mappings:
        source_path = os.path.join(base_folder, source_folder)
        dest_path = os.path.join(base_folder, dest_folder)
        
        if not os.path.exists(source_path):
            print(f"Warning: {source_folder} not found, skipping...")
            continue
        
        # Obtener todos los archivos jpg y png
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(Path(source_path).rglob(ext))
        
        if not image_files:
            print(f"No image files found in {source_folder}")
            continue
        
        print(f"Moving {len(image_files)} files from {source_folder} to {dest_folder}")
        
        # Mover archivos con barra de progreso
        for img_file in tqdm(image_files, desc=f"Moving to {dest_folder}"):
            dest_file = os.path.join(dest_path, img_file.name)
            shutil.move(str(img_file), dest_file)
        
        print(f"Completed moving files to {dest_folder}\n")
    
    # Limpiar carpetas de origen vacías
    print("Cleaning up empty source folders...")
    for source_folder, _ in mappings:
        source_path = os.path.join(base_folder, source_folder)
        if os.path.exists(source_path):
            try:
                shutil.rmtree(source_path)
                print(f"Removed {source_folder}")
            except Exception as e:
                print(f"Could not remove {source_folder}: {e}")
    
    print("\nDataset organization complete!")

def resize_with_aspect_ratio_padding(img, target_size, interpolation, pad_value=0):
    """
    Redimensiona una imagen/máscara preservando la relación de aspecto
    y añadiendo padding centrado para alcanzar el tamaño objetivo.
    
    Esto evita distorsiones geométricas que afectarían la morfología de la lesión
    (criterio diagnóstico clave en la regla ABCD para dermatoscopia).
    
    Args:
        img: Imagen o máscara a redimensionar (numpy array)
        target_size: Tupla (ancho, alto) del tamaño objetivo
        interpolation: Método de interpolación de OpenCV
        pad_value: Valor de relleno para el padding
    
    Returns:
        Imagen/máscara redimensionada con padding
    """
    target_w, target_h = target_size
    h, w = img.shape[:2]
    
    # Calcular factor de escala para que el lado más largo quepa
    scale = min(target_w / w, target_h / h)
    
    # Nuevas dimensiones manteniendo aspect ratio
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Redimensionar
    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    
    # Calcular padding (centrado)
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    
    # Aplicar padding
    if len(img.shape) == 3:
        # Imagen con canales (H, W, C)
        padded = cv2.copyMakeBorder(
            resized,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=[pad_value] * img.shape[2]
        )
    else:
        # Máscara (H, W)
        padded = cv2.copyMakeBorder(
            resized,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=pad_value
        )
    
    return padded


def preprocess_image(image_path, target_size=(256, 256)):
    """
    Preprocesa una imagen preservando la relación de aspecto.
    
    Aplica resize manteniendo aspect ratio + padding centrado para evitar
    distorsiones geométricas que afectarían la morfología de la lesión.
    
    Args:
        image_path: Ruta a la imagen
        target_size: Tamaño objetivo (ancho, alto)
    
    Returns:
        Imagen preprocesada como uint8 [0-255]
    """
    # Leer imagen
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convertir BGR a RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Redimensionar preservando aspect ratio con padding negro
    img = resize_with_aspect_ratio_padding(
        img, 
        target_size, 
        interpolation=cv2.INTER_LINEAR,
        pad_value=0  # Padding negro
    )
    
    # Mantener como uint8 [0-255] - la normalización se hace en el Dataset
    return img.astype(np.uint8)


def preprocess_mask(mask_path, target_size=(256, 256)):
    """
    Preprocesa una máscara preservando la relación de aspecto.
    
    Aplica resize manteniendo aspect ratio + padding centrado.
    Usa interpolación NEAREST para preservar valores binarios.
    
    Args:
        mask_path: Ruta a la máscara
        target_size: Tamaño objetivo (ancho, alto)
    
    Returns:
        Máscara preprocesada binaria (0 o 1)
    """
    # Leer máscara (escala de grises)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")
    
    # Binarizar ANTES de redimensionar para evitar artefactos
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Redimensionar preservando aspect ratio con padding (valor 0 = fondo)
    # Usar INTER_NEAREST para máscaras binarias
    mask = resize_with_aspect_ratio_padding(
        mask, 
        target_size, 
        interpolation=cv2.INTER_NEAREST,
        pad_value=0  # Padding con valor de fondo
    )
    
    # Convertir a 0 y 1
    mask = (mask / 255).astype(np.uint8)
    
    return mask

def preprocess_dataset(base_folder="data", target_size=(256, 256)):
    """
    Preprocesar todo el dataset y guardar como archivos .npy
    
    Args:
        base_folder: Directorio base que contiene los datos raw
        target_size: Tamaño objetivo (alto, ancho)
    """
    print("\n" + "=" * 50)
    print("PREPROCESSING DATASET")
    print("=" * 50)
    
    raw_dir = Path(base_folder) / "raw"
    processed_dir = Path(base_folder) / "processed"
    
    datasets = [
        ('isic_2018_train', 'train'),
        ('isic_2018_val', 'val'),
        ('isic_2018_test', 'test')
    ]
    
    for raw_name, proc_name in datasets:
        print(f"\n{'='*50}")
        print(f"Processing dataset: {proc_name}")
        print(f"{'='*50}")
        
        # Crear directorios de salida
        dataset_output = processed_dir / proc_name
        (dataset_output / "images").mkdir(parents=True, exist_ok=True)
        (dataset_output / "masks").mkdir(parents=True, exist_ok=True)
        
        # Rutas de entrada
        input_images = raw_dir / raw_name / "images"
        input_masks = raw_dir / raw_name / "masks"
        
        # Obtener lista de imágenes
        image_files = list(input_images.glob("*.jpg")) + list(input_images.glob("*.png"))
        
        if not image_files:
            print(f"No images found in {input_images}")
            continue
        
        print(f"Found {len(image_files)} images to process")
        
        # Procesar cada imagen
        processed_count = 0
        for img_path in tqdm(image_files, desc=f"Processing {proc_name}"):
            img_id = img_path.stem
            
            # Determinar la ruta de la máscara
            mask_path = None
            if input_masks:
                mask_path = input_masks / f"{img_id}_segmentation.png"
                if not mask_path.exists():
                    # Intentar sin el sufijo _segmentation
                    mask_path = input_masks / f"{img_id}.png"
                    if not mask_path.exists():
                        print(f"\nMask not found for {img_id}, skipping...")
                        continue
            
            try:
                # Procesar imagen (uint8 [0-255])
                image_proc = preprocess_image(img_path, target_size)
                
                # Guardar imagen procesada
                img_output = dataset_output / "images" / f"{img_id}.npy"
                np.save(img_output, image_proc)
                
                # Procesar y guardar máscara
                mask_proc = preprocess_mask(mask_path, target_size)
                mask_output = dataset_output / "masks" / f"{img_id}.npy"
                np.save(mask_output, mask_proc)
                
                processed_count += 1
                
            except Exception as e:
                print(f"\nError processing {img_id}: {e}")
                continue
        
        print(f"Successfully processed {processed_count}/{len(image_files)} images for {proc_name}")
    
    print("\nPreprocessing completed!")
    
    # Mostrar estadísticas finales
    print("\n" + "="*50)
    print("FINAL PREPROCESSING STATISTICS")
    print("="*50)
    
    for _, proc_name in datasets:
        dataset_path = processed_dir / proc_name / "images"
        n_images = len(list(dataset_path.glob("*.npy"))) if dataset_path.exists() else 0
        
        masks_path = processed_dir / proc_name / "masks"
        n_masks = len(list(masks_path.glob("*.npy"))) if masks_path.exists() else 0
        print(f"{proc_name.upper()}: {n_images} images, {n_masks} masks")

def main():
    print("ISIC 2018 Challenge Dataset Downloader & Preprocessor")
    print("=" * 50)
    print(f"Downloading {len(urls)} files...\n")
    
    downloaded_files = []
    
    # Descargar todos los archivos
    for url in urls:
        filepath = download_file(url)
        if filepath:
            downloaded_files.append(filepath)
    
    print("\n" + "=" * 50)
    print("Unzipping files...\n")
    
    # Descomprimir todos los archivos descargados
    for filepath in downloaded_files:
        unzip_file(filepath)
    
    print("=" * 50)
    print("Deleting original zip files...\n")
    
    # Eliminar archivos zip
    for filepath in downloaded_files:
        try:
            os.remove(filepath)
            print(f"Deleted {os.path.basename(filepath)}")
        except Exception as e:
            print(f"Error deleting {os.path.basename(filepath)}: {e}")
    
    # Organizar el dataset
    organize_dataset()
    
    # Preprocesar el dataset
    preprocess_dataset()
    
    print("\nDataset is ready for training")


if __name__ == "__main__":
    main()