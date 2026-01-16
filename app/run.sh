#!/bin/bash
# Ejecuta la aplicación web de segmentación de lesiones cutáneas

# Obtener el directorio donde se encuentra el script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colores para la salida
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # Sin color

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}    Skin Lesion Segmentation Web Application${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Establecer PYTHONPATH para incluir la raíz del proyecto
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Ejecutar el servidor
echo ""
echo "Starting server..."
echo "Open http://localhost:8000 in your browser"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd "$SCRIPT_DIR/backend"
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
