# Saral Seva: Automated ID Processing & Form Pre-Filling System

Saral Seva is an automated document processing system that captures identity cards via optical camera, extracts multilingual text using PaddleOCR, and structures information using on-device Ollama language models to pre-fill public service forms error-free.

---

## 🛠️ Environment Variables & Configuration

Create a `.env` file in the root directory (or configure system environment variables):

```env
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral

# Application Settings
STREAMLIT_SERVER_PORT=8501
DEBUG_MODE=False
CAMERA_INDEX=0