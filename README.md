# Asistente de CV (RAG ligero + Evaluador LLM)

Asistente conversacional que responde sobre el CV de <<nombre_persona>> usando:
- **RAG ligero** (TF-IDF + cosine) sobre un PDF.
- **Groq** (compat OpenAI) para el agente.
- **Evaluación automática** con otro modelo Groq (Llama 3.1 8B Instant).
- **Gradio** para la interfaz web.
- **Streaming** con fallback.


## Requisitos
- Python 3.11 (sugerido) o entorno Conda.
- Cuenta y API Key de **Groq**.
- (Opcional) Conda para `environment.yml`.

## Instalación (pip)
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
