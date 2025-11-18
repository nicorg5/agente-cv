# Dockerfile
FROM python:3.11-slim AS base

# Evitar que Python genere .pyc y usar stdout directo
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Crear directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema (pypdf, scikit-learn pueden necesitarlas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copiar el código de la app
COPY app ./app

# (Opcional) si tu CV y resumen los metes en el contenedor:
# COPY app/data ./app/data

# Exponer el puerto del backend
EXPOSE 8000

# Comando por defecto: lanzar FastAPI con uvicorn
CMD ["uvicorn", "app.backend:app", "--host", "0.0.0.0", "--port", "8000"]