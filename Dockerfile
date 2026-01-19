# puls-events-chatbot-intelligent-rag/Dockerfile

# Image Python légère mais compatible FAISS
FROM python:3.12-slim

# Empêche Python de générer des fichiers .pyc
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Dossier de travail dans le container
WORKDIR /app

# Dépendances système requises par FAISS
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copier uniquement les dépendances pour profiter du cache Docker
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install fastapi uvicorn ragas datasets

# Copier tout le projet
COPY . .

# Exposer le port FastAPI
EXPOSE 8000

# Lancer l'API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
