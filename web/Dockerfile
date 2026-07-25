# Container image for the FastAPI backend (Cloud Run / any container host).
# Build context must be the repo root — the backend imports
# frontend/src/components/chats.py as a sibling directory.
#
#   docker build -t ai-module-backend .
#   docker run -p 8080:8080 -e PORT=8080 ai-module-backend

FROM python:3.11-slim

# libgl1/libglib2.0-0: runtime libs opencv-python-headless still dynamically
# links against even without GUI support.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend backend
COPY frontend frontend

WORKDIR /app/backend

ENV PORT=8080
EXPOSE 8080

CMD ["python", "app.py"]
