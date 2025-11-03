FROM python:3.12-slim

# Configuración base de Python en contenedor
ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PORT=8000

WORKDIR /app

# Instala dependencias primero para aprovechar caché de capas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del código
COPY . .

# Puerto por defecto (documentación); la app respetará $PORT si la plataforma lo inyecta
EXPOSE 8000

# Respetar $PORT cuando la plataforma lo provee (Render/Railway/Fly/EB)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
