# ==============================
#   Dockerfile - FastAPI Service
# ==============================

# 1️⃣ Base image oficial leve com Python 3.13
FROM python:3.13-slim AS base

# 2️⃣ Define diretório de trabalho
WORKDIR /app

# 3️⃣ Instala dependências do sistema (build + pandas + psycopg2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 4️⃣ Copia os arquivos de dependência
COPY requirements.txt .

# 5️⃣ Instala as dependências Python (sem cache, mais rápido)
RUN pip install --no-cache-dir -r requirements.txt

# 6️⃣ Copia o código da aplicação
COPY . .

# 7️⃣ Define variáveis de ambiente padrão (podem ser sobrescritas)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# 8️⃣ Expõe a porta do FastAPI
EXPOSE 8000

# 9️⃣ Comando para iniciar o servidor
# --host 0.0.0.0 permite que ele aceite conexões externas (container)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
