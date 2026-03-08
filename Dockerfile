FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY scripts/ ./scripts/
COPY web/ ./web/
COPY cluster_model/ ./cluster_model/
COPY chroma_db/ ./chroma_db/
COPY cache_data/ ./cache_data/
COPY start.sh .

RUN chmod +x start.sh

RUN mkdir -p cache_data chroma_db cluster_model

EXPOSE 8000
EXPOSE 5000

CMD ["./start.sh"]