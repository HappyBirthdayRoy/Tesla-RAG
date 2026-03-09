FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONPATH=/app/src
ENV TESLA_RAG_CHROMA_DIR=/app/.chroma

EXPOSE 8000
CMD ["uvicorn", "tesla_rag.main:app", "--host", "0.0.0.0", "--port", "8000"]
