# ============================================
# LoRA Training Service with FastAPI
# ============================================
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python packages
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    transformers==4.38.0 \
    peft==0.8.0 \
    accelerate==0.27.0 \
    bitsandbytes==0.42.0 \
    datasets==2.17.0 \
    psycopg2-binary==2.9.9 \
    scikit-learn==1.4.0 \
    fastapi==0.109.0 \
    uvicorn[standard]==0.27.0 \
    pydantic==2.5.0

# Pre-download model
RUN python3 -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0'); \
    AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')"

# Copy training script and API server
COPY scripts/train_complete.py /workspace/scripts/train_complete.py
COPY scripts/api_server.py /workspace/scripts/api_server.py
RUN chmod +x /workspace/scripts/*.py

# Environment
ENV OUTPUT_PATH=/models/adapters
ENV PYTHONUNBUFFERED=1
ENV API_PORT=8000

# Expose API port
EXPOSE 8000

# Start API server
CMD ["python3", "/workspace/scripts/api_server.py"]