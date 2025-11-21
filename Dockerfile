# --------------------------------------------------
# Base image: PyTorch official (newer version)
# --------------------------------------------------
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# --------------------------------------------------
# Install system dependencies
# --------------------------------------------------
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------
# Create working directories
# --------------------------------------------------
WORKDIR /workspace
RUN mkdir -p /workspace/scripts /workspace/adapters

# --------------------------------------------------
# Install Python packages (with 8-bit support)
# --------------------------------------------------
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    transformers==4.38.0 \
    peft==0.8.0 \
    accelerate==0.27.0 \
    bitsandbytes==0.42.0 \
    datasets==2.17.0 \
    psycopg2-binary==2.9.9 \
    scikit-learn==1.4.0

# --------------------------------------------------
# ✅ Pre-download model to speed up training
# --------------------------------------------------
RUN python3 -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0'); \
    AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')"

RUN echo '✅ Model pre-cached in image'

# --------------------------------------------------
# Copy training script
# --------------------------------------------------
COPY scripts/train_complete.py /workspace/scripts/train_complete.py
RUN chmod +x /workspace/scripts/train_complete.py

# --------------------------------------------------
# Copy entrypoint script
# --------------------------------------------------
COPY entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

# --------------------------------------------------
# Set environment variables
# --------------------------------------------------
ENV OUTPUT_PATH=/workspace/adapters
ENV PYTHONUNBUFFERED=1

# --------------------------------------------------
# Default entrypoint (Job mode - no Flask)
# --------------------------------------------------
ENTRYPOINT ["/workspace/entrypoint.sh"]
