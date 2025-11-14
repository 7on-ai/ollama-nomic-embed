# --------------------------------------------------
# Base image: PyTorch official (CPU only)
# --------------------------------------------------
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

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
# Install Python packages (no bitsandbytes for CPU)
# --------------------------------------------------
RUN pip install --no-cache-dir \
    transformers \
    peft \
    accelerate \
    datasets \
    psycopg2-binary \
    scikit-learn

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