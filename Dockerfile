# --------------------------------------------------
# Base image: PyTorch official (CPU only)
# --------------------------------------------------
FROM pytorch/pytorch:2.3.1-cpu

# --------------------------------------------------
# Install system dependencies
# --------------------------------------------------
RUN apt-get update && apt-get install -y \
    git \
    curl \
    python3-dev \
    python3-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------
# Create working directories
# --------------------------------------------------
WORKDIR /workspace
RUN mkdir -p /workspace/scripts /workspace/models/adapters

# --------------------------------------------------
# Install Python packages
# --------------------------------------------------
RUN pip install --no-cache-dir \
    transformers \
    peft \
    accelerate \
    datasets \
    psycopg2-binary \
    scikit-learn \
    flask \
    flask-cors \
    gunicorn

# --------------------------------------------------
# Copy training script (optional)
# --------------------------------------------------
COPY scripts/train_complete.py /workspace/scripts/train_complete.py
RUN chmod +x /workspace/scripts/train_complete.py

# --------------------------------------------------
# Optional: Flask API (if you expose training service)
# --------------------------------------------------
COPY app.py /workspace/app.py

# --------------------------------------------------
# Entrypoint script (for both job & API)
# --------------------------------------------------
COPY entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

# --------------------------------------------------
# Expose Flask API port
# --------------------------------------------------
EXPOSE 5000

# --------------------------------------------------
# Default entrypoint
# --------------------------------------------------
ENTRYPOINT ["/workspace/entrypoint.sh"]
