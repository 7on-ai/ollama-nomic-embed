FROM ollama/ollama:latest

# -----------------------------
# Install system dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Create virtual environment
# -----------------------------
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# -----------------------------
# Install Python packages
# -----------------------------
RUN pip install --no-cache-dir \
    torch \
    transformers \
    peft \
    bitsandbytes \
    accelerate \
    datasets \
    psycopg2-binary \
    scikit-learn \
    flask \
    gunicorn

# -----------------------------
# Create directories
# -----------------------------
RUN mkdir -p /models/adapters /scripts

# -----------------------------
# Copy training script
# -----------------------------
COPY scripts/train_complete.py /scripts/train_lora.py
RUN chmod +x /scripts/train_lora.py

# -----------------------------
# Copy Flask API
# -----------------------------
COPY app.py /app.py

# -----------------------------
# Copy entrypoint script
# -----------------------------
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# -----------------------------
# Expose ports
# -----------------------------
EXPOSE 11434  # Ollama
EXPOSE 5000   # Training API (Flask)

# -----------------------------
# Entrypoint
# -----------------------------
ENTRYPOINT ["/entrypoint.sh"]
