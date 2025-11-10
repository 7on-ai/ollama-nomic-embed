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
# Expose ports
# -----------------------------
EXPOSE 11434  # Ollama
EXPOSE 5000   # Flask API

# -----------------------------
# Entrypoint script
# -----------------------------
RUN echo '#!/bin/bash
set -e

# Activate virtualenv
source /opt/venv/bin/activate

echo "🚀 Starting Ollama service..."
ollama serve > /var/log/ollama.log 2>&1 &
OLLAMA_PID=$!

# -----------------------------
# Wait for Ollama to be ready
# -----------------------------
echo "⏳ Waiting for Ollama to be ready..."
for i in {1..30}; do
  if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama is ready!"
    break
  fi
  echo "   Still waiting... ($i/30)"
  sleep 2
done

# -----------------------------
# Ensure required models exist
# -----------------------------
MODELS=$(curl -s http://localhost:11434/api/tags | grep -o "nomic-embed-text" || echo "")
if [ -z "$MODELS" ]; then
  echo "📥 Pulling required models..."
  ollama pull nomic-embed-text || echo "⚠️ Failed to pull nomic-embed-text"
  ollama pull mistral || echo "⚠️ Failed to pull mistral"
  echo "✅ Models pulled successfully"
else
  echo "✅ Models already exist"
fi

# -----------------------------
# Start Flask API
# -----------------------------
echo "🌐 Starting Flask API on port 5000..."
gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 7200 app:app &
API_PID=$!

# -----------------------------
# Wait for Flask API ready
# -----------------------------
for i in {1..30}; do
  if curl -s http://localhost:5000/health > /dev/null 2>&1; then
    echo "✅ Flask API is ready"
    break
  fi
  echo "   Waiting for Flask API... ($i/30)"
  sleep 2
done

echo "✅ All services started"
echo "   - Ollama: http://localhost:11434"
echo "   - Training API: http://localhost:5000"

# Keep services running
wait -n $OLLAMA_PID $API_PID
' > /entrypoint.sh && chmod +x /entrypoint.sh

# -----------------------------
# Set entrypoint
# -----------------------------
ENTRYPOINT ["/entrypoint.sh"]
