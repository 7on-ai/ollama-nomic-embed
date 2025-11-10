FROM ollama/ollama:latest

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install Python packages with compatible versions
RUN pip install --no-cache-dir \
    torch \
    transformers

RUN pip install --no-cache-dir \
    peft \
    bitsandbytes \
    accelerate

RUN pip install --no-cache-dir \
    datasets \
    psycopg2-binary \
    scikit-learn

# Create directories
RUN mkdir -p /models/adapters /scripts

# Copy training script
COPY scripts/train_complete.py /scripts/train_lora.py
RUN chmod +x /scripts/train_lora.py

# Expose port
EXPOSE 11434

# ✅ FIXED: Better entrypoint with model pulling
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Activate virtualenv\n\
source /opt/venv/bin/activate\n\
\n\
echo "🚀 Starting Ollama service..."\n\
\n\
# Start Ollama in background\n\
ollama serve > /var/log/ollama.log 2>&1 &\n\
OLLAMA_PID=$!\n\
\n\
echo "⏳ Waiting for Ollama to be ready..."\n\
sleep 10\n\
\n\
# Wait for Ollama to be fully ready\n\
for i in {1..30}; do\n\
  if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then\n\
    echo "✅ Ollama is ready!"\n\
    break\n\
  fi\n\
  echo "   Still waiting... ($i/30)"\n\
  sleep 2\n\
done\n\
\n\
# Check if models exist\n\
MODELS=$(curl -s http://localhost:11434/api/tags | grep -o "nomic-embed-text" || echo "")\n\
\n\
if [ -z "$MODELS" ]; then\n\
  echo "📥 Pulling required models..."\n\
  ollama pull nomic-embed-text || echo "⚠️  Failed to pull nomic-embed-text"\n\
  ollama pull mistral || echo "⚠️  Failed to pull mistral"\n\
  echo "✅ Models pulled successfully"\n\
else\n\
  echo "✅ Models already exist"\n\
fi\n\
\n\
# Keep Ollama running in foreground\n\
wait $OLLAMA_PID\n\
' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
