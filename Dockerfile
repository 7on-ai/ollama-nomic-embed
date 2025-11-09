FROM ollama/ollama:latest

# Install dependencies in smaller chunks to avoid I/O errors
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages in stages to reduce memory pressure
RUN pip3 install --no-cache-dir \
    torch==2.1.0 \
    transformers==4.36.0

RUN pip3 install --no-cache-dir \
    peft==0.7.0 \
    bitsandbytes==0.41.3 \
    accelerate==0.25.0

RUN pip3 install --no-cache-dir \
    datasets==2.15.0 \
    psycopg2-binary==2.9.9 \
    scikit-learn==1.3.2

# Create directories
RUN mkdir -p /models/adapters /scripts

# Copy training script
COPY scripts/train_complete.py /scripts/train_lora.py
RUN chmod +x /scripts/train_lora.py

# Expose port
EXPOSE 11434

# Pull models at runtime, not build time
# Create an entrypoint script
RUN echo '#!/bin/bash\n\
if [ ! -f /root/.ollama/models/manifests/registry.ollama.ai/library/nomic-embed-text/latest ]; then\n\
  echo "Pulling models..."\n\
  nohup ollama serve > /dev/null 2>&1 &\n\
  sleep 15\n\
  ollama pull nomic-embed-text\n\
  ollama pull mistral\n\
  pkill ollama\n\
  sleep 5\n\
fi\n\
exec ollama serve\n\
' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
