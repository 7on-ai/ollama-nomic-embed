FROM ollama/ollama:latest

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
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

# Entrypoint with venv activated
RUN echo '#!/bin/bash\n\
source /opt/venv/bin/activate\n\
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
