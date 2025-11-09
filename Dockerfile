FROM ollama/ollama:latest

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    torch==2.1.0 transformers==4.36.0 peft==0.7.0 \
    bitsandbytes==0.41.3 accelerate==0.25.0 datasets==2.15.0 \
    psycopg2-binary==2.9.9 scikit-learn==1.3.2

RUN nohup ollama serve > /dev/null 2>&1 & \
    sleep 15 && \
    ollama pull nomic-embed-text && \
    ollama pull mistral && \
    pkill ollama

RUN mkdir -p /models/adapters /scripts

COPY scripts/train_complete.py /scripts/train_lora.py
RUN chmod +x /scripts/train_lora.py

EXPOSE 11434
CMD ["serve"]
