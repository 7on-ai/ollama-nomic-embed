FROM ollama/ollama:latest

# Start Ollama in background, pull model, then stop
RUN nohup ollama serve > /dev/null 2>&1 & \
    sleep 15 && \
    ollama pull nomic-embed-text && \
    pkill ollama

# Expose port
EXPOSE 11434

# Run Ollama server
CMD ["ollama", "serve"]