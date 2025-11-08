FROM ollama/ollama:latest

# Start Ollama in background, pull model, then stop
RUN nohup ollama serve > /dev/null 2>&1 & \
    sleep 15 && \
    ollama pull nomic-embed-text && \
    pkill ollama

# Expose port
EXPOSE 11434

# Default command (will be overridden by Northflank)
CMD ["serve"]