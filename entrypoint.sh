#!/bin/bash
set -e

# -----------------------------
# Activate virtualenv
# -----------------------------
source /opt/venv/bin/activate

# -----------------------------
# Start Ollama service
# -----------------------------
echo "🚀 Starting Ollama service..."
ollama serve > /var/log/ollama.log 2>&1 &
OLLAMA_PID=$!

# Wait until Ollama is ready
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
# Pull required models
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
# Start Flask API (Training service)
# -----------------------------
echo "🌐 Starting Flask API on port 5000..."
gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 7200 app:app &
API_PID=$!

# Wait for Flask API ready
for i in {1..30}; do
    if curl -s http://localhost:5000/health > /dev/null 2>&1; then
        echo "✅ Flask API is ready"
        break
    fi
    echo "   Waiting for Flask API... ($i/30)"
    sleep 2
done

# -----------------------------
# All services started
# -----------------------------
echo "✅ All services started"
echo "   - Ollama: http://localhost:11434"
echo "   - Training API: http://localhost:5000"

# Keep both services running
wait -n $OLLAMA_PID $API_PID
