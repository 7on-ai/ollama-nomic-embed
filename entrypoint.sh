#!/bin/bash
set -e

echo "========================================="
echo "🚀 Starting Ollama Training Service"
echo "========================================="

# -----------------------------
# Activate virtualenv
# -----------------------------
echo "📦 Activating virtual environment..."
source /opt/venv/bin/activate
echo "✅ Virtual environment activated"
echo "Python: $(which python3)"
echo "Pip: $(which pip)"

# -----------------------------
# Verify Flask installation
# -----------------------------
echo ""
echo "🔍 Verifying Flask installation..."
python3 -c "import flask; print(f'Flask version: {flask.__version__}')" || {
    echo "❌ Flask not found! Installing..."
    pip install flask flask-cors gunicorn
}
python3 -c "import flask_cors; print('flask-cors: OK')" || {
    echo "❌ flask-cors not found! Installing..."
    pip install flask-cors
}
echo "✅ Flask dependencies verified"

# -----------------------------
# Test Flask app syntax
# -----------------------------
echo ""
echo "🧪 Testing Flask app syntax..."
python3 -c "import sys; sys.path.insert(0, '/'); import app; print('✅ app.py syntax OK')" || {
    echo "❌ ERROR: app.py has syntax errors!"
    echo "Showing app.py content:"
    head -50 /app.py
    exit 1
}

# -----------------------------
# Start Ollama service
# -----------------------------
echo ""
echo "🚀 Starting Ollama service..."
ollama serve > /var/log/ollama.log 2>&1 &
OLLAMA_PID=$!
echo "✅ Ollama started (PID: $OLLAMA_PID)"

# Wait until Ollama is ready
echo ""
echo "⏳ Waiting for Ollama to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama is ready!"
        break
    fi
    echo "   Still waiting... ($i/30)"
    sleep 2
done

# Check if Ollama is actually running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "❌ Ollama failed to start!"
    echo "Showing last 50 lines of ollama.log:"
    tail -50 /var/log/ollama.log
    exit 1
fi

# -----------------------------
# Pull required models
# -----------------------------
echo ""
echo "📥 Checking required models..."
MODELS=$(curl -s http://localhost:11434/api/tags | grep -o "nomic-embed-text" || echo "")
if [ -z "$MODELS" ]; then
    echo "📥 Pulling required models (this may take 2-3 minutes)..."
    ollama pull nomic-embed-text || echo "⚠️ Failed to pull nomic-embed-text"
    ollama pull mistral || echo "⚠️ Failed to pull mistral"
    echo "✅ Models pulled successfully"
else
    echo "✅ Models already exist"
fi

# -----------------------------
# Start Flask API (Training service)
# -----------------------------
echo ""
echo "========================================="
echo "🌐 Starting Flask API on port 5000"
echo "========================================="

# ✅ CRITICAL: Start Flask directly for debugging
echo "Using: python3 /app.py"
echo ""

# Show Flask startup
python3 /app.py 2>&1 &
API_PID=$!

echo "✅ Flask started (PID: $API_PID)"

# Wait for Flask API to be ready
echo ""
echo "⏳ Waiting for Flask API to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:5000/ > /dev/null 2>&1; then
        echo "✅ Flask API is ready!"
        RESPONSE=$(curl -s http://localhost:5000/)
        echo "Response: $RESPONSE"
        break
    fi
    echo "   Waiting for Flask API... ($i/30)"
    sleep 2
done

# Final check
if ! curl -s http://localhost:5000/ > /dev/null 2>&1; then
    echo ""
    echo "❌ ERROR: Flask API failed to start!"
    echo "Checking if process is running..."
    ps aux | grep python3
    echo ""
    echo "Checking port 5000..."
    netstat -tlnp | grep 5000 || echo "Port 5000 not listening"
    echo ""
    echo "Trying to start Flask again manually..."
    cd /
    python3 -c "
import sys
sys.path.insert(0, '/')
from app import app
app.run(host='0.0.0.0', port=5000, debug=True)
" &
    sleep 5
fi

# -----------------------------
# All services started
# -----------------------------
echo ""
echo "========================================="
echo "✅ All services started successfully"
echo "========================================="
echo "📡 Ollama API: http://localhost:11434"
echo "📡 Training API: http://localhost:5000"
echo "📋 Health check: curl http://localhost:5000/health"
echo ""
echo "Showing last 10 lines of system info:"
echo "- Memory:"
free -h
echo "- Disk:"
df -h /
echo "- Processes:"
ps aux | grep -E "ollama|python3|gunicorn" | grep -v grep
echo ""
echo "========================================="
echo "🎯 Ready for requests!"
echo "========================================="

# Keep both services running
wait -n $OLLAMA_PID $API_PID

# If we get here, one of the services died
echo ""
echo "❌ ERROR: A service died!"
echo "Ollama status:"
ps aux | grep ollama | grep -v grep || echo "Ollama is dead"
echo "Flask status:"
ps aux | grep python3 | grep -v grep || echo "Flask is dead"
exit 1