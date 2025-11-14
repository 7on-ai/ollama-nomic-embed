#!/bin/bash
set -e

echo "=========================================="
echo "🚀 LoRA Training Job Started"
echo "=========================================="
echo "📅 Time: $(date)"
echo "💻 Hostname: $(hostname)"
echo "📂 Working dir: $(pwd)"
echo "=========================================="

# ===== Validate Environment Variables =====
echo ""
echo "📋 Validating environment variables..."

if [ -z "$POSTGRES_URI" ]; then
  echo "❌ ERROR: POSTGRES_URI is not set"
  exit 1
fi

if [ -z "$USER_ID" ]; then
  echo "❌ ERROR: USER_ID is not set"
  exit 1
fi

if [ -z "$MODEL_NAME" ]; then
  echo "❌ ERROR: MODEL_NAME is not set"
  exit 1
fi

echo "✅ All required environment variables are set"

# ===== Generate and Export Adapter Version =====
export ADAPTER_VERSION=${ADAPTER_VERSION:-"v$(date +%s)"}

# ===== Set and Export Output Directory =====
export OUTPUT_DIR="${OUTPUT_PATH:-/workspace/adapters}/${USER_ID}/${ADAPTER_VERSION}"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "📦 Configuration:"
echo "   USER_ID:         $USER_ID"
echo "   MODEL_NAME:      $MODEL_NAME"
echo "   ADAPTER_VERSION: $ADAPTER_VERSION"
echo "   OUTPUT_DIR:      $OUTPUT_DIR"
echo "   POSTGRES_URI:    [hidden for security]"
echo "=========================================="

# ===== Run Training Script =====
echo ""
echo "🏋️  Starting training..."
echo "=========================================="

python3 /workspace/scripts/train_complete.py

# ===== Check Result =====
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Training completed successfully!"
    echo "=========================================="
    echo "📁 Output location: $OUTPUT_DIR"
    echo "📊 Files created:"
    ls -lh "$OUTPUT_DIR"
    echo "=========================================="
    exit 0
else
    echo ""
    echo "=========================================="
    echo "❌ Training failed!"
    echo "=========================================="
    exit 1
fi