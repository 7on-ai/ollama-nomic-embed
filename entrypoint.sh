#!/bin/bash
set -e

echo "🚀 Starting LoRA Training Job..."

# Required ENV
if [ -z "$POSTGRES_URI" ] || [ -z "$USER_ID" ] || [ -z "$BASE_MODEL" ]; then
  echo "❌ Missing required environment variables:"
  echo "   POSTGRES_URI, USER_ID, BASE_MODEL are required."
  exit 1
fi

# Adapter name (auto generate if not provided)
ADAPTER_NAME=${ADAPTER_NAME:-"lora_$(date +%s)"}

# Output directory
OUTPUT_DIR="/workspace/models/adapters/${USER_ID}/${ADAPTER_NAME}"
mkdir -p "$OUTPUT_DIR"

echo "📦 Environment setup:"
echo "   USER_ID:        $USER_ID"
echo "   BASE_MODEL:     $BASE_MODEL"
echo "   ADAPTER_NAME:   $ADAPTER_NAME"
echo "   OUTPUT_DIR:     $OUTPUT_DIR"
echo "   POSTGRES_URI:   [hidden]"

# Run training
python3 /workspace/scripts/train_complete.py \
    "$POSTGRES_URI" \
    "$USER_ID" \
    "$BASE_MODEL" \
    "$ADAPTER_NAME"

# Check result
if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully."
    echo "📁 Output stored at: $OUTPUT_DIR"
else
    echo "❌ Training failed."
    exit 1
fi

# Optional: keep container alive for inspection (comment out if not needed)
# tail -f /dev/null
