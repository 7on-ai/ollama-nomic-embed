#!/usr/bin/env python3
"""
LoRA Training Pipeline - COMPLETE WITH POSTGRES STORAGE
✅ Incremental training
✅ Save adapter to Postgres (not Volume)
✅ Ollama integration (for cloud Ollama only)
"""

import os
import sys
import json
import random
import gc
import requests
import time
import re
from datetime import datetime
from pathlib import Path

print("="*60)
print("🚀 LoRA Training + Postgres Storage")
print("="*60)
print(f"Time: {datetime.utcnow().isoformat()}Z")
print(f"Python: {sys.version}")
print(f"Working Dir: {os.getcwd()}")
print("="*60)

# ===== Imports =====
print("\n📋 STEP 1: Checking imports...")
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
    from peft import LoraConfig, get_peft_model, PeftModel, TaskType
    from datasets import Dataset
    import psycopg2
    from psycopg2.extras import RealDictCursor
    print(f"  ✅ All imports successful")
except Exception as e:
    print(f"  ❌ Import failed: {e}")
    sys.exit(1)

import warnings
warnings.filterwarnings("ignore")

# ===== Configuration =====
CONFIG = {
    "r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "learning_rate": 2e-5,
    "num_epochs": 2,
    "batch_size": 1,
    "max_length": 512,
    "gradient_accumulation_steps": 4,
    "min_samples_new": 3,
    "max_samples_per_training": 500,
}

# ===== Environment =====
print("\n📋 STEP 2: Validating environment...")
POSTGRES_URI = os.environ.get("POSTGRES_URI")
USER_ID = os.environ.get("USER_ID")
MODEL_NAME = os.environ.get("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
OUTPUT_BASE = "/tmp/adapters"  # ✅ Temporary only (won't persist)
NORTHFLANK_API_TOKEN = os.environ.get("NORTHFLANK_API_TOKEN")
NORTHFLANK_PROJECT_ID = os.environ.get("NORTHFLANK_PROJECT_ID")

# Ollama (optional - for cloud Ollama only)
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434")
OLLAMA_ENABLED = os.environ.get("OLLAMA_ENABLED", "false").lower() == "true"

print(f"  POSTGRES_URI: {'✅ SET' if POSTGRES_URI else '❌ MISSING'}")
print(f"  USER_ID: {USER_ID or '❌ MISSING'}")
print(f"  MODEL_NAME: {MODEL_NAME}")
print(f"  OUTPUT_BASE: {OUTPUT_BASE}")
print(f"  OLLAMA_ENABLED: {OLLAMA_ENABLED}")

if not POSTGRES_URI or not USER_ID:
    print("\n❌ FATAL: Missing required environment variables")
    sys.exit(1)

# ===== Helper Functions =====

def save_adapter_to_postgres(postgres_uri: str, user_id: str, version: str, adapter_dir: str):
    """
    ✅ NEW: Save adapter files to Postgres as BYTEA
    
    Args:
        postgres_uri: Postgres connection string
        user_id: User ID
        version: Adapter version (e.g., v1, v2)
        adapter_dir: Directory containing adapter files
    
    Returns:
        bool: True if successful
    """
    try:
        print("\n📦 STEP 11: Saving adapter to Postgres...")
        
        conn = psycopg2.connect(postgres_uri)
        cursor = conn.cursor()
        
        # Create table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_data_schema.adapter_files (
                id SERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                version TEXT NOT NULL,
                filename TEXT NOT NULL,
                data BYTEA NOT NULL,
                size BIGINT NOT NULL,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(user_id, version, filename)
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_adapter_files_user_version 
            ON user_data_schema.adapter_files(user_id, version)
        """)
        
        conn.commit()
        
        # Files to save
        files_to_save = [
            'adapter_model.safetensors',
            'adapter_config.json',
            'metadata.json',
        ]
        
        total_size = 0
        saved_count = 0
        
        for filename in files_to_save:
            file_path = os.path.join(adapter_dir, filename)
            
            if not os.path.exists(file_path):
                print(f"  ⚠️  Skipping {filename} (not found)")
                continue
            
            # Read file as binary
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            file_size = len(file_data)
            total_size += file_size
            
            print(f"  📤 Saving {filename} ({file_size:,} bytes)...")
            
            # Check size limit (10 MB per file)
            if file_size > 10 * 1024 * 1024:
                print(f"     ⚠️  File too large ({file_size / 1024 / 1024:.1f} MB), skipping")
                continue
            
            # Metadata
            metadata = {
                'original_size': file_size,
                'saved_at': datetime.utcnow().isoformat() + 'Z',
                'base_model': MODEL_NAME,
            }
            
            # Insert or update
            cursor.execute("""
                INSERT INTO user_data_schema.adapter_files
                (user_id, version, filename, data, size, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, version, filename)
                DO UPDATE SET 
                    data = EXCLUDED.data,
                    size = EXCLUDED.size,
                    metadata = EXCLUDED.metadata,
                    created_at = NOW()
            """, (
                user_id,
                version,
                filename,
                psycopg2.Binary(file_data),
                file_size,
                json.dumps(metadata)
            ))
            
            saved_count += 1
            print(f"     ✅ Saved to Postgres")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"\n  ✅ Total saved: {total_size:,} bytes ({total_size / 1024 / 1024:.2f} MB)")
        print(f"  📊 Files saved: {saved_count}/{len(files_to_save)}")
        
        return saved_count > 0
        
    except Exception as e:
        print(f"  ❌ Failed to save to Postgres: {e}")
        import traceback
        traceback.print_exc()
        return False

def scale_service_to_zero():
    """Scale service to 0"""
    if not NORTHFLANK_API_TOKEN or not NORTHFLANK_PROJECT_ID:
        print("\n⚠️  Scale down skipped: Missing credentials")
        return False
    
    try:
        print("\n📊 Scaling service to 0...")
        
        response = requests.post(
            f"https://api.northflank.com/v1/projects/{NORTHFLANK_PROJECT_ID}/services/lora-training/scale",
            headers={
                'Authorization': f'Bearer {NORTHFLANK_API_TOKEN}',
                'Content-Type': 'application/json',
            },
            json={'instances': 0},
            timeout=30
        )
        
        if response.ok:
            print(f"  ✅ Scaled to 0")
            return True
        else:
            print(f"  ❌ Scale failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"  ❌ Scale error: {e}")
        return False

def update_training_status(training_id: str, status: str, error_message: str = None):
    """Update training_jobs table"""
    try:
        conn = psycopg2.connect(POSTGRES_URI)
        cursor = conn.cursor()
        
        if status == 'completed':
            cursor.execute("""
                UPDATE user_data_schema.training_jobs
                SET status = 'completed',
                    completed_at = NOW(),
                    error_message = NULL,
                    updated_at = NOW()
                WHERE job_id = %s
            """, (training_id,))
            
        elif status == 'failed':
            cursor.execute("""
                UPDATE user_data_schema.training_jobs
                SET status = 'failed',
                    completed_at = NOW(),
                    error_message = %s,
                    updated_at = NOW()
                WHERE job_id = %s
            """, (error_message, training_id))
        
        conn.commit()
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"  ⚠️  DB update failed: {e}")

def get_next_version_number(postgres_uri: str, user_id: str):
    """Calculate next version"""
    try:
        conn = psycopg2.connect(postgres_uri)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT adapter_version
            FROM user_data_schema.training_jobs
            WHERE user_id = %s
              AND status = 'completed'
            ORDER BY completed_at DESC
            LIMIT 1
        """, (user_id,))
        
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not row or not row[0]:
            return "v1"
        
        last_version = row[0]
        match = re.match(r'v(\d+)', last_version)
        
        if match:
            next_num = int(match.group(1)) + 1
            return f"v{next_num}"
        
        return "v1"
        
    except Exception as e:
        print(f"  ❌ Version calc failed: {e}")
        return "v1"

def fetch_interaction_memories(postgres_uri: str, user_id: str, limit: int = 500):
    """Fetch memories for training"""
    try:
        conn = psycopg2.connect(postgres_uri)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = """
        SELECT text, classification, ethical_scores, gentle_guidance, 
               reflection_prompt, training_weight, created_at
        FROM user_data_schema.interaction_memories
        WHERE user_id = %s
          AND approved_for_training = TRUE
        ORDER BY created_at DESC
        LIMIT %s
        """
        cursor.execute(query, (user_id, limit))
        
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        print(f"  ✅ Fetched {len(rows)} samples")
        
        return rows
        
    except Exception as e:
        print(f"  ❌ Fetch failed: {e}")
        raise

def check_ollama_health():
    """Check if Ollama is available (cloud only)"""
    if not OLLAMA_ENABLED:
        return False
    
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        return response.ok
    except:
        return False

def register_adapter_with_ollama(adapter_path: str, user_id: str, version: str):
    """Register with cloud Ollama (optional)"""
    if not OLLAMA_ENABLED:
        return False
    
    try:
        adapter_file = os.path.join(adapter_path, "adapter_model.safetensors")
        if not os.path.exists(adapter_file):
            return False
        
        model_name = f"ethical-{user_id[:8]}-{version}"
        
        modelfile = f"""FROM tinyllama
ADAPTER {adapter_file}
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 2048
"""
        
        response = requests.post(
            f"{OLLAMA_URL}/api/create",
            json={"name": model_name, "modelfile": modelfile},
            timeout=300
        )
        
        return response.ok
        
    except Exception as e:
        print(f"  ⚠️  Ollama registration failed: {e}")
        return False

# ===== MAIN EXECUTION =====
TRAINING_ID = None
training_success = False

try:
    # Test DB
    conn = psycopg2.connect(POSTGRES_URI)
    cursor = conn.cursor()
    cursor.execute("SELECT version()")
    version = cursor.fetchone()[0]
    print(f"  ✅ Connected: {version[:60]}...")
    cursor.close()
    conn.close()

    # Auto-approve memories
    conn = psycopg2.connect(POSTGRES_URI)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE user_data_schema.interaction_memories
        SET approved_for_training = TRUE
        WHERE user_id = %s 
          AND approved_for_training = FALSE
          AND classification != 'needs_support'
    """, (USER_ID,))
    approved_count = cursor.rowcount
    conn.commit()
    cursor.close()
    conn.close()
    print(f"  ✅ Auto-approved {approved_count} memories")
    
    # Get version
    NEW_VERSION = get_next_version_number(POSTGRES_URI, USER_ID)
    OUTPUT_DIR = os.path.join(OUTPUT_BASE, USER_ID, NEW_VERSION)
    TRAINING_ID = f"train-{USER_ID[:8]}-{NEW_VERSION}"
    
    print(f"\n  📊 Configuration:")
    print(f"     Version: {NEW_VERSION}")
    print(f"     Training ID: {TRAINING_ID}")
    print(f"     Output: {OUTPUT_DIR}")
    
    # Fetch data
    memories = fetch_interaction_memories(POSTGRES_URI, USER_ID, CONFIG['max_samples_per_training'])
    
    total_samples = len(memories)
    
    if total_samples < 10:
        error_msg = f"Need 10 samples (have {total_samples})"
        print(f"\n  ❌ {error_msg}")
        update_training_status(TRAINING_ID, 'failed', error_msg)
        raise Exception(error_msg)
    
    # Load model
    print("\n📋 Loading model...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    print(f"  ✅ Model loaded")
    
    gc.collect()

    # Prepare dataset
    print("\n📋 Preparing dataset...")

    def create_training_pairs(memories):
        pairs = []
        for item in memories:
            classification = item['classification']
            text = item['text']
            
            if classification == 'growth_memory':
                instruction = 'Respond supportively to encourage growth'
                output = text
                weight = 1.5
            elif classification == 'challenge_memory':
                instruction = 'Respond with compassion to a challenge'
                output = item.get('gentle_guidance') or f"I understand. {text}"
                weight = 2.0
            elif classification == 'wisdom_moment':
                instruction = 'Share wisdom'
                output = text
                weight = 2.5
            else:
                instruction = 'Respond naturally'
                output = text
                weight = 1.0
            
            pairs.append({
                'instruction': instruction,
                'input': text,
                'output': output,
                'weight': weight,
            })
        
        return pairs

    all_pairs = create_training_pairs(memories)
    random.shuffle(all_pairs)
    
    print(f"  ✅ {len(all_pairs)} pairs")
    
    texts = [
        f"{p['instruction']}\n\n{p['input']}\n\n{p['output']}{tokenizer.eos_token}"
        for p in all_pairs
    ]
    
    dataset = Dataset.from_dict({"text": texts})
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=CONFIG["max_length"],
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    print(f"  ✅ Dataset ready")

    gc.collect()

    # Configure LoRA
    print("\n📋 Configuring LoRA...")
    lora_config = LoraConfig(
        r=CONFIG['r'],
        lora_alpha=CONFIG['lora_alpha'],
        target_modules=CONFIG['target_modules'],
        lora_dropout=CONFIG['lora_dropout'],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, lora_config)
    print(f"  ✅ LoRA configured")

    gc.collect()

    # Training
    print("\n📋 Training...")
    print(f"  Epochs: {CONFIG['num_epochs']}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=CONFIG['num_epochs'],
        per_device_train_batch_size=CONFIG['batch_size'],
        gradient_accumulation_steps=CONFIG['gradient_accumulation_steps'],
        learning_rate=CONFIG['learning_rate'],
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        remove_unused_columns=False,
        fp16=False,
        optim="adamw_torch",
        warmup_steps=10,
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    result = trainer.train()
    
    print(f"\n  ✅ Training completed!")
    print(f"     Loss: {result.training_loss:.4f}")

    # Save model to /tmp
    print("\n📋 STEP 10: Saving model...")
    
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"  ✅ Saved to {OUTPUT_DIR}")
    
    # Save metadata
    metadata = {
        "user_id": USER_ID,
        "adapter_version": NEW_VERSION,
        "base_model": MODEL_NAME,
        "total_samples": total_samples,
        "loss": float(result.training_loss),
        "trained_at": datetime.utcnow().isoformat() + "Z",
    }
    
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # ===== ✅ STEP 11: SAVE TO POSTGRES =====
    save_success = save_adapter_to_postgres(
        POSTGRES_URI, 
        USER_ID, 
        NEW_VERSION, 
        OUTPUT_DIR
    )

    if not save_success:
        print("⚠️  Warning: Failed to save to Postgres")
    
    # ===== STEP 12: Ollama (optional, cloud only) =====
    ollama_available = check_ollama_health()
    ollama_success = False
    
    if ollama_available:
        print("\n📋 STEP 12: Registering with Ollama (cloud)...")
        ollama_success = register_adapter_with_ollama(OUTPUT_DIR, USER_ID, NEW_VERSION)
        if ollama_success:
            print("  ✅ Registered with cloud Ollama")
        else:
            print("  ⚠️  Cloud Ollama registration failed")
    else:
        print("\n📋 STEP 12: Ollama not available (skipped)")
    
    # Update DB
    print("\n📋 Updating database...")
    update_training_status(TRAINING_ID, 'completed')

    training_success = True

    print("\n" + "="*60)
    print("✅✅✅ TRAINING COMPLETED")
    print("="*60)
    print(f"📊 Summary:")
    print(f"   Training ID: {TRAINING_ID}")
    print(f"   Version: {NEW_VERSION}")
    print(f"   Samples: {total_samples}")
    print(f"   Loss: {result.training_loss:.4f}")
    print(f"   Postgres: {'✅ Saved' if save_success else '❌ Failed'}")
    print(f"   Cloud Ollama: {'✅ Registered' if ollama_success else '⚠️  Skipped'}")
    print("="*60)

except Exception as e:
    print(f"\n❌ TRAINING FAILED: {e}")
    import traceback
    traceback.print_exc()
    
    if TRAINING_ID:
        update_training_status(TRAINING_ID, 'failed', str(e))
    
    training_success = False

finally:
    print("\n🔄 Cleanup...")
    time.sleep(3)
    scale_service_to_zero()
    
    print(f"\n{'✅ COMPLETED' if training_success else '❌ FAILED'}")
    sys.exit(0 if training_success else 1)
