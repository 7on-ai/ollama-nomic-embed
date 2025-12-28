#!/usr/bin/env python3
"""
LoRA Training Pipeline - COMPLETE WITH OLLAMA INTEGRATION
✅ Fixed: Proper gradient handling for incremental training
✅ Fixed: Ensure all LoRA parameters require grad
✅ Fixed: Version increment logic
✅ NEW: Ollama Modelfile registration after training
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
print("🚀 LoRA Incremental Training + Ollama Integration")
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
    from peft import LoraConfig, get_peft_model, PeftModel, TaskType, prepare_model_for_kbit_training
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
OUTPUT_BASE = os.environ.get("OUTPUT_PATH", "/workspace/adapters")
NORTHFLANK_API_TOKEN = os.environ.get("NORTHFLANK_API_TOKEN")
NORTHFLANK_PROJECT_ID = os.environ.get("NORTHFLANK_PROJECT_ID")

# ✅ NEW: Ollama configuration
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434")
OLLAMA_ENABLED = os.environ.get("OLLAMA_ENABLED", "true").lower() == "true"

print(f"  POSTGRES_URI: {'✅ SET' if POSTGRES_URI else '❌ MISSING'}")
print(f"  USER_ID: {USER_ID or '❌ MISSING'}")
print(f"  MODEL_NAME: {MODEL_NAME}")
print(f"  OUTPUT_BASE: {OUTPUT_BASE}")
print(f"  OLLAMA_URL: {OLLAMA_URL}")
print(f"  OLLAMA_ENABLED: {OLLAMA_ENABLED}")

if not POSTGRES_URI or not USER_ID:
    print("\n❌ FATAL: Missing required environment variables")
    sys.exit(1)

# ===== Helper Functions =====
def scale_service_to_zero():
    """Scale service to 0"""
    if not NORTHFLANK_API_TOKEN or not NORTHFLANK_PROJECT_ID:
        print("\n⚠️  Scale down skipped: Missing credentials")
        return False
    
    try:
        print("\n" + "="*60)
        print("📊 SCALING SERVICE TO 0")
        print("="*60)
        
        service_id = "lora-training"
        url = f"https://api.northflank.com/v1/projects/{NORTHFLANK_PROJECT_ID}/services/{service_id}/scale"
        
        headers = {
            'Authorization': f'Bearer {NORTHFLANK_API_TOKEN}',
            'Content-Type': 'application/json',
        }
        
        payload = {'instances': 0}
        
        print(f"  🔄 Sending scale request...")
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.ok:
            print(f"  ✅ SUCCESS: Service scaled to 0 replicas")
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
        print(f"\n📊 Updating DB: job_id={training_id}, status={status}")
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
        print(f"  ✅ DB updated successfully")
        
    except Exception as e:
        print(f"  ⚠️  DB update failed: {e}")

def get_last_completed_training(postgres_uri: str, user_id: str):
    """Get last successful training"""
    try:
        conn = psycopg2.connect(postgres_uri)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = """
        SELECT job_id, adapter_version, completed_at, created_at
        FROM user_data_schema.training_jobs
        WHERE user_id = %s
          AND status = 'completed'
        ORDER BY completed_at DESC
        LIMIT 1
        """
        cursor.execute(query, (user_id,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        
        return dict(row) if row else None
        
    except Exception as e:
        print(f"  ⚠️  Query failed: {e}")
        return None

def find_adapter_by_version(output_base: str, user_id: str, version: str):
    """Find adapter by version"""
    adapter_path = os.path.join(output_base, user_id, version)
    adapter_file = os.path.join(adapter_path, 'adapter_model.safetensors')
    
    if os.path.exists(adapter_file):
        return adapter_path
    return None

def get_next_version_number(postgres_uri: str, user_id: str):
    """✅ FIXED: Calculate next version number correctly"""
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
            print(f"  📌 No previous version found -> v1")
            return "v1"
        
        last_version = row[0]
        print(f"  📌 Last version: {last_version}")
        
        # ✅ FIX: Extract number correctly
        match = re.match(r'v(\d+)', last_version)
        
        if match:
            last_num = int(match.group(1))
            next_version = f"v{last_num + 1}"
            print(f"  ✅ Next version: {next_version}")
            return next_version
        else:
            print(f"  ⚠️  Invalid format '{last_version}' -> defaulting to v1")
            return "v1"
        
    except Exception as e:
        print(f"  ❌ Version calc failed: {e}")
        import traceback
        traceback.print_exc()
        return "v1"

def fetch_interaction_memories(postgres_uri: str, user_id: str, last_trained_at=None, limit: int = 500):
    """Fetch memories"""
    try:
        conn = psycopg2.connect(postgres_uri)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        if last_trained_at:
            query = """
            SELECT text, classification, ethical_scores, gentle_guidance, 
                   reflection_prompt, training_weight, created_at
            FROM user_data_schema.interaction_memories
            WHERE user_id = %s
              AND approved_for_training = TRUE
              AND created_at > %s
            ORDER BY created_at DESC
            LIMIT %s
            """
            cursor.execute(query, (user_id, last_trained_at, limit))
            mode = "INCREMENTAL"
        else:
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
            mode = "FULL"
        
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        print(f"  ✅ Fetched {len(rows)} samples (mode: {mode})")
        if last_trained_at:
            print(f"     After: {last_trained_at}")
        
        return rows, mode
        
    except Exception as e:
        print(f"  ❌ Fetch failed: {e}")
        raise

# ===== ✅ NEW: Ollama Integration Functions =====

def check_ollama_health():
    """Check if Ollama service is available"""
    if not OLLAMA_ENABLED:
        print("\n⚠️  Ollama integration disabled")
        return False
    
    try:
        print(f"\n🔍 Checking Ollama at {OLLAMA_URL}...")
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        
        if response.ok:
            print("  ✅ Ollama is online")
            return True
        else:
            print(f"  ⚠️  Ollama returned status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"  ⚠️  Ollama not reachable: {e}")
        return False

def register_adapter_with_ollama(adapter_path: str, user_id: str, version: str, base_model: str):
    """
    Register trained adapter with Ollama using Modelfile API
    
    Args:
        adapter_path: Full path to adapter directory (e.g., /workspace/adapters/user123/v1)
        user_id: User ID (first 8 chars used in model name)
        version: Version (e.g., v1, v2)
        base_model: Base model name (e.g., TinyLlama/TinyLlama-1.1B-Chat-v1.0)
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not OLLAMA_ENABLED:
        print("\n⚠️  Ollama registration skipped (disabled)")
        return False
    
    try:
        print("\n" + "="*60)
        print("🤖 REGISTERING ADAPTER WITH OLLAMA")
        print("="*60)
        
        # Validate adapter file exists
        adapter_file = os.path.join(adapter_path, "adapter_model.safetensors")
        if not os.path.exists(adapter_file):
            print(f"  ❌ Adapter file not found: {adapter_file}")
            return False
        
        print(f"  ✅ Adapter file found: {adapter_file}")
        print(f"  📊 File size: {os.path.getsize(adapter_file):,} bytes")
        
        # Create model name (consistent with N8N)
        model_name = f"ethical-{user_id[:8]}-{version}"
        print(f"  📝 Model name: {model_name}")
        
        # Create Modelfile content
        modelfile = f"""FROM tinyllama
ADAPTER {adapter_file}
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 2048
"""
        
        print(f"  📄 Modelfile content:")
        print("  " + "-"*56)
        for line in modelfile.strip().split('\n'):
            print(f"  {line}")
        print("  " + "-"*56)
        
        # Call Ollama API
        print(f"\n  🔄 Calling Ollama API: POST {OLLAMA_URL}/api/create")
        
        response = requests.post(
            f"{OLLAMA_URL}/api/create",
            json={
                "name": model_name,
                "modelfile": modelfile,
            },
            timeout=300  # 5 minutes for model creation
        )
        
        if response.ok:
            print(f"  ✅ SUCCESS: Model registered as '{model_name}'")
            print(f"  🎯 N8N Ollama node can now use: {model_name}")
            
            # Verify registration
            verify_response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
            if verify_response.ok:
                models = verify_response.json().get('models', [])
                if any(m.get('name', '').startswith(model_name) for m in models):
                    print(f"  ✅ Verified: Model appears in Ollama model list")
                else:
                    print(f"  ⚠️  Warning: Model not found in list (may need refresh)")
            
            return True
        else:
            error_text = response.text
            print(f"  ❌ Ollama API error: {response.status_code}")
            print(f"  📄 Response: {error_text}")
            return False
            
    except requests.Timeout:
        print(f"  ❌ Timeout: Model creation took too long")
        return False
    except Exception as e:
        print(f"  ❌ Registration error: {e}")
        import traceback
        traceback.print_exc()
        return False

def cleanup_old_ollama_models(user_id: str, keep_versions: int = 3):
    """
    Optional: Remove old model versions from Ollama to save space
    
    Args:
        user_id: User ID
        keep_versions: Number of recent versions to keep
    """
    if not OLLAMA_ENABLED:
        return
    
    try:
        print(f"\n🧹 Cleaning up old Ollama models (keep {keep_versions} versions)...")
        
        # Get all models
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        if not response.ok:
            return
        
        models = response.json().get('models', [])
        
        # Filter user's models
        user_prefix = f"ethical-{user_id[:8]}-"
        user_models = [m for m in models if m.get('name', '').startswith(user_prefix)]
        
        print(f"  📊 Found {len(user_models)} models for this user")
        
        if len(user_models) <= keep_versions:
            print(f"  ✅ No cleanup needed")
            return
        
        # Sort by name (assuming v1, v2, v3... format)
        user_models.sort(key=lambda m: m.get('name', ''), reverse=True)
        
        # Delete old versions
        for model in user_models[keep_versions:]:
            model_name = model.get('name', '')
            print(f"  🗑️  Deleting old model: {model_name}")
            
            delete_response = requests.delete(
                f"{OLLAMA_URL}/api/delete",
                json={"name": model_name},
                timeout=30
            )
            
            if delete_response.ok:
                print(f"     ✅ Deleted")
            else:
                print(f"     ⚠️  Failed to delete")
        
    except Exception as e:
        print(f"  ⚠️  Cleanup error: {e}")

# ===== MAIN EXECUTION =====
TRAINING_ID = None
training_success = False

try:
    # Test DB connection
    print("\n📋 STEP 3: Testing database connection...")
    try:
        conn = psycopg2.connect(POSTGRES_URI)
        cursor = conn.cursor()
        cursor.execute("SELECT version()")
        version = cursor.fetchone()[0]
        print(f"  ✅ Connected: {version[:60]}...")
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"  ❌ Database connection failed: {e}")
        sys.exit(1)

    # ✅ NEW: Check Ollama health
    ollama_available = check_ollama_health()

    # Check previous training
    print("\n📋 STEP 4: Checking for previous training...")
    last_training = get_last_completed_training(POSTGRES_URI, USER_ID)

    if last_training:
        print(f"\n  ✅ Found previous training:")
        print(f"     Job ID: {last_training.get('job_id', 'N/A')}")
        print(f"     Version: {last_training['adapter_version']}")
        print(f"     Completed: {last_training['completed_at']}")
        
        previous_adapter_path = find_adapter_by_version(
            OUTPUT_BASE, USER_ID, last_training['adapter_version']
        )
        
        if previous_adapter_path:
            print(f"     Adapter Path: {previous_adapter_path}")
            IS_INCREMENTAL = True
            LAST_TRAINED_AT = last_training['completed_at']
            PREVIOUS_VERSION = last_training['adapter_version']
            NEW_VERSION = get_next_version_number(POSTGRES_URI, USER_ID)
            
            # Validate version increment
            if NEW_VERSION == PREVIOUS_VERSION:
                print(f"     ⚠️  WARNING: Version didn't increment!")
                match = re.match(r'v(\d+)', PREVIOUS_VERSION)
                if match:
                    num = int(match.group(1))
                    NEW_VERSION = f"v{num + 1}"
                    print(f"     ✅ Forced to: {NEW_VERSION}")
        else:
            print(f"     ⚠️  Adapter files not found")
            IS_INCREMENTAL = False
            LAST_TRAINED_AT = None
            previous_adapter_path = None
            PREVIOUS_VERSION = None
            NEW_VERSION = "v1"
    else:
        print(f"\n  📝 First training")
        IS_INCREMENTAL = False
        LAST_TRAINED_AT = None
        previous_adapter_path = None
        PREVIOUS_VERSION = None
        NEW_VERSION = "v1"

    # Final configuration
    ADAPTER_VERSION = NEW_VERSION
    OUTPUT_DIR = os.path.join(OUTPUT_BASE, USER_ID, ADAPTER_VERSION)
    TRAINING_ID = f"train-{USER_ID[:8]}-{ADAPTER_VERSION}"

    print(f"\n  📊 Final Configuration:")
    print(f"     Mode: {'INCREMENTAL' if IS_INCREMENTAL else 'FULL'}")
    print(f"     Previous Version: {PREVIOUS_VERSION or 'None'}")
    print(f"     New Version: {NEW_VERSION}")
    print(f"     Adapter Version: {ADAPTER_VERSION}")
    print(f"     Training ID: {TRAINING_ID}")
    print(f"     Output Dir: {OUTPUT_DIR}")
    print("="*60)

    # Fetch data
    print("\n📋 STEP 5: Fetching training data...")
    
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
    
    memories, fetch_mode = fetch_interaction_memories(
        POSTGRES_URI, USER_ID, 
        last_trained_at=LAST_TRAINED_AT,
        limit=CONFIG['max_samples_per_training']
    )
    
    total_samples = len(memories)
    print(f"\n  📊 Data: {total_samples} samples")
    
    by_class = {}
    for mem in memories:
        cls = mem['classification']
        by_class[cls] = by_class.get(cls, 0) + 1
    
    for cls, count in by_class.items():
        print(f"     {cls}: {count}")
    
    min_required = CONFIG['min_samples_new'] if IS_INCREMENTAL else 10
    if total_samples < min_required:
        error_msg = f"Need {min_required} samples (have {total_samples})"
        print(f"\n  ❌ {error_msg}")
        update_training_status(TRAINING_ID, 'failed', error_msg)
        raise Exception(error_msg)
    
    # Load model
    print("\n📋 STEP 6: Loading model...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  ✅ Tokenizer loaded")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    print(f"  ✅ Base model loaded")
    
    # Load previous adapter if incremental
    if IS_INCREMENTAL and previous_adapter_path:
        print(f"\n  🔄 Loading previous adapter for incremental training...")
        try:
            model = PeftModel.from_pretrained(base_model, previous_adapter_path)
            
            # Ensure all LoRA parameters require grad
            print(f"  🔧 Ensuring all LoRA parameters require gradients...")
            for name, param in model.named_parameters():
                if 'lora_' in name:
                    param.requires_grad = True
            
            print(f"  ✅ Incremental mode ready")
            
        except Exception as e:
            print(f"  ⚠️  Adapter load failed: {e}")
            print(f"  🔄 Falling back to FULL training")
            IS_INCREMENTAL = False
            model = base_model
    else:
        model = base_model

    gc.collect()

    # Prepare dataset
    print("\n📋 STEP 7: Preparing dataset...")

    def create_ethical_training_pairs(memories):
        pairs = []
        for item in memories:
            classification = item['classification']
            text = item['text']
            
            if classification == 'growth_memory':
                instruction = 'Respond supportively to encourage personal growth'
                output = text
                weight = 1.5
            elif classification == 'challenge_memory':
                instruction = 'Respond with compassion and support to a challenge'
                output = item.get('gentle_guidance') or f"I understand this is challenging. {text}"
                weight = 2.0
            elif classification == 'wisdom_moment':
                instruction = 'Share wisdom and deeper insight'
                reflection = item.get('reflection_prompt', '')
                output = f"{text}\n\n💭 {reflection}" if reflection else text
                weight = 2.5
            elif classification == 'neutral_interaction':
                instruction = 'Respond naturally to everyday conversation'
                output = text
                weight = 0.8
            elif classification == 'needs_support':
                instruction = 'Provide caring support with empathy'
                output = item.get('gentle_guidance') or "I care about you."
                weight = 1.0
            else:
                instruction = 'Respond helpfully'
                output = text
                weight = 1.0
            
            pairs.append({
                'instruction': instruction,
                'input': text,
                'output': output,
                'weight': weight,
                'classification': classification,
            })
        
        return pairs

    all_pairs = create_ethical_training_pairs(memories)
    
    # Use all pairs
    sampled = all_pairs
    random.shuffle(sampled)
    
    print(f"\n  ✅ {len(sampled)} training pairs")
    
    texts = [
        f"{p['instruction']}\n\n{p['input']}\n\n{p['output']}{tokenizer.eos_token}"
        for p in sampled
    ]
    
    dataset = Dataset.from_dict({"text": texts})
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=CONFIG["max_length"],
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )
    print(f"  ✅ Dataset ready: {len(tokenized_dataset)} samples")

    gc.collect()

    # Configure LoRA
    if not IS_INCREMENTAL:
        print("\n📋 STEP 8: Configuring LoRA...")
        lora_config = LoraConfig(
            r=CONFIG['r'],
            lora_alpha=CONFIG['lora_alpha'],
            target_modules=CONFIG['target_modules'],
            lora_dropout=CONFIG['lora_dropout'],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        print(f"  ✅ LoRA configured")
        model.print_trainable_parameters()
    else:
        print("\n📋 STEP 8: Using existing adapter (incremental)")
        model.print_trainable_parameters()

    gc.collect()

    # Training
    print("\n📋 STEP 9: Starting training...")
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
    
    print(f"\n  🏋️  Training started: {datetime.utcnow().isoformat()}Z")
    result = trainer.train()
    
    print(f"\n  ✅ Training completed!")
    print(f"     Loss: {result.training_loss:.4f}")

    # Save model
    print("\n📋 STEP 10: Saving model...")
    print(f"  📁 Output directory: {OUTPUT_DIR}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"  💾 Saving adapter model...")
    model.save_pretrained(OUTPUT_DIR)
    print(f"  ✅ Adapter saved")
    
    print(f"  💾 Saving tokenizer...")
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"  ✅ Tokenizer saved")
    
    # Verify files
    print(f"  🔍 Verifying saved files...")
    saved_files = os.listdir(OUTPUT_DIR)
    print(f"  📄 Files in {OUTPUT_DIR}:")
    for f in saved_files:
        file_path = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(file_path)
        print(f"     - {f} ({size:,} bytes)")
    
    # Save metadata
    metadata = {
        "user_id": USER_ID,
        "adapter_version": ADAPTER_VERSION,
        "training_id": TRAINING_ID,
        "base_model": MODEL_NAME,
        "training_mode": "incremental" if IS_INCREMENTAL else "full",
        "previous_version": PREVIOUS_VERSION,
        "data_stats": {
            "total_samples": total_samples,
            "trained_samples": len(sampled),
            "by_classification": by_class,
            "last_trained_at": str(LAST_TRAINED_AT) if LAST_TRAINED_AT else None,
        },
        "config": CONFIG,
        "metrics": {"loss": float(result.training_loss)},
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "ollama_registered": False,  # Will be updated below
    }
    
    metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  ✅ Metadata saved: {metadata_path}")

    # ===== ✅ NEW: STEP 11 - Register with Ollama =====
    print("\n📋 STEP 11: Registering with Ollama...")
    
    ollama_success = False
    if ollama_available:
        ollama_success = register_adapter_with_ollama(
            adapter_path=OUTPUT_DIR,
            user_id=USER_ID,
            version=ADAPTER_VERSION,
            base_model=MODEL_NAME
        )
        
        if ollama_success:
            # Update metadata
            metadata["ollama_registered"] = True
            metadata["ollama_model_name"] = f"ethical-{USER_ID[:8]}-{ADAPTER_VERSION}"
            metadata["ollama_registered_at"] = datetime.utcnow().isoformat() + "Z"
            
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            print(f"  ✅ Metadata updated with Ollama info")
            
            # Optional: Cleanup old models
            cleanup_old_ollama_models(USER_ID, keep_versions=3)
        else:
            print(f"  ⚠️  Ollama registration failed (training still successful)")
    else:
        print(f"  ⚠️  Ollama not available (skipping registration)")

    # Update DB
    print("\n📋 STEP 12: Updating database...")
    update_training_status(TRAINING_ID, 'completed')

    training_success = True

    print("\n" + "="*60)
    print("✅✅✅ TRAINING COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"📊 Summary:")
    print(f"   Training ID: {TRAINING_ID}")
    print(f"   Mode: {'INCREMENTAL' if IS_INCREMENTAL else 'FULL'}")
    print(f"   Version: {ADAPTER_VERSION}")
    print(f"   Samples: {len(sampled)}/{total_samples}")
    print(f"   Loss: {result.training_loss:.4f}")
    print(f"   Output: {OUTPUT_DIR}")
    print(f"   Ollama: {'✅ Registered' if ollama_success else '⚠️  Not registered'}")
    if ollama_success:
        print(f"   Model Name: ethical-{USER_ID[:8]}-{ADAPTER_VERSION}")
    print("="*60)

except Exception as e:
    print(f"\n" + "="*60)
    print(f"❌ TRAINING FAILED")
    print("="*60)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    print("="*60)
    
    if TRAINING_ID:
        update_training_status(TRAINING_ID, 'failed', str(e))
    
    training_success = False

finally:
    # Always try to scale down
    print("\n" + "="*60)
    print("🔄 CLEANUP: Attempting to scale down service...")
    print("="*60)
    
    time.sleep(3)
    
    scale_success = scale_service_to_zero()
    
    if scale_success:
        print("\n✅ Service scaled down successfully")
    else:
        print("\n⚠️  Service scale down failed")
    
    print("\n" + "="*60)
    print(f"{'✅ PROCESS COMPLETED' if training_success else '❌ PROCESS FAILED'}")
    print("="*60)
    
    sys.exit(0 if training_success else 1)
    
