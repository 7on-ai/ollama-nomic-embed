#!/usr/bin/env python3
"""
LoRA Training Pipeline - INCREMENTAL TRAINING (COMPLETE FIXED)
✅ เทรนแบบสะสมประสบการณ์
✅ Fixed: Version management + Previous adapter detection
✅ Debug output เต็มรูปแบบ
✅ Auto scale down to 0 after completion (FIXED)
"""

import os
import sys
import json
import random
import gc
import requests
import time
from datetime import datetime
from pathlib import Path

# ===== Startup Info =====
print("="*60)
print("🚀 LoRA Incremental Training (COMPLETE FIXED)")
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
    "growth_weight": 0.30,
    "challenge_weight": 0.25,
    "wisdom_weight": 0.25,
    "neutral_weight": 0.15,
    "support_weight": 0.05,
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

print(f"  POSTGRES_URI: {'✅ SET' if POSTGRES_URI else '❌ MISSING'}")
print(f"  USER_ID: {USER_ID or '❌ MISSING'}")
print(f"  MODEL_NAME: {MODEL_NAME}")
print(f"  OUTPUT_BASE: {OUTPUT_BASE}")
print(f"  NORTHFLANK_API_TOKEN: {'✅ SET' if NORTHFLANK_API_TOKEN else '⚠️  MISSING'}")
print(f"  NORTHFLANK_PROJECT_ID: {'✅ SET' if NORTHFLANK_PROJECT_ID else '⚠️  MISSING'}")

if not POSTGRES_URI or not USER_ID:
    print("\n❌ FATAL: Missing required environment variables")
    sys.exit(1)

# ===== Scale Down Function - COMPLETE FIXED =====
def scale_service_to_zero():
    """Scale Northflank service to 0 replicas - COMPLETE FIXED VERSION"""
    if not NORTHFLANK_API_TOKEN or not NORTHFLANK_PROJECT_ID:
        print("\n⚠️  Scale down skipped: Missing credentials")
        print(f"     NORTHFLANK_API_TOKEN: {'SET' if NORTHFLANK_API_TOKEN else 'MISSING'}")
        print(f"     NORTHFLANK_PROJECT_ID: {'SET' if NORTHFLANK_PROJECT_ID else 'MISSING'}")
        return False
    
    try:
        print("\n" + "="*60)
        print("📊 SCALING SERVICE TO 0")
        print("="*60)
        
        service_id = "lora-training"
        url = f"https://api.northflank.com/v1/projects/{NORTHFLANK_PROJECT_ID}/services/{service_id}/scale"
        
        print(f"  Project ID: {NORTHFLANK_PROJECT_ID}")
        print(f"  Service ID: {service_id}")
        print(f"  URL: {url}")
        print(f"  Token: {NORTHFLANK_API_TOKEN[:20]}...")
        
        headers = {
            'Authorization': f'Bearer {NORTHFLANK_API_TOKEN}',
            'Content-Type': 'application/json',
        }
        
        payload = {'instances': 0}
        
        print(f"\n  🔄 Sending scale request...")
        print(f"     Payload: {json.dumps(payload)}")
        
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"\n  📡 Response:")
        print(f"     Status: {response.status_code}")
        print(f"     Headers: {dict(response.headers)}")
        
        if response.ok:
            print(f"\n  ✅ SUCCESS: Service scaled to 0 replicas")
            try:
                response_data = response.json()
                print(f"     Response: {json.dumps(response_data, indent=2)}")
            except:
                print(f"     Response text: {response.text}")
            return True
        else:
            print(f"\n  ❌ PRIMARY ENDPOINT FAILED: {response.status_code}")
            print(f"     Error: {response.text}")
            
            # ✅ Try alternative PATCH endpoint
            print(f"\n  🔄 Trying alternative PATCH endpoint...")
            alt_url = f"https://api.northflank.com/v1/projects/{NORTHFLANK_PROJECT_ID}/services/{service_id}"
            
            alt_response = requests.patch(
                alt_url,
                headers=headers,
                json={'spec': {'replicas': 0}},
                timeout=30
            )
            
            print(f"     Status: {alt_response.status_code}")
            
            if alt_response.ok:
                print(f"  ✅ ALTERNATIVE ENDPOINT SUCCESS!")
                try:
                    alt_data = alt_response.json()
                    print(f"     Response: {json.dumps(alt_data, indent=2)}")
                except:
                    print(f"     Response text: {alt_response.text}")
                return True
            else:
                print(f"  ❌ ALTERNATIVE ALSO FAILED: {alt_response.status_code}")
                print(f"     Error: {alt_response.text}")
                return False
            
    except requests.exceptions.Timeout:
        print(f"\n  ⏱️  TIMEOUT: Request took too long")
        return False
    except requests.exceptions.RequestException as e:
        print(f"\n  ❌ REQUEST ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n  ❌ UNEXPECTED ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        print("="*60)

# ===== DB Helper Functions =====
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
    """Calculate next version number"""
    try:
        conn = psycopg2.connect(postgres_uri)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT adapter_version
            FROM user_data_schema.training_jobs
            WHERE user_id = %s
              AND status = 'completed'
              AND adapter_version LIKE 'v%'
            ORDER BY completed_at DESC
            LIMIT 1
        """, (user_id,))
        
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if row and row[0]:
            try:
                last_num = int(row[0].replace('v', ''))
                return f"v{last_num + 1}"
            except:
                return "v1"
        
        return "v1"
        
    except Exception as e:
        print(f"  ⚠️  Version calc failed: {e}")
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

def fetch_ethical_profile(postgres_uri: str, user_id: str):
    """Fetch ethical profile"""
    try:
        conn = psycopg2.connect(postgres_uri)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = """
        SELECT growth_stage, self_awareness, emotional_regulation, compassion,
               integrity, growth_mindset, wisdom, transcendence,
               total_interactions, breakthrough_moments
        FROM user_data_schema.ethical_profiles
        WHERE user_id = %s
        """
        cursor.execute(query, (user_id,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        
        return dict(row) if row else None
    except Exception as e:
        print(f"  ⚠️  No ethical profile: {e}")
        return None

# ===== MAIN EXECUTION =====
TRAINING_ID = None
training_success = False

try:
    # ===== STEP 3: Test DB =====
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

    # ===== STEP 4: Check Previous Training =====
    print("\n📋 STEP 4: Checking for previous training...")
    print(f"  OUTPUT_BASE: {OUTPUT_BASE}")
    print(f"  USER_ID: {USER_ID}")

    last_training = get_last_completed_training(POSTGRES_URI, USER_ID)

    if last_training:
        print(f"\n  ✅ Found previous training:")
        print(f"     Version: {last_training['adapter_version']}")
        print(f"     Completed: {last_training['completed_at']}")
        
        previous_adapter_path = find_adapter_by_version(
            OUTPUT_BASE, USER_ID, last_training['adapter_version']
        )
        
        if previous_adapter_path:
            print(f"     Adapter: {previous_adapter_path}")
            IS_INCREMENTAL = True
            LAST_TRAINED_AT = last_training['completed_at']
            PREVIOUS_VERSION = last_training['adapter_version']
            NEW_VERSION = get_next_version_number(POSTGRES_URI, USER_ID)
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

    ADAPTER_VERSION = NEW_VERSION
    OUTPUT_DIR = os.path.join(OUTPUT_BASE, USER_ID, ADAPTER_VERSION)
    TRAINING_ID = f"train-{USER_ID[:8]}-{ADAPTER_VERSION}"

    print(f"\n  📊 Configuration:")
    print(f"     Mode: {'INCREMENTAL' if IS_INCREMENTAL else 'FULL'}")
    print(f"     New Version: {NEW_VERSION}")
    print(f"     Training ID: {TRAINING_ID}")

    # ===== STEP 5: Fetch Data =====
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
    
    ethical_profile = fetch_ethical_profile(POSTGRES_URI, USER_ID)
    
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
    
    # ===== STEP 6: Load Model =====
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
    
    if IS_INCREMENTAL and previous_adapter_path:
        print(f"\n  🔄 Loading previous adapter...")
        try:
            model = PeftModel.from_pretrained(base_model, previous_adapter_path)
            print(f"  ✅ Incremental mode")
        except Exception as e:
            print(f"  ⚠️  Adapter load failed: {e}")
            IS_INCREMENTAL = False
            model = base_model
    else:
        model = base_model

    gc.collect()

    # ===== STEP 7: Prepare Dataset =====
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
    
    pairs_by_class = {}
    for pair in all_pairs:
        cls = pair['classification']
        if cls not in pairs_by_class:
            pairs_by_class[cls] = []
        pairs_by_class[cls].append(pair)
    
    sampled = []
    total = len(all_pairs)
    
    print(f"\n  📊 Sampling:")
    for cls, weight_key in [
        ('growth_memory', 'growth_weight'),
        ('challenge_memory', 'challenge_weight'),
        ('wisdom_moment', 'wisdom_weight'),
        ('neutral_interaction', 'neutral_weight'),
        ('needs_support', 'support_weight'),
    ]:
        if cls in pairs_by_class:
            target = int(total * CONFIG[weight_key])
            available = pairs_by_class[cls]
            sample_size = min(target, len(available))
            sampled.extend(random.sample(available, sample_size))
            print(f"     {cls}: {sample_size}/{len(available)}")
    
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

    # ===== STEP 8: Configure LoRA =====
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
        print("\n📋 STEP 8: Using existing adapter")
        model.print_trainable_parameters()

    gc.collect()

    # ===== STEP 9: Training =====
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

    # ===== STEP 10: Save =====
    print("\n📋 STEP 10: Saving model...")

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"  ✅ Model saved: {OUTPUT_DIR}")
    
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
        "ethical_profile": ethical_profile,
        "config": CONFIG,
        "metrics": {"loss": float(result.training_loss)},
        "trained_at": datetime.utcnow().isoformat() + "Z",
    }
    
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  ✅ Metadata saved")

    # ===== STEP 11: Update DB =====
    print("\n📋 STEP 11: Updating database...")
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
    # ✅ ALWAYS try to scale down
    print("\n" + "="*60)
    print("🔄 CLEANUP: Attempting to scale down service...")
    print("="*60)
    
    # Wait for logs to flush
    time.sleep(3)
    
    # Try to scale down
    scale_success = scale_service_to_zero()
    
    if scale_success:
        print("\n✅ Service scaled down successfully")
    else:
        print("\n⚠️  Service scale down failed")
        print("    Manual intervention may be required")
        print("    Go to Northflank Dashboard → lora-training → Scale to 0")
    
    print("\n" + "="*60)
    print(f"{'✅ PROCESS COMPLETED' if training_success else '❌ PROCESS FAILED'}")
    print("="*60)
    
    # Exit with appropriate code
    sys.exit(0 if training_success else 1)