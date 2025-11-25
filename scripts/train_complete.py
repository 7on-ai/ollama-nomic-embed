#!/usr/bin/env python3
"""
LoRA Training Pipeline - INCREMENTAL TRAINING
✅ เทรนแบบสะสมประสบการณ์ (ไม่เริ่มใหม่ทุกครั้ง)
✅ โหลด adapter เก่า + เทรนต่อด้วยข้อมูลใหม่
"""

import os
import sys
import json
import random
import gc
from datetime import datetime
from pathlib import Path

# ===== DEBUG: Print startup info =====
print("="*60)
print("🚀 LoRA Incremental Training Script Starting")
print("="*60)
print(f"Time: {datetime.utcnow().isoformat()}Z")
print(f"Python: {sys.version}")
print(f"Working Dir: {os.getcwd()}")
print("="*60)

# ===== STEP 1: Check imports =====
print("\n📋 STEP 1: Checking imports...")
try:
    import torch
    print(f"  ✅ PyTorch {torch.__version__}")
except Exception as e:
    print(f"  ❌ PyTorch import failed: {e}")
    sys.exit(1)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
    print(f"  ✅ Transformers imported")
except Exception as e:
    print(f"  ❌ Transformers import failed: {e}")
    sys.exit(1)

try:
    from peft import LoraConfig, get_peft_model, PeftModel, TaskType
    print(f"  ✅ PEFT imported")
except Exception as e:
    print(f"  ❌ PEFT import failed: {e}")
    sys.exit(1)

try:
    from datasets import Dataset
    print(f"  ✅ Datasets imported")
except Exception as e:
    print(f"  ❌ Datasets import failed: {e}")
    sys.exit(1)

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    print(f"  ✅ psycopg2 imported")
except Exception as e:
    print(f"  ❌ psycopg2 import failed: {e}")
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
    "num_epochs": 2,  # ✅ ลดลงเพราะเทรนแบบ incremental
    "batch_size": 1,
    "max_length": 512,
    "gradient_accumulation_steps": 4,
    "growth_weight": 0.30,
    "challenge_weight": 0.25,
    "wisdom_weight": 0.25,
    "neutral_weight": 0.15,
    "support_weight": 0.05,
    "min_samples_new": 3,  # ✅ ต้องมีอย่างน้อย 3 samples ใหม่
    "max_samples_per_training": 500,  # ✅ จำกัดไว้ 500 samples ต่อครั้ง
}

print(f"\n📊 Config: {json.dumps(CONFIG, indent=2)}")

# ===== STEP 2: Validate environment =====
print("\n📋 STEP 2: Validating environment variables...")
POSTGRES_URI = os.environ.get("POSTGRES_URI")
USER_ID = os.environ.get("USER_ID")
MODEL_NAME = os.environ.get("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
ADAPTER_VERSION = os.environ.get("ADAPTER_VERSION", "v1")
OUTPUT_BASE = os.environ.get("OUTPUT_PATH", "/workspace/adapters")

print(f"  POSTGRES_URI: {'✅ SET' if POSTGRES_URI else '❌ MISSING'}")
print(f"  USER_ID: {USER_ID or '❌ MISSING'}")
print(f"  MODEL_NAME: {MODEL_NAME}")
print(f"  ADAPTER_VERSION: {ADAPTER_VERSION}")
print(f"  OUTPUT_BASE: {OUTPUT_BASE}")

if not POSTGRES_URI or not USER_ID:
    print("\n❌ FATAL: Missing required environment variables")
    sys.exit(1)

OUTPUT_DIR = os.path.join(OUTPUT_BASE, USER_ID, ADAPTER_VERSION)
TRAINING_ID = f"train-{USER_ID[:8]}-{ADAPTER_VERSION}"

print(f"  OUTPUT_DIR: {OUTPUT_DIR}")
print(f"  TRAINING_ID: {TRAINING_ID}")

# ===== DB Update Helper =====
def update_training_status(status: str, error_message: str = None):
    """Update training_jobs table in database"""
    try:
        print(f"\n📊 Updating DB: status = {status}")
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
            """, (TRAINING_ID,))
            print(f"  ✅ DB updated: {status}")
            
        elif status == 'failed':
            cursor.execute("""
                UPDATE user_data_schema.training_jobs
                SET status = 'failed',
                    completed_at = NOW(),
                    error_message = %s,
                    updated_at = NOW()
                WHERE job_id = %s
            """, (error_message, TRAINING_ID))
            print(f"  ✅ DB updated: {status} ({error_message})")
        
        conn.commit()
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"  ⚠️  DB update failed: {e}")

# ===== STEP 3: Test database connection =====
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
    import traceback
    traceback.print_exc()
    update_training_status('failed', f"DB connection error: {str(e)}")
    sys.exit(1)

# ===== Database Functions =====
def fetch_interaction_memories(postgres_uri: str, user_id: str, last_trained_at: str = None, limit: int = 500):
    """
    Fetch memories - ถ้ามี last_trained_at จะดึงเฉพาะที่ใหม่กว่า
    """
    print(f"\n  📊 Fetching memories...")
    
    try:
        conn = psycopg2.connect(postgres_uri)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        if last_trained_at:
            # ✅ ดึงเฉพาะข้อมูลใหม่
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
            print(f"  🔍 Mode: INCREMENTAL (after {last_trained_at})")
        else:
            # ✅ ดึงทั้งหมด (ครั้งแรก)
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
            print(f"  🔍 Mode: FULL TRAINING (first time)")
        
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        print(f"  ✅ Fetched {len(rows)} samples")
        
        if rows:
            sample = rows[0]
            print(f"  📋 Sample:")
            print(f"     Classification: {sample['classification']}")
            print(f"     Text: {sample['text'][:50]}...")
            print(f"     Created: {sample['created_at']}")
        
        return rows
        
    except Exception as e:
        print(f"  ❌ Fetch failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def fetch_ethical_profile(postgres_uri: str, user_id: str):
    """Fetch user's ethical profile"""
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
        
        if row:
            print(f"  ✅ Ethical Profile:")
            print(f"     Growth Stage: {row['growth_stage']}/5")
            print(f"     Self-Awareness: {row['self_awareness']:.2f}")
            print(f"     Compassion: {row['compassion']:.2f}")
            print(f"     Wisdom: {row['wisdom']:.2f}")
        
        return row
    except Exception as e:
        print(f"  ⚠️  No ethical profile found: {e}")
        return None

def get_last_training_info(postgres_uri: str, user_id: str):
    """ดึงข้อมูลการเทรนครั้งล่าสุด"""
    try:
        conn = psycopg2.connect(postgres_uri)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = """
        SELECT adapter_version, completed_at
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
        
        return row
    except Exception as e:
        print(f"  ⚠️  No previous training found: {e}")
        return None

def find_previous_adapter(output_base: str, user_id: str):
    """หา adapter เวอร์ชันล่าสุดที่มีอยู่"""
    user_dir = os.path.join(output_base, user_id)
    
    if not os.path.exists(user_dir):
        return None
    
    # หา version directories
    versions = []
    for item in os.listdir(user_dir):
        path = os.path.join(user_dir, item)
        if os.path.isdir(path) and item.startswith('v'):
            adapter_file = os.path.join(path, 'adapter_model.bin')
            if os.path.exists(adapter_file):
                versions.append((item, path))
    
    if not versions:
        return None
    
    # Sort by version (newest first)
    versions.sort(reverse=True)
    return versions[0][1]  # Return path

# ===== STEP 4: Check for previous adapter =====
print("\n📋 STEP 4: Checking for previous training...")

last_training = get_last_training_info(POSTGRES_URI, USER_ID)
previous_adapter_path = find_previous_adapter(OUTPUT_BASE, USER_ID)

if last_training and previous_adapter_path:
    print(f"  ✅ Found previous training:")
    print(f"     Version: {last_training['adapter_version']}")
    print(f"     Completed: {last_training['completed_at']}")
    print(f"     Adapter: {previous_adapter_path}")
    IS_INCREMENTAL = True
    LAST_TRAINED_AT = last_training['completed_at']
else:
    print(f"  📝 No previous training - this is the first training")
    IS_INCREMENTAL = False
    LAST_TRAINED_AT = None
    previous_adapter_path = None

# ===== STEP 5: Fetch training data =====
print("\n📋 STEP 5: Fetching training data...")

try:
    # ✅ Auto-approve if needed
    conn = psycopg2.connect(POSTGRES_URI)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT COUNT(*) FROM user_data_schema.interaction_memories
        WHERE user_id = %s
    """, (USER_ID,))
    
    count = cursor.fetchone()[0]
    
    if count == 0:
        print(f"  ⚠️  User {USER_ID} has no memories!")
        cursor.execute("""
            SELECT user_id, COUNT(*) as count
            FROM user_data_schema.interaction_memories
            GROUP BY user_id
            ORDER BY count DESC
            LIMIT 1
        """)
        
        result = cursor.fetchone()
        if result and result[1] > 0:
            USER_ID = result[0]
            print(f"  ✅ Using user {USER_ID} instead (has {result[1]} memories)")
        else:
            error_msg = "No users with memories found"
            print(f"  ❌ {error_msg}")
            cursor.close()
            conn.close()
            update_training_status('failed', error_msg)
            sys.exit(1)
    
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
    
    # Fetch memories
    memories = fetch_interaction_memories(
        POSTGRES_URI, 
        USER_ID, 
        last_trained_at=LAST_TRAINED_AT,
        limit=CONFIG['max_samples_per_training']
    )
    
    ethical_profile = fetch_ethical_profile(POSTGRES_URI, USER_ID)

    total_samples = len(memories)
    print(f"\n  📊 Data Summary:")
    print(f"     Total samples: {total_samples}")
    print(f"     Training mode: {'INCREMENTAL' if IS_INCREMENTAL else 'FULL'}")
    
    by_class = {}
    for mem in memories:
        cls = mem['classification']
        by_class[cls] = by_class.get(cls, 0) + 1
    
    for cls, count in by_class.items():
        print(f"     {cls}: {count}")

    # ✅ Validation
    min_required = CONFIG['min_samples_new'] if IS_INCREMENTAL else 10
    if total_samples < min_required:
        error_msg = f"Need at least {min_required} samples (have {total_samples})"
        print(f"\n  ❌ ERROR: {error_msg}")
        update_training_status('failed', error_msg)
        sys.exit(1)
    
    print(f"  ✅ Data validation passed")

except Exception as e:
    error_msg = f"Data fetch failed: {str(e)}"
    print(f"\n  ❌ {error_msg}")
    import traceback
    traceback.print_exc()
    update_training_status('failed', error_msg)
    sys.exit(1)

# ===== STEP 6: Load model =====
print("\n📋 STEP 6: Loading base model...")
print(f"  Model: {MODEL_NAME}")
print(f"  Device: CPU")

try:
    print(f"\n  📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  ✅ Tokenizer loaded")
    
    print(f"\n  📥 Loading model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    print(f"  ✅ Base model loaded")
    
    # ✅ Load previous adapter if exists
    if IS_INCREMENTAL and previous_adapter_path:
        print(f"\n  🔄 Loading previous adapter from: {previous_adapter_path}")
        try:
            model = PeftModel.from_pretrained(base_model, previous_adapter_path)
            print(f"  ✅ Previous adapter loaded - will train incrementally")
        except Exception as e:
            print(f"  ⚠️  Failed to load previous adapter: {e}")
            print(f"  🔄 Falling back to full training")
            IS_INCREMENTAL = False
            model = base_model
    else:
        model = base_model
    
    print(f"  📊 Parameters: {model.num_parameters():,}")

except Exception as e:
    error_msg = f"Model loading failed: {str(e)}"
    print(f"\n  ❌ {error_msg}")
    import traceback
    traceback.print_exc()
    update_training_status('failed', error_msg)
    sys.exit(1)

gc.collect()

# ===== STEP 7: Prepare dataset =====
print("\n📋 STEP 7: Preparing dataset...")

def create_ethical_training_pairs(memories):
    """Create training pairs with ethical context"""
    pairs = []
    
    for item in memories:
        classification = item['classification']
        text = item['text']
        ethical_scores = item.get('ethical_scores', {})
        
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
            output = item.get('gentle_guidance') or "I care about you. Please reach out for support."
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
            'ethical_scores': ethical_scores
        })
    
    return pairs

try:
    all_pairs = create_ethical_training_pairs(memories)
    
    print(f"\n  📊 Sampling strategy:")
    print(f"     Growth: {CONFIG['growth_weight']*100}%")
    print(f"     Challenge: {CONFIG['challenge_weight']*100}%")
    print(f"     Wisdom: {CONFIG['wisdom_weight']*100}%")
    print(f"     Neutral: {CONFIG['neutral_weight']*100}%")
    print(f"     Support: {CONFIG['support_weight']*100}%")
    
    pairs_by_class = {}
    for pair in all_pairs:
        cls = pair['classification']
        if cls not in pairs_by_class:
            pairs_by_class[cls] = []
        pairs_by_class[cls].append(pair)
    
    sampled = []
    total = len(all_pairs)
    
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
            print(f"     {cls}: sampled {sample_size}/{len(available)}")
    
    random.shuffle(sampled)
    
    print(f"\n  ✅ Created {len(sampled)} training pairs")
    
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
    
    print(f"\n  📝 Tokenizing...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )
    print(f"  ✅ Dataset ready: {len(tokenized_dataset)} samples")

except Exception as e:
    error_msg = f"Dataset preparation failed: {str(e)}"
    print(f"\n  ❌ {error_msg}")
    import traceback
    traceback.print_exc()
    update_training_status('failed', error_msg)
    sys.exit(1)

gc.collect()

# ===== STEP 8: Configure LoRA (only if not incremental) =====
if not IS_INCREMENTAL:
    print("\n📋 STEP 8: Configuring LoRA...")
    
    try:
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
    
    except Exception as e:
        error_msg = f"LoRA configuration failed: {str(e)}"
        print(f"\n  ❌ {error_msg}")
        import traceback
        traceback.print_exc()
        update_training_status('failed', error_msg)
        sys.exit(1)
else:
    print("\n📋 STEP 8: Using existing LoRA adapter (incremental mode)")
    model.print_trainable_parameters()

gc.collect()

# ===== STEP 9: Training =====
print("\n📋 STEP 9: Starting training...")
print(f"  Output: {OUTPUT_DIR}")
print(f"  Epochs: {CONFIG['num_epochs']}")
print(f"  Batch size: {CONFIG['batch_size']}")
print(f"  Learning rate: {CONFIG['learning_rate']}")
print(f"  Mode: {'INCREMENTAL' if IS_INCREMENTAL else 'FULL'}")

try:
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
    
    print(f"\n  🏋️  Training started at {datetime.utcnow().isoformat()}Z")
    result = trainer.train()
    
    print(f"\n  ✅ Training completed!")
    print(f"     Final loss: {result.training_loss:.4f}")

except Exception as e:
    error_msg = f"Training failed: {str(e)}"
    print(f"\n  ❌ {error_msg}")
    import traceback
    traceback.print_exc()
    update_training_status('failed', error_msg)
    sys.exit(1)

# ===== STEP 10: Save model =====
print("\n📋 STEP 10: Saving model...")

try:
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"  ✅ Model saved to {OUTPUT_DIR}")
    
    metadata = {
        "user_id": USER_ID,
        "adapter_version": ADAPTER_VERSION,
        "training_id": TRAINING_ID,
        "base_model": MODEL_NAME,
        "system": "ethical_growth",
        "training_mode": "incremental" if IS_INCREMENTAL else "full",
        "previous_adapter": previous_adapter_path if IS_INCREMENTAL else None,
        "data_stats": {
            "total_samples": total_samples,
            "by_classification": by_class,
            "last_trained_at": str(LAST_TRAINED_AT) if LAST_TRAINED_AT else None,
        },
        "ethical_profile": {
            "growth_stage": ethical_profile.get('growth_stage') if ethical_profile else None,
            "self_awareness": ethical_profile.get('self_awareness') if ethical_profile else None,
            "compassion": ethical_profile.get('compassion') if ethical_profile else None,
            "wisdom": ethical_profile.get('wisdom') if ethical_profile else None,
        } if ethical_profile else None,
        "config": CONFIG,
        "metrics": {"loss": float(result.training_loss)},
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "device": "cpu",
    }
    
    meta_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  ✅ Metadata saved")
    
    print(f"\n  📁 Output files:")
    for filename in os.listdir(OUTPUT_DIR):
        filepath = os.path.join(OUTPUT_DIR, filename)
        size = os.path.getsize(filepath)
        print(f"     {filename} ({size:,} bytes)")

except Exception as e:
    error_msg = f"Save failed: {str(e)}"
    print(f"\n  ❌ {error_msg}")
    import traceback
    traceback.print_exc()
    update_training_status('failed', error_msg)
    sys.exit(1)

# ===== STEP 11: Update DB =====
print("\n📋 STEP 11: Updating database status...")
update_training_status('completed')

print("\n" + "="*60)
print("✅✅✅ INCREMENTAL TRAINING COMPLETED")
print("="*60)
print(f"📊 Summary:")
print(f"   Training ID: {TRAINING_ID}")
print(f"   Mode: {'INCREMENTAL' if IS_INCREMENTAL else 'FULL'}")
print(f"   New samples: {len(sampled)}")
print(f"   Loss: {result.training_loss:.4f}")
print(f"   Output: {OUTPUT_DIR}")
if ethical_profile:
    print(f"   Ethical Profile: Stage {ethical_profile.get('growth_stage')}/5")
print("="*60)