#!/usr/bin/env python3
"""
LoRA Training Pipeline - Ethical Growth System WITH DEBUG LOGS
✅ เก็บ ethical scoring ไว้ทั้งหมด
✅ เพิ่ม DB update เมื่อเสร็จหรือล้มเหลว
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
print("🚀 LoRA Training Script Starting")
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
    from peft import LoraConfig, get_peft_model, TaskType
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
    "num_epochs": 3,
    "batch_size": 1,
    "max_length": 512,
    "gradient_accumulation_steps": 4,
    "growth_weight": 0.30,
    "challenge_weight": 0.25,
    "wisdom_weight": 0.25,
    "neutral_weight": 0.15,
    "support_weight": 0.05,
    "min_samples_total": 10,
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

# ===== ✅ NEW: DB Update Helper =====
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
def fetch_interaction_memories(postgres_uri: str, user_id: str, classification: str = None, limit: int = 500):
    """Fetch from interaction_memories with ethical scores"""
    print(f"\n  📊 Fetching memories (classification: {classification or 'all'}, limit: {limit})...")
    
    try:
        conn = psycopg2.connect(postgres_uri)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        if classification:
            query = """
            SELECT text, classification, ethical_scores, gentle_guidance, 
                   reflection_prompt, training_weight
            FROM user_data_schema.interaction_memories
            WHERE user_id = %s
              AND classification = %s
              AND approved_for_training = TRUE
            ORDER BY created_at DESC
            LIMIT %s
            """
            cursor.execute(query, (user_id, classification, limit))
        else:
            query = """
            SELECT text, classification, ethical_scores, gentle_guidance, 
                   reflection_prompt, training_weight
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
        
        # Show sample
        if rows:
            sample = rows[0]
            print(f"  📋 Sample:")
            print(f"     Classification: {sample['classification']}")
            print(f"     Text: {sample['text'][:50]}...")
            if sample.get('ethical_scores'):
                print(f"     Ethical Scores: {json.dumps(sample['ethical_scores'])[:80]}...")
        
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

# ===== STEP 4: Fetch training data =====
print("\n📋 STEP 4: Fetching training data with ethical scores...")

try:
    memories = fetch_interaction_memories(POSTGRES_URI, USER_ID)
    ethical_profile = fetch_ethical_profile(POSTGRES_URI, USER_ID)

    total_samples = len(memories)
    print(f"\n  📊 Data Summary:")
    print(f"     Total samples: {total_samples}")
    
    # Count by classification
    by_class = {}
    for mem in memories:
        cls = mem['classification']
        by_class[cls] = by_class.get(cls, 0) + 1
    
    for cls, count in by_class.items():
        print(f"     {cls}: {count}")

    if total_samples < CONFIG['min_samples_total']:
        error_msg = f"Need at least {CONFIG['min_samples_total']} samples (have {total_samples})"
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

# ===== STEP 5: Load model =====
print("\n📋 STEP 5: Loading base model...")
print(f"  Model: {MODEL_NAME}")
print(f"  Device: CPU")
print(f"  ⚠️  This may take 2-5 minutes on first run...")

try:
    print(f"\n  📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  ✅ Tokenizer loaded")
    
    print(f"\n  📥 Loading model (this is slow)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    print(f"  ✅ Model loaded")
    print(f"  📊 Parameters: {model.num_parameters():,}")

except Exception as e:
    error_msg = f"Model loading failed: {str(e)}"
    print(f"\n  ❌ {error_msg}")
    import traceback
    traceback.print_exc()
    update_training_status('failed', error_msg)
    sys.exit(1)

gc.collect()

# ===== STEP 6: Prepare dataset with ethical scoring =====
print("\n📋 STEP 6: Preparing dataset with ethical scoring...")

def create_ethical_training_pairs(memories):
    """Create training pairs with ethical context"""
    pairs = []
    
    for item in memories:
        classification = item['classification']
        text = item['text']
        ethical_scores = item.get('ethical_scores', {})
        
        # ✅ Use ethical scores to create better training pairs
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
    # Create pairs
    all_pairs = create_ethical_training_pairs(memories)
    
    # Sample based on ethical weights
    print(f"\n  📊 Sampling strategy:")
    print(f"     Growth: {CONFIG['growth_weight']*100}%")
    print(f"     Challenge: {CONFIG['challenge_weight']*100}%")
    print(f"     Wisdom: {CONFIG['wisdom_weight']*100}%")
    print(f"     Neutral: {CONFIG['neutral_weight']*100}%")
    print(f"     Support: {CONFIG['support_weight']*100}%")
    
    # Group by classification
    pairs_by_class = {}
    for pair in all_pairs:
        cls = pair['classification']
        if cls not in pairs_by_class:
            pairs_by_class[cls] = []
        pairs_by_class[cls].append(pair)
    
    # Sample each category
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
    
    print(f"\n  ✅ Created {len(sampled)} ethical training pairs")
    
    # Show sample pair
    if sampled:
        sample = sampled[0]
        print(f"\n  📋 Sample training pair:")
        print(f"     Instruction: {sample['instruction']}")
        print(f"     Input: {sample['input'][:50]}...")
        print(f"     Output: {sample['output'][:50]}...")
        print(f"     Weight: {sample['weight']}")
    
    # Tokenize
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

# ===== STEP 7: Configure LoRA =====
print("\n📋 STEP 7: Configuring LoRA...")

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

gc.collect()

# ===== STEP 8: Training =====
print("\n📋 STEP 8: Starting training...")
print(f"  Output: {OUTPUT_DIR}")
print(f"  Epochs: {CONFIG['num_epochs']}")
print(f"  Batch size: {CONFIG['batch_size']}")
print(f"  Learning rate: {CONFIG['learning_rate']}")
print(f"  ⚠️  This will take 10-30 minutes on CPU...")

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

# ===== STEP 9: Save model =====
print("\n📋 STEP 9: Saving model...")

try:
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"  ✅ Model saved to {OUTPUT_DIR}")
    
    # Save metadata with ethical info
    metadata = {
        "user_id": USER_ID,
        "adapter_version": ADAPTER_VERSION,
        "training_id": TRAINING_ID,
        "base_model": MODEL_NAME,
        "system": "ethical_growth",
        "data_stats": {
            "total_samples": total_samples,
            "by_classification": by_class,
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
    
    # List files
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

# ===== ✅ STEP 10: Update DB Status to Completed =====
print("\n📋 STEP 10: Updating database status...")
update_training_status('completed')

print("\n" + "="*60)
print("✅✅✅ TRAINING COMPLETED SUCCESSFULLY")
print("="*60)
print(f"📊 Summary:")
print(f"   Training ID: {TRAINING_ID}")
print(f"   Samples: {len(sampled)}")
print(f"   Loss: {result.training_loss:.4f}")
print(f"   Output: {OUTPUT_DIR}")
print(f"   Ethical Profile: Stage {ethical_profile.get('growth_stage') if ethical_profile else 'N/A'}/5")
print("="*60)