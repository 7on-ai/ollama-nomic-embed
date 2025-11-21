#!/usr/bin/env python3
"""
LoRA Training Pipeline - Ethical Growth System with Volume Integration
- Uses interaction_memories table (NEW)
- Maps classifications: growth_memory, challenge_memory, wisdom_moment
- CPU-compatible: torch.float32, fp16=False
- ✅ NEW: Copies trained adapter to shared volume for Ollama
"""

import os
import sys
import json
import random
import gc
import shutil
from datetime import datetime
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling,
)
import warnings
warnings.filterwarnings("ignore")

from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import psycopg2
from psycopg2.extras import RealDictCursor

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
    # Weights by classification (5 types)
    "growth_weight": 0.30,
    "challenge_weight": 0.25,
    "wisdom_weight": 0.25,
    "neutral_weight": 0.15,
    "support_weight": 0.05,
    "min_samples_total": 10,
}

# ===== Progress Callback =====
class ProgressCallback(TrainerCallback):
    """Print progress during training"""
    def on_step_end(self, args, state, control, **kwargs):
        try:
            if state.global_step and state.max_steps:
                if state.global_step % 5 == 0:
                    progress = (state.global_step / state.max_steps) * 100
                    print(f"📊 Progress: {progress:.1f}% (Step {state.global_step}/{state.max_steps})")
        except Exception:
            pass

    def on_epoch_end(self, args, state, control, **kwargs):
        try:
            print(f"✅ Epoch {int(state.epoch)} completed")
        except Exception:
            pass

def print_step(step_num, title):
    print("\n" + "="*60)
    print(f"Step {step_num}: {title}")
    print("="*60)

# ===== Database Fetchers =====

def fetch_interaction_memories(postgres_uri: str, user_id: str, classification: str = None, limit: int = 500):
    """Fetch from interaction_memories table"""
    print(f"  🔍 Connecting to DB for interaction_memories (classification: {classification or 'all'})...")
    conn = psycopg2.connect(postgres_uri)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    if classification:
        query = """
        SELECT text, classification, ethical_scores, gentle_guidance, reflection_prompt, training_weight
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
        SELECT text, classification, ethical_scores, gentle_guidance, reflection_prompt, training_weight
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

def fetch_ethical_profile(postgres_uri: str, user_id: str):
    """Fetch user's ethical profile"""
    conn = psycopg2.connect(postgres_uri)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    query = """
    SELECT 
        growth_stage,
        self_awareness,
        emotional_regulation,
        compassion,
        integrity,
        growth_mindset,
        wisdom,
        transcendence,
        total_interactions,
        breakthrough_moments
    FROM user_data_schema.ethical_profiles
    WHERE user_id = %s
    """
    cursor.execute(query, (user_id,))
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    return row

# ===== Data Pair Creation =====

def create_training_pairs(memories):
    """Create instruction-input-output pairs from interaction_memories"""
    pairs = []
    
    for item in memories:
        classification = item['classification']
        text = item['text']
        
        if classification == 'growth_memory':
            pairs.append({
                'instruction': 'Respond as helpful personal assistant supporting growth',
                'input': text,
                'output': text,
                'weight': 1.5
            })
        
        elif classification == 'challenge_memory':
            output = item.get('gentle_guidance') or f"I understand this is challenging. {text}"
            pairs.append({
                'instruction': 'Respond with compassion to a challenge',
                'input': text,
                'output': output,
                'weight': 2.0
            })
        
        elif classification == 'wisdom_moment':
            reflection = item.get('reflection_prompt', '')
            output = f"{text}\n\n💭 {reflection}" if reflection else text
            pairs.append({
                'instruction': 'Share wisdom and insight',
                'input': text,
                'output': output,
                'weight': 2.5
            })
        
        elif classification == 'neutral_interaction':
            pairs.append({
                'instruction': 'Respond naturally to everyday conversation',
                'input': text,
                'output': text,
                'weight': 0.8
            })
        
        elif classification == 'needs_support':
            pairs.append({
                'instruction': 'Provide supportive response with care',
                'input': text,
                'output': item.get('gentle_guidance') or "I care about you. Please reach out for support.",
                'weight': 1.0
            })
    
    return pairs

def prepare_lora_dataset(memories, tokenizer):
    """Prepare dataset from interaction_memories"""
    if not memories:
        return None
    
    # Group by classification
    by_class = {
        'growth_memory': [],
        'challenge_memory': [],
        'wisdom_moment': [],
        'neutral_interaction': [],
        'needs_support': []
    }
    
    for mem in memories:
        cls = mem['classification']
        if cls in by_class:
            by_class[cls].append(mem)
    
    # Calculate sampling based on weights
    total = len(memories)
    
    growth_samples = int(total * CONFIG['growth_weight'])
    challenge_samples = int(total * CONFIG['challenge_weight'])
    wisdom_samples = int(total * CONFIG['wisdom_weight'])
    neutral_samples = int(total * CONFIG['neutral_weight'])
    support_samples = int(total * CONFIG['support_weight'])
    
    sampled = []
    
    # Sample each category
    if by_class['growth_memory']:
        sampled.extend(random.sample(by_class['growth_memory'], 
                                     min(growth_samples, len(by_class['growth_memory']))))
    
    if by_class['challenge_memory']:
        sampled.extend(random.sample(by_class['challenge_memory'], 
                                     min(challenge_samples, len(by_class['challenge_memory']))))
    
    if by_class['wisdom_moment']:
        sampled.extend(random.sample(by_class['wisdom_moment'], 
                                     min(wisdom_samples, len(by_class['wisdom_moment']))))
    
    if by_class['neutral_interaction']:
        sampled.extend(random.sample(by_class['neutral_interaction'], 
                                     min(neutral_samples, len(by_class['neutral_interaction']))))
    
    if by_class['needs_support']:
        sampled.extend(random.sample(by_class['needs_support'], 
                                     min(support_samples, len(by_class['needs_support']))))
    
    # Create pairs
    all_pairs = create_training_pairs(sampled)
    random.shuffle(all_pairs)
    
    print(f"  📊 Dataset composition:")
    print(f"     Growth: {len(by_class['growth_memory'])} | Challenge: {len(by_class['challenge_memory'])} | Wisdom: {len(by_class['wisdom_moment'])}")
    print(f"     Neutral: {len(by_class['neutral_interaction'])} | Support: {len(by_class['needs_support'])}")
    print(f"     Total pairs: {len(all_pairs)}")
    
    # Tokenize
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
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )
    
    return tokenized_dataset

# ===== Memory cleanup =====
def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass
    print("  🧹 Memory cleaned")

# ===== ✅ NEW: Volume Integration =====

def copy_to_volume(output_dir: str, volume_mount: str, user_id: str, adapter_version: str):
    """
    Copy trained adapter to shared volume for Ollama
    
    Args:
        output_dir: Local output directory (/workspace/adapters/user_id/version)
        volume_mount: Volume mount path (/models/adapters)
        user_id: User ID
        adapter_version: Version string (v1234567890)
    
    Returns:
        bool: True if copy successful, False otherwise
    """
    try:
        print_step(7, "Copying to Volume for Ollama")
        
        # Check if volume is mounted
        if not os.path.exists(volume_mount):
            print(f"⚠️  Volume not mounted at {volume_mount}")
            print("   Files will remain in job storage only")
            print("   To enable volume mount, add in Northflank Job config:")
            print(f"   volumes:")
            print(f"     - name: lora-adapters")
            print(f"       mountPath: {volume_mount}")
            return False
        
        # Create destination directory
        dest_dir = os.path.join(volume_mount, user_id, adapter_version)
        os.makedirs(dest_dir, exist_ok=True)
        
        print(f"  📂 Source: {output_dir}")
        print(f"  📂 Destination: {dest_dir}")
        
        # List of essential files to copy
        files_to_copy = [
            'adapter_model.safetensors',
            'adapter_config.json',
            'tokenizer.json',
            'tokenizer_config.json',
            'special_tokens_map.json',
            'README.md',
            'metadata.json',
        ]
        
        copied_count = 0
        total_size = 0
        
        print("  📦 Copying files...")
        for filename in files_to_copy:
            src = os.path.join(output_dir, filename)
            dst = os.path.join(dest_dir, filename)
            
            if os.path.exists(src):
                shutil.copy2(src, dst)
                file_size = os.path.getsize(dst)
                total_size += file_size
                copied_count += 1
                print(f"    ✅ {filename} ({file_size / 1024 / 1024:.1f} MB)")
            else:
                print(f"    ⚠️  Missing: {filename}")
        
        print(f"\n  ✅ Copied {copied_count}/{len(files_to_copy)} files")
        print(f"  📊 Total size: {total_size / 1024 / 1024:.1f} MB")
        
        # ✅ Create marker file for Ollama integration
        marker_data = {
            'user_id': user_id,
            'adapter_version': adapter_version,
            'copied_at': datetime.utcnow().isoformat() + 'Z',
            'file_count': copied_count,
            'total_size_bytes': total_size,
            'ollama_model_name': f'sunday-ai-{user_id}',
            'adapter_path': dest_dir,
            'status': 'ready',
        }
        
        marker_path = os.path.join(dest_dir, '.ready')
        with open(marker_path, 'w') as f:
            json.dump(marker_data, f, indent=2)
        
        print(f"  ✅ Created marker file: .ready")
        print(f"\n  🎯 Ollama Integration:")
        print(f"     Model name: sunday-ai-{user_id}")
        print(f"     Adapter path: {dest_dir}")
        print(f"     Status: READY")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Volume copy error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# ===== Core Training Pipeline =====

def train_complete_lora(postgres_uri, user_id, base_model, adapter_name, output_dir):
    print("\n" + "="*60)
    print("🚀 LoRA Training Pipeline (Ethical Growth System)")
    print("="*60)
    print(f"👤 User: {user_id}")
    print(f"📦 Model: {base_model}")
    print(f"📝 Adapter: {adapter_name}")
    print(f"💾 Output: {output_dir}")
    print("="*60)

    device = "cpu"
    print(f"🖥️  Device: {device.upper()} (CPU-only)")

    # Step 1: Fetch data
    print_step(1, "Fetching Data from Interaction Memories")
    memories = fetch_interaction_memories(postgres_uri, user_id)
    ethical_profile = fetch_ethical_profile(postgres_uri, user_id)

    total_samples = len(memories)
    print("\n  📊 Validation:")
    print(f"     Total memories: {total_samples}")
    print(f"     Growth stage: {ethical_profile.get('growth_stage', 2) if ethical_profile else 2}")

    if total_samples < CONFIG['min_samples_total']:
        raise ValueError(f"❌ Need at least {CONFIG['min_samples_total']} samples (have {total_samples})")
    
    print("  ✅ Validation passed")

    cleanup_memory()

    # Step 2: Load model & tokenizer
    print_step(2, "Loading Base Model (CPU-friendly)")
    print(f"  📥 Loading {base_model} ...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"  ❗ Warning: model.load failed with error: {e}")
        print("  Trying without trust_remote_code...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
            low_cpu_mem_usage=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cleanup_memory()

    # Step 3: Prepare dataset
    print_step(3, "Preparing Training Dataset")
    dataset = prepare_lora_dataset(memories, tokenizer)
    if dataset is None:
        raise RuntimeError("No dataset prepared for training")
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Step 4: Configure LoRA
    print_step(4, "Configuring LoRA")
    lora_config = LoraConfig(
        r=CONFIG['r'],
        lora_alpha=CONFIG['lora_alpha'],
        target_modules=CONFIG['target_modules'],
        lora_dropout=CONFIG['lora_dropout'],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    cleanup_memory()

    # Step 5: Training
    print_step(5, "Training LoRA Adapter (CPU, float32)")
    training_args = TrainingArguments(
        output_dir=output_dir,
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        callbacks=[ProgressCallback()],
    )

    print("  🏋️  Starting training (may be slow on CPU)...")
    result = trainer.train()

    # Step 6: Save artifacts
    print_step(6, "Saving Artifacts")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("  ✅ Model and tokenizer saved")

    # Cleanup before volume copy
    del model, trainer, dataset
    cleanup_memory()

    # Step 7: Save metadata
    metadata = {
        "user_id": user_id,
        "adapter_name": adapter_name,
        "base_model": base_model,
        "system": "ethical_growth",
        "data_stats": {
            "total_samples": total_samples,
        },
        "ethical_profile": {
            "growth_stage": ethical_profile.get('growth_stage', 2) if ethical_profile else 2,
            "self_awareness": ethical_profile.get('self_awareness', 0.5) if ethical_profile else 0.5,
        } if ethical_profile else None,
        "config": CONFIG,
        "metrics": {"loss": float(getattr(result, "training_loss", -1))},
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "device": "cpu",
        "optimizations": "CPU-compatible float32",
    }

    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*60)
    print("✅ Training Completed Successfully!")
    print("="*60)
    print(f"📊 Final loss: {metadata['metrics']['loss']}")
    print(f"📈 Samples: {total_samples}")
    print(f"📁 Output: {output_dir}")
    print("="*60 + "\n")

    return metadata

# ===== Entrypoint =====

if __name__ == "__main__":
    POSTGRES_URI = os.environ.get("POSTGRES_URI")
    USER_ID = os.environ.get("USER_ID")
    MODEL_NAME = os.environ.get("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ADAPTER_VERSION = os.environ.get("ADAPTER_VERSION", "v1")
    OUTPUT_BASE = os.environ.get("OUTPUT_PATH", "/workspace/adapters")
    VOLUME_MOUNT = os.environ.get("VOLUME_MOUNT", "/models/adapters")
    
    OUTPUT_DIR = os.path.join(OUTPUT_BASE, USER_ID or "unknown_user", ADAPTER_VERSION)

    # Validate env
    if not POSTGRES_URI or not USER_ID:
        print("❌ Missing required env vars: POSTGRES_URI and USER_ID are required", file=sys.stderr)
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        # Run training
        metadata = train_complete_lora(
            postgres_uri=POSTGRES_URI,
            user_id=USER_ID,
            base_model=MODEL_NAME,
            adapter_name=ADAPTER_VERSION,
            output_dir=OUTPUT_DIR,
        )

        # ✅ NEW: Copy to volume
        volume_success = copy_to_volume(
            output_dir=OUTPUT_DIR,
            volume_mount=VOLUME_MOUNT,
            user_id=USER_ID,
            adapter_version=ADAPTER_VERSION
        )

        # Add volume info to metadata
        if volume_success:
            metadata['volume_integration'] = {
                'copied': True,
                'volume_path': os.path.join(VOLUME_MOUNT, USER_ID, ADAPTER_VERSION),
                'ollama_ready': True,
            }
            print("\n🎉 Adapter ready for Ollama inference!")
        else:
            metadata['volume_integration'] = {
                'copied': False,
                'note': 'Volume not mounted - files in job storage only',
            }
            print("\n⚠️  Volume not available - files saved to job storage")

        # Print machine-parseable metadata
        print("\n===METADATA_START===")
        print(json.dumps(metadata))
        print("===METADATA_END===")
        
        print(f"\n[{datetime.utcnow().isoformat()}Z INFO ] Process terminated with exit code 0")

    except Exception as e:
        import traceback
        print(f"\n❌ Training failed: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)