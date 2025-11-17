#!/usr/bin/env python3
"""
Patched CPU-compatible LoRA training pipeline
- Model default: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- DB schema mapped to stm_good, stm_bad, mcl_chains (user_data_schema)
- LoRA target modules: ["q_proj","k_proj","v_proj","o_proj"] (full attention)
- CPU-only mode: torch.float32, fp16=False
"""

import os
import sys
import json
import random
import gc
from datetime import datetime

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

# Optional detector training libs (lightweight)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

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
    "good_weight": 0.4,
    "counterfactual_weight": 0.3,
    "mcl_weight": 0.3,
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

# ===== Database fetchers (map to schema from script1) =====
def fetch_good_channel(postgres_uri: str, user_id: str, limit: int = 500):
    print("  🔍 Connecting to DB for stm_good...")
    conn = psycopg2.connect(postgres_uri)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    query = """
    SELECT text, metadata
    FROM user_data_schema.stm_good
    WHERE user_id = %s
      AND approved_for_consolidation = TRUE
    ORDER BY created_at DESC
    LIMIT %s
    """
    cursor.execute(query, (user_id, limit))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    print(f"  ✅ Fetched {len(rows)} good samples")
    return [{'text': r['text'], 'metadata': r.get('metadata')} for r in rows]

def fetch_bad_channel(postgres_uri: str, user_id: str, limit: int = 500):
    print("  🔍 Connecting to DB for stm_bad...")
    conn = psycopg2.connect(postgres_uri)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    query = """
    SELECT text, shadow_tag, safe_counterfactual, severity_score
    FROM user_data_schema.stm_bad
    WHERE user_id = %s
      AND approved_for_shadow_learning = TRUE
      AND safe_counterfactual IS NOT NULL
    ORDER BY created_at DESC
    LIMIT %s
    """
    cursor.execute(query, (user_id, limit))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    print(f"  ✅ Fetched {len(rows)} bad samples")
    return rows

def fetch_mcl_chains(postgres_uri: str, user_id: str, limit: int = 200):
    print("  🔍 Connecting to DB for mcl_chains...")
    conn = psycopg2.connect(postgres_uri)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    query = """
    SELECT event_chain, moral_classification, summary,
           intention_score, necessity_score, harm_score, benefit_score
    FROM user_data_schema.mcl_chains
    WHERE user_id = %s
      AND approved_for_training = TRUE
    ORDER BY created_at DESC
    LIMIT %s
    """
    cursor.execute(query, (user_id, limit))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    print(f"  ✅ Fetched {len(rows)} MCL chains")
    return rows

# ===== Data pair creation (adapted from script1 & script2) =====
def create_good_channel_pairs(good_data):
    pairs = []
    for item in good_data:
        pairs.append({
            'instruction': 'Respond as helpful personal assistant',
            'input': item['text'],
            'output': item['text']
        })
    return pairs

def create_counterfactual_pairs(bad_data):
    pairs = []
    for item in bad_data:
        # safe_counterfactual expected present per fetch query
        pairs.append({
            'instruction': 'Respond safely to potentially harmful request',
            'input': item['text'],
            'output': item['safe_counterfactual']
        })
    return pairs

def create_mcl_pairs(mcl_data):
    pairs = []
    for item in mcl_data:
        # event_chain assumed to be list of dicts with 'text' key (as in script1)
        try:
            chain_text = ' → '.join([e['text'] for e in item['event_chain']])
        except Exception:
            # fallback if event_chain is stored as text
            chain_text = str(item.get('event_chain', ''))
        pairs.append({
            'instruction': 'Analyze this sequence from a moral perspective',
            'input': f"Events: {chain_text}\nClassification: {item.get('moral_classification')}",
            'output': f"{item.get('summary', '')} (Intent: {item.get('intention_score', 0):.2f})"
        })
    return pairs

def prepare_lora_dataset(good_pairs, counterfactual_pairs, mcl_pairs, tokenizer):
    total = len(good_pairs) + len(counterfactual_pairs) + len(mcl_pairs)
    if total == 0:
        return None

    good_samples = int(total * CONFIG['good_weight'])
    counter_samples = int(total * CONFIG['counterfactual_weight'])
    mcl_samples = int(total * CONFIG['mcl_weight'])

    sampled_good = random.sample(good_pairs, min(good_samples or 1, len(good_pairs))) if good_pairs else []
    sampled_counter = random.sample(counterfactual_pairs, min(counter_samples or 1, len(counterfactual_pairs))) if counterfactual_pairs else []
    sampled_mcl = random.sample(mcl_pairs, min(mcl_samples or 1, len(mcl_pairs))) if mcl_pairs else []

    all_pairs = sampled_good + sampled_counter + sampled_mcl
    random.shuffle(all_pairs)

    print(f"  📊 Dataset composition:")
    print(f"     Good: {len(sampled_good)} | Counter: {len(sampled_counter)} | MCL: {len(sampled_mcl)}")
    print(f"     Total: {len(all_pairs)} training pairs")

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

# ===== Lightweight detector trainer (optional) =====
def train_detectors(bad_data, output_dir):
    detectors = {}
    from collections import defaultdict
    by_tag = defaultdict(list)
    for item in bad_data:
        by_tag[item['shadow_tag']].append(item['text'])

    print(f"  🔍 Training {len(by_tag)} detectors (lightweight)...")
    for tag, texts in by_tag.items():
        if len(texts) < 5:
            continue
        negative_samples = ["Normal conversation"] * len(texts)
        X_texts = texts + negative_samples
        y = [1] * len(texts) + [0] * len(negative_samples)
        vectorizer = TfidfVectorizer(max_features=100)
        X = vectorizer.fit_transform(X_texts)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)
        detectors[tag] = {'vectorizer': vectorizer, 'classifier': clf, 'samples': len(texts)}

    os.makedirs(output_dir, exist_ok=True)
    detector_path = f"{output_dir}/detectors.pkl"
    with open(detector_path, 'wb') as f:
        pickle.dump(detectors, f)
    print(f"  ✅ Saved {len(detectors)} detectors -> {detector_path}")
    return detectors

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

# ===== Core training pipeline =====
def train_complete_lora(postgres_uri, user_id, base_model, adapter_name, output_dir):
    print("\n" + "="*60)
    print("🚀 LoRA Training Pipeline (Patched - CPU mode)")
    print("="*60)
    print(f"👤 User: {user_id}")
    print(f"📦 Model: {base_model}")
    print(f"📝 Adapter: {adapter_name}")
    print(f"💾 Output: {output_dir}")
    print("="*60)

    device = "cpu"
    print(f"🖥️  Device: {device.upper()} (CPU-only)")

    # Step 1: Fetch data
    print_step(1, "Fetching Data")
    good_data = fetch_good_channel(postgres_uri, user_id)
    bad_data = fetch_bad_channel(postgres_uri, user_id)
    mcl_data = fetch_mcl_chains(postgres_uri, user_id)

    total_samples = len(good_data) + len(bad_data) + len(mcl_data)
    print("\n  📊 Validation:")
    print(f"     Total: {total_samples} | Good: {len(good_data)} | Bad: {len(bad_data)} | MCL: {len(mcl_data)}")

    if total_samples < CONFIG['min_samples_total']:
        raise ValueError(f"❌ Need at least {CONFIG['min_samples_total']} samples (have {total_samples})")
    if len(good_data) < 5:
        raise ValueError(f"❌ Need at least 5 good samples (have {len(good_data)})")
    print("  ✅ Validation passed")

    # Step 2: Create pairs
    print_step(2, "Creating Training Pairs")
    good_pairs = create_good_channel_pairs(good_data)
    counterfactual_pairs = create_counterfactual_pairs(bad_data) if bad_data else []
    mcl_pairs = create_mcl_pairs(mcl_data) if mcl_data else []
    print(f"  ✅ Created {len(good_pairs) + len(counterfactual_pairs) + len(mcl_pairs)} pairs")

    # Optional: train detectors (light)
    print_step(3, "(Optional) Training detectors")
    detectors = {}
    try:
        detectors = train_detectors(bad_data, output_dir)
    except Exception as e:
        print(f"  ⚠️ Detector training skipped/error: {e}")

    cleanup_memory()

    # Step 4: Load model & tokenizer (CPU-friendly)
    print_step(4, "Loading Base Model (CPU-friendly)")
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

    # Step 5: Configure LoRA (user chose full attention)
    print_step(5, "Configuring LoRA")
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

    # Step 6: Prepare dataset
    print_step(6, "Preparing Dataset")
    dataset = prepare_lora_dataset(good_pairs, counterfactual_pairs, mcl_pairs, tokenizer)
    if dataset is None:
        raise RuntimeError("No dataset prepared for training")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # free pairs
    del good_pairs, counterfactual_pairs, mcl_pairs
    cleanup_memory()

    # Step 7: Training
    print_step(7, "Training LoRA Adapter (CPU, float32)")
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

    print("  🏋️ Starting training (may be slow on CPU)...")
    result = trainer.train()

    # Step 8: Save artifacts
    print_step(8, "Saving Artifacts")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("  ✅ Model and tokenizer saved")

    # Final cleanup
    del model, trainer, dataset
    cleanup_memory()

    # Step 9: Save metadata
    metadata = {
        "user_id": user_id,
        "adapter_name": adapter_name,
        "base_model": base_model,
        "data_stats": {"total_samples": total_samples},
        "detectors_trained": list(detectors.keys()) if isinstance(detectors, dict) else [],
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
    OUTPUT_DIR = os.path.join(OUTPUT_BASE, USER_ID or "unknown_user", ADAPTER_VERSION)

    # Validate env
    if not POSTGRES_URI or not USER_ID:
        print("❌ Missing required env vars: POSTGRES_URI and USER_ID are required", file=sys.stderr)
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        metadata = train_complete_lora(
            postgres_uri=POSTGRES_URI,
            user_id=USER_ID,
            base_model=MODEL_NAME,
            adapter_name=ADAPTER_VERSION,
            output_dir=OUTPUT_DIR,
        )

        # print machine-parseable metadata block
        print("===METADATA_START===")
        print(json.dumps(metadata))
        print("===METADATA_END===")

    except Exception as e:
        import traceback
        print(f"\n❌ Training failed: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
