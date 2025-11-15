#!/usr/bin/env python3
"""
Complete LoRA Training Pipeline with Memory Optimization
"""

import json
import sys
import os
from datetime import datetime
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
import warnings
warnings.filterwarnings('ignore')
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
from datasets import Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import psycopg2
from psycopg2.extras import RealDictCursor
import pickle
import gc

# ===== Configuration =====
CONFIG = {
    "r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj"],
    "learning_rate": 1e-5,
    "num_epochs": 2,
    "batch_size": 1,
    "max_length": 512,
    "gradient_accumulation_steps": 4,  # ✅ ลด memory usage
    "good_weight": 0.4,
    "counterfactual_weight": 0.3,
    "mcl_weight": 0.3,
}

# ===== Progress Callback =====
class ProgressCallback(TrainerCallback):
    """Print progress during training"""
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            progress = (state.global_step / state.max_steps) * 100
            print(f"📊 Progress: {progress:.1f}% (Step {state.global_step}/{state.max_steps})")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"✅ Epoch {int(state.epoch)} completed")

def print_step(step_num, title):
    """Print formatted step header"""
    print(f"\n{'='*50}")
    print(f"Step {step_num}: {title}")
    print('='*50)

def fetch_good_channel(postgres_uri: str, user_id: str):
    """Fetch approved good channel data"""
    print("  🔍 Connecting to database...")
    conn = psycopg2.connect(postgres_uri)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    query = """
    SELECT text, metadata
    FROM user_data_schema.stm_good
    WHERE user_id = %s 
      AND approved_for_consolidation = TRUE
    ORDER BY created_at DESC
    LIMIT 500
    """
    
    cursor.execute(query, (user_id,))
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    
    print(f"  ✅ Fetched {len(data)} good channel samples")
    return [{'text': row['text'], 'metadata': row['metadata']} for row in data]

def fetch_bad_channel(postgres_uri: str, user_id: str):
    """Fetch approved bad channel data"""
    conn = psycopg2.connect(postgres_uri)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    query = """
    SELECT text, shadow_tag, safe_counterfactual, severity_score
    FROM user_data_schema.stm_bad
    WHERE user_id = %s 
      AND approved_for_shadow_learning = TRUE
      AND safe_counterfactual IS NOT NULL
    ORDER BY created_at DESC
    LIMIT 500
    """
    
    cursor.execute(query, (user_id,))
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    
    print(f"  ✅ Fetched {len(data)} bad channel samples")
    return data

def fetch_mcl_chains(postgres_uri: str, user_id: str):
    """Fetch approved MCL chains"""
    conn = psycopg2.connect(postgres_uri)
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    query = """
    SELECT event_chain, moral_classification, summary, 
           intention_score, necessity_score, harm_score, benefit_score
    FROM user_data_schema.mcl_chains
    WHERE user_id = %s 
      AND approved_for_training = TRUE
    ORDER BY created_at DESC
    LIMIT 200
    """
    
    cursor.execute(query, (user_id,))
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    
    print(f"  ✅ Fetched {len(data)} MCL chains")
    return data

def create_good_channel_pairs(good_data):
    """Create instruction pairs from good channel"""
    pairs = []
    for item in good_data:
        pairs.append({
            'instruction': 'Respond as helpful personal assistant',
            'input': item['text'],
            'output': item['text']
        })
    return pairs

def create_counterfactual_pairs(bad_data):
    """Create safe counterfactual pairs"""
    pairs = []
    for item in bad_data:
        pairs.append({
            'instruction': 'Respond safely to potentially harmful request',
            'input': item['text'],
            'output': item['safe_counterfactual']
        })
    return pairs

def create_mcl_pairs(mcl_data):
    """Create moral reasoning pairs"""
    pairs = []
    for item in mcl_data:
        chain_text = ' → '.join([e['text'] for e in item['event_chain']])
        pairs.append({
            'instruction': 'Analyze this sequence from a moral perspective',
            'input': f"Events: {chain_text}\nClassification: {item['moral_classification']}",
            'output': f"{item['summary']} (Intent: {item['intention_score']:.2f})"
        })
    return pairs

def prepare_lora_dataset(good_pairs, counterfactual_pairs, mcl_pairs, tokenizer):
    """Mix datasets according to weights"""
    import random
    
    total = len(good_pairs) + len(counterfactual_pairs) + len(mcl_pairs)
    
    good_samples = int(total * CONFIG['good_weight'])
    counter_samples = int(total * CONFIG['counterfactual_weight'])
    mcl_samples = int(total * CONFIG['mcl_weight'])
    
    sampled_good = random.sample(good_pairs, min(good_samples, len(good_pairs)))
    sampled_counter = random.sample(counterfactual_pairs, min(counter_samples, len(counterfactual_pairs)))
    sampled_mcl = random.sample(mcl_pairs, min(mcl_samples, len(mcl_pairs)))
    
    all_pairs = sampled_good + sampled_counter + sampled_mcl
    random.shuffle(all_pairs)
    
    print(f"  📊 Dataset composition:")
    print(f"     Good: {len(sampled_good)} | Counter: {len(sampled_counter)} | MCL: {len(sampled_mcl)}")
    print(f"     Total: {len(all_pairs)} training pairs")
    
    texts = [
        f"### Instruction:\n{p['instruction']}\n\n### Input:\n{p['input']}\n\n### Response:\n{p['output']}"
        for p in all_pairs
    ]
    
    def tokenize(examples):
        # ✅ Tokenize and create labels
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=CONFIG["max_length"],
            padding="max_length",
        )
        # ✅ Add labels (same as input_ids for causal LM)
        result["labels"] = result["input_ids"].copy()
        return result
    
    dataset = Dataset.from_dict({"text": texts})
    return dataset.map(tokenize, batched=True, remove_columns=["text"])

def train_detectors(bad_data, output_dir):
    """Train simple detectors"""
    detectors = {}
    from collections import defaultdict
    by_tag = defaultdict(list)
    
    for item in bad_data:
        by_tag[item['shadow_tag']].append(item['text'])
    
    print(f"  🔍 Training {len(by_tag)} detectors...")
    
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
        
        detectors[tag] = {
            'vectorizer': vectorizer,
            'classifier': clf,
            'samples': len(texts)
        }
    
    detector_path = f"{output_dir}/detectors.pkl"
    with open(detector_path, 'wb') as f:
        pickle.dump(detectors, f)
    
    print(f"  ✅ Saved {len(detectors)} detectors")
    return detectors

def cleanup_memory():
    """Aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("  🧹 Memory cleaned")

def train_complete_lora(postgres_uri, user_id, base_model, adapter_name, output_dir):
    """Complete training pipeline with memory optimization"""
    
    print("\n" + "="*50)
    print("🚀 LoRA Training Pipeline Started")
    print("="*50)
    print(f"👤 User: {user_id}")
    print(f"📦 Model: {base_model}")
    print(f"📝 Adapter: {adapter_name}")
    print(f"💾 Output: {output_dir}")
    print("="*50)
    
    # Step 1: Fetch data
    print_step(1, "Fetching Data")
    good_data = fetch_good_channel(postgres_uri, user_id)
    bad_data = fetch_bad_channel(postgres_uri, user_id)
    mcl_data = fetch_mcl_chains(postgres_uri, user_id)
    
    total_samples = len(good_data) + len(bad_data) + len(mcl_data)
    
    # Validation
    print("\n  📊 Validation:")
    print(f"     Total: {total_samples} | Good: {len(good_data)}")
    
    if total_samples < 10:
        raise ValueError(f"❌ Need at least 10 samples (have {total_samples})")
    
    if len(good_data) < 5:
        raise ValueError(f"❌ Need at least 5 good samples (have {len(good_data)})")
    
    print("  ✅ Validation passed")
    
    # Step 2: Create pairs
    print_step(2, "Creating Training Pairs")
    good_pairs = create_good_channel_pairs(good_data)
    counterfactual_pairs = create_counterfactual_pairs(bad_data) if bad_data else []
    mcl_pairs = create_mcl_pairs(mcl_data) if mcl_data else []
    print(f"  ✅ Created {len(good_pairs) + len(counterfactual_pairs) + len(mcl_pairs)} pairs")
    
    # ✅ Clean up data after creating pairs
    del good_data, bad_data, mcl_data
    cleanup_memory()
    
    # Step 3: Train detectors (skip if no bad data to save memory)
    detectors = {}
    if counterfactual_pairs and len(counterfactual_pairs) >= 5:
        print_step(3, "Training Detectors")
        # Recreate bad_data only for detector training
        conn = psycopg2.connect(postgres_uri)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT text, shadow_tag
            FROM user_data_schema.stm_bad
            WHERE user_id = %s AND approved_for_shadow_learning = TRUE
            LIMIT 100
        """, (user_id,))
        bad_data_small = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if bad_data_small:
            detectors = train_detectors(bad_data_small, output_dir)
            del bad_data_small
            cleanup_memory()
    else:
        print_step(3, "Skipping Detectors (insufficient data)")
    
    # Step 4: Load model with memory optimization
    print_step(4, "Loading Base Model")
    print(f"  📥 Loading {base_model}...")
    
    # ✅ Load with float16 (8-bit requires extra packages)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,  # ✅ Use half precision
        low_cpu_mem_usage=True,  # ✅ Reduce memory during loading
    )
    print("  ✅ Model loaded with float16")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    
    cleanup_memory()
    
    # Step 5: Setup LoRA
    print_step(5, "Configuring LoRA")
    lora_config = LoraConfig(
        r=CONFIG["r"],
        lora_alpha=CONFIG["lora_alpha"],
        target_modules=CONFIG["target_modules"],
        lora_dropout=CONFIG["lora_dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    cleanup_memory()
    
    # Step 6: Prepare dataset
    print_step(6, "Preparing Dataset")
    dataset = prepare_lora_dataset(good_pairs, counterfactual_pairs, mcl_pairs, tokenizer)
    
    # ✅ Clean up pairs after tokenization
    del good_pairs, counterfactual_pairs, mcl_pairs
    cleanup_memory()
    
    # Step 7: Training with memory optimization
    print_step(7, "Training LoRA Adapter")
    
    # ✅ Check if CUDA available
    use_cuda = torch.cuda.is_available()
    print(f"  🖥️  Device: {'CUDA (GPU)' if use_cuda else 'CPU'}")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=CONFIG["num_epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"],
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        fp16=use_cuda,  # ✅ Only use fp16 if GPU available
        gradient_checkpointing=True,
        optim="adamw_torch",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=[ProgressCallback()],
    )
    
    print("\n  🏋️  Starting training...")
    if use_cuda:
        print(f"  💾 Memory optimizations: fp16 + gradient checkpointing")
    else:
        print(f"  💾 Memory optimizations: gradient checkpointing (CPU mode)")
    
    result = trainer.train()
    
    # Step 8: Save
    print_step(8, "Saving Artifacts")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("  ✅ Model and tokenizer saved")
    
    # ✅ Final cleanup
    del model, trainer, dataset
    cleanup_memory()
    
    # Step 9: Metadata
    metadata = {
        "user_id": user_id,
        "adapter_name": adapter_name,
        "base_model": base_model,
        "training_composition": {
            "good_channel": len(good_pairs) if 'good_pairs' in locals() else 0,
            "counterfactuals": len(counterfactual_pairs) if 'counterfactual_pairs' in locals() else 0,
            "mcl_chains": len(mcl_pairs) if 'mcl_pairs' in locals() else 0,
        },
        "data_stats": {
            "total_samples": total_samples,
        },
        "detectors_trained": list(detectors.keys()),
        "config": CONFIG,
        "metrics": {
            "loss": float(result.training_loss),
        },
        "trained_at": datetime.now().isoformat(),
        "optimizations": "8bit/fp16 + gradient_checkpointing + gradient_accumulation",
    }
    
    with open(f"{output_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*50)
    print("✅ Training Completed Successfully!")
    print("="*50)
    print(f"📊 Final loss: {result.training_loss:.4f}")
    print(f"🔍 Detectors: {len(detectors)}")
    print(f"📈 Samples: {total_samples}")
    print(f"📁 Output: {output_dir}")
    print("="*50 + "\n")
    
    return metadata

if __name__ == "__main__":
    # Read from environment variables
    postgres_uri = os.environ.get('POSTGRES_URI')
    user_id = os.environ.get('USER_ID')
    base_model = os.environ.get('MODEL_NAME')
    adapter_name = os.environ.get('ADAPTER_VERSION')
    
    # Validation
    if not all([postgres_uri, user_id, base_model, adapter_name]):
        print("❌ Missing required environment variables", file=sys.stderr)
        print(f"POSTGRES_URI: {'✓' if postgres_uri else '✗'}")
        print(f"USER_ID: {'✓' if user_id else '✗'}")
        print(f"MODEL_NAME: {'✓' if base_model else '✗'}")
        print(f"ADAPTER_VERSION: {'✓' if adapter_name else '✗'}")
        sys.exit(1)
    
    output_base = os.environ.get('OUTPUT_PATH', '/workspace/adapters')
    output_dir = f"{output_base}/{user_id}/{adapter_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        metadata = train_complete_lora(
            postgres_uri=postgres_uri,
            user_id=user_id,
            base_model=base_model,
            adapter_name=adapter_name,
            output_dir=output_dir,
        )
        
        print("===METADATA_START===")
        print(json.dumps(metadata))
        print("===METADATA_END===")
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)