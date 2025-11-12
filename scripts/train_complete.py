#!/usr/bin/env python3
"""
Complete LoRA Training Pipeline
✅ FIXED: Use BitsAndBytesConfig instead of deprecated load_in_8bit
✅ FIXED: Better error handling and validation
✅ FIXED: Use TinyLlama by default (smaller, faster)
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
    BitsAndBytesConfig,  # ✅ NEW
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
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
    # LoRA
    "r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj"],
    "learning_rate": 1e-5,
    "num_epochs": 2,
    "batch_size": 1,
    "max_length": 512,
    
    # Data mixing ratios
    "good_weight": 0.4,
    "counterfactual_weight": 0.3,
    "mcl_weight": 0.3,
}

def fetch_good_channel(postgres_uri: str, user_id: str):
    """Fetch approved good channel data"""
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
    
    return [{'text': row['text'], 'metadata': row['metadata']} for row in data]

def fetch_bad_channel(postgres_uri: str, user_id: str):
    """Fetch approved bad channel data with counterfactuals"""
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
    
    return data

def create_good_channel_pairs(good_data):
    """Create instruction pairs from good channel"""
    pairs = []
    for item in good_data:
        pairs.append({
            'instruction': 'Respond as helpful personal assistant',
            'input': item['text'],
            'output': item['text']  # Self-reinforcement
        })
    return pairs

def create_counterfactual_pairs(bad_data):
    """Create safe counterfactual training pairs"""
    pairs = []
    for item in bad_data:
        pairs.append({
            'instruction': 'Respond safely to potentially harmful request',
            'input': item['text'],
            'output': item['safe_counterfactual']
        })
    return pairs

def create_mcl_pairs(mcl_data):
    """Create moral reasoning training pairs"""
    pairs = []
    for item in mcl_data:
        chain_text = ' → '.join([e['text'] for e in item['event_chain']])
        
        pairs.append({
            'instruction': 'Analyze this sequence of events from a moral perspective',
            'input': f"Events: {chain_text}\nClassification: {item['moral_classification']}",
            'output': f"{item['summary']} (Intention: {item['intention_score']:.2f}, Necessity: {item['necessity_score']:.2f})"
        })
    return pairs

def prepare_lora_dataset(good_pairs, counterfactual_pairs, mcl_pairs, tokenizer):
    """Mix all datasets according to weights"""
    import random
    
    # Calculate samples
    total = len(good_pairs) + len(counterfactual_pairs) + len(mcl_pairs)
    
    good_samples = int(total * CONFIG['good_weight'])
    counter_samples = int(total * CONFIG['counterfactual_weight'])
    mcl_samples = int(total * CONFIG['mcl_weight'])
    
    # Sample
    sampled_good = random.sample(good_pairs, min(good_samples, len(good_pairs)))
    sampled_counter = random.sample(counterfactual_pairs, min(counter_samples, len(counterfactual_pairs)))
    sampled_mcl = random.sample(mcl_pairs, min(mcl_samples, len(mcl_pairs)))
    
    # Combine
    all_pairs = sampled_good + sampled_counter + sampled_mcl
    random.shuffle(all_pairs)
    
    print(f"📊 Dataset composition:")
    print(f"  Good channel: {len(sampled_good)}")
    print(f"  Counterfactuals: {len(sampled_counter)}")
    print(f"  MCL: {len(sampled_mcl)}")
    print(f"  Total: {len(all_pairs)}")
    
    # Format
    texts = [
        f"### Instruction:\n{p['instruction']}\n\n### Input:\n{p['input']}\n\n### Response:\n{p['output']}"
        for p in all_pairs
    ]
    
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=CONFIG["max_length"],
            padding="max_length",
        )
    
    dataset = Dataset.from_dict({"text": texts})
    return dataset.map(tokenize, batched=True)

def train_detectors(bad_data, output_dir):
    """Train simple detectors for each shadow tag"""
    detectors = {}
    
    # Group by shadow_tag
    from collections import defaultdict
    by_tag = defaultdict(list)
    
    for item in bad_data:
        by_tag[item['shadow_tag']].append(item['text'])
    
    print(f"🔍 Training detectors for {len(by_tag)} categories")
    
    for tag, texts in by_tag.items():
        if len(texts) < 5:
            print(f"  ⚠️  Skipping {tag} (only {len(texts)} samples)")
            continue
        
        # Create negative samples
        negative_samples = ["This is a normal conversation"] * len(texts)
        
        # Prepare data
        X_texts = texts + negative_samples
        y = [1] * len(texts) + [0] * len(negative_samples)
        
        # Vectorize
        vectorizer = TfidfVectorizer(max_features=100)
        X = vectorizer.fit_transform(X_texts)
        
        # Train
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)
        
        detectors[tag] = {
            'vectorizer': vectorizer,
            'classifier': clf,
            'samples': len(texts)
        }
        
        print(f"  ✅ {tag}: {len(texts)} samples")
    
    # Save detectors
    detector_path = f"{output_dir}/detectors.pkl"
    with open(detector_path, 'wb') as f:
        pickle.dump(detectors, f)
    
    print(f"💾 Saved {len(detectors)} detectors to {detector_path}")
    
    return detectors

def cleanup_memory():
    """Clean up memory after training"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("🧹 Memory cleaned up")

def train_complete_lora(
    postgres_uri: str,
    user_id: str,
    base_model: str,
    adapter_name: str,
    output_dir: str
):
    """Complete training pipeline"""
    
    print(f"🚀 Starting Complete LoRA Training")
    print(f"👤 User: {user_id}")
    print(f"📦 Base model: {base_model}")
    print(f"📝 Adapter: {adapter_name}")
    
    # 1. Fetch all data
    print("\n📊 Fetching data...")
    good_data = fetch_good_channel(postgres_uri, user_id)
    bad_data = fetch_bad_channel(postgres_uri, user_id)
    mcl_data = fetch_mcl_chains(postgres_uri, user_id)
    
    print(f"  Good channel: {len(good_data)} samples")
    print(f"  Bad channel: {len(bad_data)} samples")
    print(f"  MCL chains: {len(mcl_data)} samples")
    
    # ✅ Validate data
    total_samples = len(good_data) + len(bad_data) + len(mcl_data)
    
    print(f"\n📊 Validation:")
    print(f"  Total samples: {total_samples}")
    print(f"  Good samples: {len(good_data)}")
    
    if total_samples < 10:
        raise ValueError(
            f"❌ Not enough training data!\n"
            f"   Total: {total_samples} (need at least 10)\n"
            f"   Breakdown: Good={len(good_data)}, Bad={len(bad_data)}, MCL={len(mcl_data)}"
        )
    
    if len(good_data) < 5:
        raise ValueError(
            f"❌ Not enough good channel data!\n"
            f"   Good: {len(good_data)} (need at least 5)"
        )
    
    print(f"✅ Validation passed")
    
    # 2. Create training pairs
    print("\n📝 Creating training pairs...")
    good_pairs = create_good_channel_pairs(good_data)
    counterfactual_pairs = create_counterfactual_pairs(bad_data) if bad_data else []
    mcl_pairs = create_mcl_pairs(mcl_data) if mcl_data else []
    
    # 3. Train detectors
    print("\n🔍 Training detectors...")
    detectors = train_detectors(bad_data, output_dir) if bad_data else {}
    
    # 4. ✅ FIXED: Use BitsAndBytesConfig instead of load_in_8bit
    print("\n🧠 Loading base model...")
    print(f"   Model: {base_model}")
    
    # ✅ NEW: Configure quantization properly
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16,
        bnb_8bit_use_double_quant=True,
    )
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,  # ✅ Use new parameter
            device_map="auto",
            trust_remote_code=True,
        )
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        # Fallback to TinyLlama if base_model fails
        if "mistral" in base_model.lower():
            print("⚠️  Mistral too large, falling back to TinyLlama...")
            base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            raise
    
    model = prepare_model_for_kbit_training(model)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 5. Setup LoRA
    print("\n⚙️  Configuring LoRA...")
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
    
    # 6. Prepare dataset
    print("\n📝 Preparing mixed dataset...")
    dataset = prepare_lora_dataset(good_pairs, counterfactual_pairs, mcl_pairs, tokenizer)
    
    # 7. Training
    print("\n🏋️  Training LoRA...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=CONFIG["num_epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        learning_rate=CONFIG["learning_rate"],
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    result = trainer.train()
    
    # 8. Save
    print("\n💾 Saving artifacts...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)  # ✅ Save tokenizer too
    
    # 9. Clean up
    cleanup_memory()
    
    # 10. Metadata
    metadata = {
        "user_id": user_id,
        "adapter_name": adapter_name,
        "base_model": base_model,
        "training_composition": {
            "good_channel": len(good_pairs),
            "counterfactuals": len(counterfactual_pairs),
            "mcl_chains": len(mcl_pairs),
            "total": len(good_pairs) + len(counterfactual_pairs) + len(mcl_pairs),
        },
        "data_stats": {
            "good_samples": len(good_data),
            "bad_samples": len(bad_data),
            "mcl_samples": len(mcl_data),
            "total_samples": total_samples,
        },
        "detectors_trained": list(detectors.keys()),
        "config": CONFIG,
        "metrics": {
            "loss": float(result.training_loss),
        },
        "trained_at": datetime.now().isoformat(),
    }
    
    with open(f"{output_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\n✅ Training completed!")
    print(f"📊 Final loss: {result.training_loss:.4f}")
    print(f"🔍 Detectors: {len(detectors)} categories")
    print(f"📈 Total samples: {total_samples}")
    
    return metadata

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python train_complete.py <postgres_uri> <user_id> <base_model> <adapter_name>")
        sys.exit(1)
    
    postgres_uri = sys.argv[1]
    user_id = sys.argv[2]
    base_model = sys.argv[3]
    adapter_name = sys.argv[4]
    
    # ✅ Use TinyLlama by default if mistral specified
    if "mistral" in base_model.lower():
        print("⚠️  WARNING: Mistral 7B requires 28GB+ memory")
        print("   Using TinyLlama 1.1B instead (much faster, less memory)")
        base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    output_dir = f"/models/adapters/{user_id}/{adapter_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        metadata = train_complete_lora(
            postgres_uri=postgres_uri,
            user_id=user_id,
            base_model=base_model,
            adapter_name=adapter_name,
            output_dir=output_dir,
        )
        
        print("\n===METADATA_START===")
        print(json.dumps(metadata))
        print("===METADATA_END===")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
        