Private Score : 155

Public Score : 168

# ========================= Continue Training NLLB LoRA từ checkpoint-4750 =========================
# Train tiếp từ checkpoint đã có với dataset từ CSV files
# ===================================================================================================

import os
import torch
import pandas as pd
from glob import glob
from collections import defaultdict
import re
import gc
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments
)
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import Seq2SeqTrainer
from typing import Any
import datasets

# ============================================================
# ⚙️ CONFIG
# ============================================================
BASE_DIR = "/kaggle/input/checkpoint-facebook-nllb-200-distilled-1-3b/nllb-200-distilled-1.3B"
ADAPTER_DIR = "/kaggle/input/checkpoint-facebook-nllb-200-distilled-1-3b/nllb-lora/checkpoint-4750"
COMP_DIR = "/kaggle/input/mm-lo-so-2025"  # Thư mục chứa CSV training data
WORK_DIR = "/kaggle/working"
OUTPUT_DIR = f"{WORK_DIR}/nllb-lora-adapter"
FINAL_ADAPTER = f"{WORK_DIR}/nllb-lora-adapter"

# Training config
CONTINUE_EPOCHS = 1  # Train thêm 2 epochs nữa
BATCH_SIZE = 4
GRAD_ACCUM = 8
LEARNING_RATE = 1e-4  # Giảm LR một chút vì đã train 1 epoch rồi
SAVE_STEPS = 2000
MAX_LEN = 256

print("=" * 80)
print("🔄 CONTINUE TRAINING NLLB LoRA FROM CHECKPOINT-4750")
print("=" * 80)

# ============================================================
# 🛠️ UTILITIES
# ============================================================
_WS_RE = re.compile(r"\s+")

def normalize_space(s: str) -> str:
    return _WS_RE.sub(" ", str(s)).strip()

def preprocess_text_for_lang(text, lang):
    """Preprocess with script handling"""
    import unicodedata
    text = str(text)
    if lang == "Santali":
        text = unicodedata.normalize('NFC', text)
    return normalize_space(text)

# ============================================================
# 🌐 LANGUAGE MAP
# ============================================================
NLLB_LANG_CODE = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Bhilli": "bhi_Deva",
    "Gondi": "gon_Deva",
    "Mundari": "unr_Deva",
    "Santali": "sat_Olck"
}

OFFICIAL_LANGS = {"Bhilli", "Hindi", "Mundari", "Gondi", "English", "Santali"}
SUB_LANG_CANON = {
    "bhili": "Bhilli", "bhilli": "Bhilli",
    "hindi": "Hindi", "mundari": "Mundari", "gondi": "Gondi",
    "english": "English", "santali": "Santali"
}

def canon_label(lang: str) -> str:
    k = lang.strip().lower()
    return SUB_LANG_CANON.get(k, lang)

# ============================================================
# 📦 LOAD TOKENIZER & MODEL (FIXED ORDER)
# ============================================================
print("\n📦 LOADING MODEL WITH PROPER ORDER")
print("=" * 80)

# STEP 1: Load tokenizer từ ADAPTER (có vocab đã extended)
print("\n[1/4] 🔄 Loading tokenizer from adapter (extended vocab)...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, use_fast=True)
print(f"      ✅ Tokenizer vocab size: {len(tokenizer):,}")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    print(f"      ⚙️  Set pad_token to: {tokenizer.pad_token}")

if not hasattr(tokenizer, "lang_code_to_id"):
    print("      ⚙️  Building lang_code_to_id mapping...")
    tokenizer.lang_code_to_id = {
        code: tokenizer.convert_tokens_to_ids(code)
        for code in NLLB_LANG_CODE.values()
    }
    tokenizer.id_to_lang_code = {v: k for k, v in tokenizer.lang_code_to_id.items()}
    print(f"      ✅ Added {len(tokenizer.lang_code_to_id)} language codes")

# STEP 2: Load base model
print("\n[2/4] 🔄 Loading base NLLB model...")
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_DIR,
    torch_dtype=torch.float16,
    device_map="auto"
)
original_vocab_size = base_model.model.shared.weight.shape[0]
print(f"      ✅ Base model original vocab size: {original_vocab_size:,}")

# STEP 3: Resize embeddings để match tokenizer (CRITICAL!)
print("\n[3/4] 🔧 Resizing model embeddings to match tokenizer...")
if len(tokenizer) != original_vocab_size:
    print(f"      ⚠️  Size mismatch: {original_vocab_size:,} → {len(tokenizer):,}")
    print(f"      🔄 Adding {len(tokenizer) - original_vocab_size:,} new embeddings...")
    base_model.resize_token_embeddings(len(tokenizer))
    print(f"      ✅ Resized to: {base_model.model.shared.weight.shape[0]:,}")
else:
    print(f"      ℹ️  No resize needed")

# STEP 4: Load LoRA adapter từ checkpoint
print("\n[4/4] 🔄 Loading LoRA adapter from checkpoint-4750...")
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_DIR,
    torch_dtype=torch.float16,
    is_trainable=True  # IMPORTANT: Enable training
)
print("      ✅ LoRA adapter loaded successfully!")

# ============================================================
# 🔧 MODEL CONFIGURATION FOR TRAINING
# ============================================================
print("\n🔧 Configuring model for training...")

model.config.use_cache = False

if hasattr(model, "gradient_checkpointing_enable"):
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("   ✅ Gradient checkpointing enabled")
    except TypeError:
        model.gradient_checkpointing_enable()

if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()
    print("   ✅ Input gradients enabled")

# Ensure gradients for embeddings
def _make_inputs_require_grad(module, inp, out):
    if isinstance(out, torch.Tensor):
        out.requires_grad_(True)

try:
    model.get_input_embeddings().register_forward_hook(_make_inputs_require_grad)
    print("   ✅ Embedding gradients hook registered")
except Exception as e:
    print(f"   ⚠️  Hook registration failed: {e}")

try:
    model.print_trainable_parameters()
except Exception:
    pass

print("\n✅ MODEL READY FOR TRAINING")
print("=" * 80)

# ============================================================
# 📊 LOAD TRAINING DATA FROM CSV
# ============================================================
print("\n📊 Loading training data from CSV files...")

def read_train_pairs(csv_path):
    df = pd.read_csv(csv_path)
    cols = [c.strip() for c in df.columns]
    lang_cols = [c for c in cols if c.strip().lower() in SUB_LANG_CANON]
    if len(lang_cols) < 2:
        lang_cols = cols[:2]
    src_col, tgt_col = lang_cols[0], lang_cols[1]
    src_name, tgt_name = canon_label(src_col), canon_label(tgt_col)
    pairs = []
    for s, t in zip(df[src_col].astype(str), df[tgt_col].astype(str)):
        s, t = normalize_space(s), normalize_space(t)
        if s and t:
            pairs.append((s, t))
    return src_name, tgt_name, pairs

train_files = sorted(glob(f"{COMP_DIR}/*.csv"))
train_files = [f for f in train_files if 'test' not in os.path.basename(f).lower()]
print(f"   Found {len(train_files)} training files")

bitext_by_dir = defaultdict(list)
for fp in train_files:
    try:
        s_lang, t_lang, pairs = read_train_pairs(fp)
        bitext_by_dir[(s_lang, t_lang)].extend(pairs)
        bitext_by_dir[(t_lang, s_lang)].extend([(t, s) for (s, t) in pairs])
        print(f"   ✅ {os.path.basename(fp)}: {len(pairs):,} pairs ({s_lang}↔{t_lang})")
    except Exception as e:
        print(f"   ⚠️  Skip {os.path.basename(fp)}: {e}")

total_pairs = sum(len(pairs) for pairs in bitext_by_dir.values())
print(f"\n   📊 Total: {total_pairs:,} training pairs across {len(bitext_by_dir)} directions")

# ============================================================
# 🏗️ CREATE DATASET
# ============================================================
print("\n🏗️  Creating training dataset...")

def nllb_pair_supported(src_lang, tgt_lang):
    return (src_lang in NLLB_LANG_CODE) and (tgt_lang in NLLB_LANG_CODE)

rows = []
for (s_lang, t_lang), pairs in bitext_by_dir.items():
    if not nllb_pair_supported(s_lang, t_lang):
        print(f"   ⚠️  Skipping unsupported pair: {s_lang} → {t_lang}")
        continue

    for s, t in pairs:
        s = preprocess_text_for_lang(s, s_lang)
        t = preprocess_text_for_lang(t, t_lang)
        if s and t:
            rows.append({"src": s, "tgt": t, "sl": s_lang, "tl": t_lang})

print(f"   ✅ Created {len(rows):,} training examples")

if not rows:
    raise ValueError("❌ No training data available!")

# Shuffle
import random
random.seed(42)
random.shuffle(rows)

# Split train/dev (95/5)
split_idx = int(len(rows) * 0.95)
train_rows = rows[:split_idx]
dev_rows = rows[split_idx:]

print(f"   📊 Split: {len(train_rows):,} train / {len(dev_rows):,} dev")

train_df = pd.DataFrame(train_rows)
dev_df = pd.DataFrame(dev_rows)

train_ds = datasets.Dataset.from_pandas(train_df)
dev_ds = datasets.Dataset.from_pandas(dev_df) if len(dev_rows) > 0 else None

# Preprocess function
def preprocess(batch):
    input_ids, attn_mask, labels = [], [], []
    for s, t, sl, tl in zip(batch["src"], batch["tgt"], batch["sl"], batch["tl"]):
        src_code = NLLB_LANG_CODE[sl]
        tgt_code = NLLB_LANG_CODE[tl]

        tokenizer.src_lang = src_code
        tokenizer.tgt_lang = tgt_code

        enc = tokenizer(str(s), truncation=True, max_length=MAX_LEN)
        lab = tokenizer(text_target=str(t), truncation=True, max_length=MAX_LEN)

        input_ids.append(enc["input_ids"])
        attn_mask.append(enc["attention_mask"])
        labels.append(lab["input_ids"])

    return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}

print("\n   🔄 Preprocessing dataset...")
train_ds = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
if dev_ds is not None:
    dev_ds = dev_ds.map(preprocess, batched=True, remove_columns=dev_ds.column_names)

print(f"   ✅ Dataset ready: {len(train_ds):,} train samples")

# ============================================================
# 🎯 CUSTOM TRAINER
# ============================================================
class LossTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs: bool=False, **kwargs: Any):
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else getattr(outputs, "loss", None)
        if loss is None:
            loss = outputs[0]
        return (loss, outputs) if return_outputs else loss

# ============================================================
# ⚙️ TRAINING ARGUMENTS
# ============================================================
print("\n⚙️  Setting up training arguments...")

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    padding="longest",
    label_pad_token_id=-100
)

# Check for existing checkpoints in output dir
checkpoint_dirs = sorted(glob(f"{OUTPUT_DIR}/checkpoint-*"))
resume_checkpoint = checkpoint_dirs[-1] if checkpoint_dirs else None

if resume_checkpoint:
    print(f"   🔄 Found checkpoint to resume from: {resume_checkpoint}")

args = Seq2SeqTrainingArguments(
    output_dir=f"{WORK_DIR}/nllb-lora",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    num_train_epochs=CONTINUE_EPOCHS,  # Train thêm N epochs
    warmup_ratio=0.03,
    fp16=True,
    logging_steps=50,
    save_steps=SAVE_STEPS,
    eval_strategy="no",  # Disable eval để tiết kiệm memory
    save_total_limit=2,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    remove_unused_columns=False,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    report_to="none",
    load_best_model_at_end=False,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    prediction_loss_only=True,
)

print(f"\n   📋 Training Configuration:")
print(f"      Batch size: {BATCH_SIZE}")
print(f"      Gradient accumulation: {GRAD_ACCUM}")
print(f"      Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
print(f"      Learning rate: {LEARNING_RATE}")
print(f"      Additional epochs: {CONTINUE_EPOCHS}")
print(f"      Save every: {SAVE_STEPS} steps")
print(f"      Max length: {MAX_LEN}")

trainer = LossTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=None,  # No eval to save memory
    data_collator=data_collator,
    compute_metrics=None,
)

# ============================================================
# 🚀 START TRAINING
# ============================================================
print("\n" + "=" * 80)
print("🚀 STARTING CONTINUED TRAINING")
if resume_checkpoint:
    print(f"   Resuming from: {resume_checkpoint}")
else:
    print(f"   Continuing from checkpoint-4750")
print("=" * 80 + "\n")

torch.cuda.empty_cache()
gc.collect()

try:
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    print("\n✅ Training completed successfully!")
except KeyboardInterrupt:
    print("\n⚠️  Training interrupted by user")
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    raise

# ============================================================
# 💾 SAVE FINAL MODEL
# ============================================================
print("\n💾 Saving final adapter...")
torch.cuda.empty_cache()

try:
    model.save_pretrained(FINAL_ADAPTER)
    tokenizer.save_pretrained(FINAL_ADAPTER)
    print(f"   ✅ Saved to: {FINAL_ADAPTER}")
except Exception as e:
    print(f"   ❌ Save failed: {e}")

# Save training log
if hasattr(trainer, 'state') and hasattr(trainer.state, 'log_history'):
    import json
    log_path = f"{WORK_DIR}/training_log.json"
    try:
        with open(log_path, "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)
        print(f"   📊 Training log: {log_path}")
    except Exception as e:
        print(f"   ⚠️  Log save failed: {e}")

# ============================================================
# 🎉 SUMMARY
# ============================================================
print("\n" + "=" * 80)
print("🎉 TRAINING COMPLETE!")
print("=" * 80)
print(f"   📁 Final adapter: {FINAL_ADAPTER}")
print(f"   📁 Checkpoints: {OUTPUT_DIR}")
print(f"   📊 Training samples: {len(train_ds):,}")
if dev_ds:
    print(f"   📊 Dev samples: {len(dev_ds):,}")
print("\n   Next steps:")
print("   1. Use the final adapter for inference")
print("   2. Or continue training by running this script again")
print("=" * 80 + "\n")
✅ Resized to: 256,404

[4/4] 🔄 Loading LoRA adapter from checkpoint-4750...
      ✅ LoRA adapter loaded successfully!

🔧 Configuring model for training...
   ✅ Gradient checkpointing enabled
   ✅ Input gradients enabled
   ✅ Embedding gradients hook registered
trainable params: 21,233,664 || all params: 1,392,074,752 || trainable%: 1.5253

✅ MODEL READY FOR TRAINING
================================================================================

📊 Loading training data from CSV files...
   Found 4 training files
   ✅ bhili-train.csv: 20,000 pairs (Hindi↔Bhilli)
   ✅ gondi-train.csv: 20,000 pairs (Hindi↔Gondi)
   ✅ mundari-train.csv: 20,000 pairs (Hindi↔Mundari)
   ✅ santali-train.csv: 20,000 pairs (English↔Santali)

   📊 Total: 160,000 training pairs across 8 directions

🏗️  Creating training dataset...
   ✅ Created 160,000 training examples
   📊 Split: 152,000 train / 8,000 dev

   🔄 Preprocessing dataset...
✅ Dataset ready: 152,000 train samples

⚙️  Setting up training arguments...

   📋 Training Configuration:
      Batch size: 4
      Gradient accumulation: 8
      Effective batch size: 32
      Learning rate: 0.0001
      Additional epochs: 1
      Save every: 2000 steps
      Max length: 256

================================================================================
🚀 STARTING CONTINUED TRAINING
   Continuing from checkpoint-4750
================================================================================

Step	Training Loss
50	2.746400
100	2.672600
150	2.715200
200	2.658600
250	2.602200
300	2.584800
350	2.678100
400	2.614800
450	2.674200
500	2.676500
550	2.612600
600	2.648700
650	2.636000
700	2.570900
750	2.608900
800	2.593200
850	2.583100
900	2.614500
950	2.612300
1000	2.583300
1050	2.619200
1100	2.593000
1150	2.660100
1200	2.578500
1250	2.510700
1300	2.526900
1350	2.565300
1400	2.562300
1450	2.505800
1500	2.619600
1550	2.644700
1600	2.534500
1650	2.544100
1700	2.565900
1750	2.576900
1800	2.571500
1850	2.567500
1900	2.573300
1950	2.516700
2000	2.629100
2050	2.501700
2100	2.586000
2150	2.619000
2200	2.529100
2250	2.554200
2300	2.604000
2350	2.564000
2400	2.592300
2450	2.586500
2500	2.572900
2550	2.523100
2600	2.579600
2650	2.537800
2700	2.575900
2750	2.610600
2800	2.606300
2850	2.551000
2900	2.562600
2950	2.623000
3000	2.492900
3050	2.591600
3100	2.539800
3150	2.549300
3200	2.544100
3250	2.480600
3300	2.557400
3350	2.585400
3400	2.535400
3450	2.546500
3500	2.529100
3550	2.563400
3600	2.528500
3650	2.581200
3700	2.576400
3750	2.548600
3800	2.618900
3850	2.645200
3900	2.511100
3950	2.549500
4000	2.623400
4050	2.579900
4100	2.563600
4150	2.595500
4200	2.567300
4250	2.578800
4300	2.596000
4350	2.596800
4400	2.651600
4450	2.615200
4500	2.571100
4550	2.663100
4600	2.575600
4650	2.576300
4700	2.586100
4750	2.660200
/usr/local/lib/python3.11/dist-packages/peft/utils/save_and_load.py:238: UserWarning: Could not find a config file in /kaggle/working/nllb-200-distilled-1.3B - will assume that the vocabulary was not modified.
  warnings.warn(
   ✅ Saved to: /kaggle/working/nllb-lora-adapter
   📊 Training log: /kaggle/working/training_log.json

================================================================================
🎉 TRAINING COMPLETE!
================================================================================
   📁 Final adapter: /kaggle/working/nllb-lora-adapter
   📁 Checkpoints: /kaggle/working/nllb-lora-adapter
   📊 Training samples: 152,000
   📊 Dev samples: 8,000

   Next steps:
   1. Use the final adapter for inference
   2. Or continue training by running this script again
================================================================================