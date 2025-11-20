Public Score :  171.63542

Private Score : 161.09653


# ========================= MMLoSo 2025 – NLLB LoRA + Dice Fallback (Kaggle T4-friendly, FULL AUTO, FIXED) ====================
#  - FIX: LossTrainer.compute_loss(**kwargs) + keep tensor with grad.
#  - FIX: gradient_checkpointing use_reentrant=False + embedding forward hook to ensure requires_grad with PEFT.
#  - FIX: remove_unused_columns=False để Trainer không lọc mất 'labels' khi dùng PeftModel.
#  - KEEP: use_cache=False + gradient_checkpointing_enable + enable_input_require_grads.
# ============================================================================================================================

import os, re, gc, sys, socket, subprocess, time
import pandas as pd
from collections import defaultdict
from glob import glob
from heapq import nlargest

# -------------------- Kaggle / paths --------------------
INPUT_DIR = "/kaggle/input"
WORK_DIR  = "/kaggle/working"
cand_dirs = [p for p in glob(f"{INPUT_DIR}/*") if os.path.isdir(p)]
COMP_DIR = None
for d in cand_dirs:
    if any(os.path.basename(fp).lower().startswith(("bhili","bhiili","gondi","mundari","santali"))
           for fp in glob(f"{d}/*.csv")):
        COMP_DIR = d
        break
if COMP_DIR is None:
    COMP_DIR = "/kaggle/input/mm-lo-so-2025"
print("Using competition directory:", COMP_DIR)

# -------------------- Utilities --------------------
_WS_RE   = re.compile(r"\s+")
_PUNC_RE = re.compile(r"([.,!?;:()\[\]{}\"'“”‘’।|/\\\-])")
def normalize_space(s: str) -> str:
    return _WS_RE.sub(" ", str(s)).strip()
def simple_tokenize(s: str):
    s = normalize_space(s)
    s = _PUNC_RE.sub(r" \1 ", s)
    s = normalize_space(s)
    return s.split()
def detokenize(tokens):
    out = []
    for i, t in enumerate(tokens):
        if i > 0 and t in {".", ",", "!", "?", ";", ":", ")", "”", "’", "।"}:
            out[-1] = out[-1] + t
        elif t in {"(", "“", "‘"} and len(out) > 0:
            out.append(t)
        else:
            out.append(t)
    txt = " ".join(out)
    txt = txt.replace("( ", "(").replace(" )", ")")
    txt = txt.replace("“ ", "“").replace(" ”", "”")
    txt = txt.replace("‘ ", "‘").replace(" ’", "’")
    return normalize_space(txt)

OFFICIAL_LANGS = {"Bhilli","Hindi","Mundari","Gondi","English","Santali"}
SUB_LANG_CANON = {
    "bhili":"Bhilli","bhilli":"Bhilli",
    "hindi":"Hindi","mundari":"Mundari","gondi":"Gondi",
    "english":"English","santali":"Santali"
}
def canon_label(lang: str) -> str:
    k = lang.strip().lower()
    return SUB_LANG_CANON.get(k, lang)

# -------------------- Load training CSVs --------------------
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
print("Train files found:", [os.path.basename(x) for x in train_files])

bitext_by_dir = defaultdict(list)
for fp in train_files:
    try:
        s_lang, t_lang, pairs = read_train_pairs(fp)
        bitext_by_dir[(s_lang, t_lang)].extend(pairs)
        bitext_by_dir[(t_lang, s_lang)].extend([(t, s) for (s, t) in pairs])
        print(f"Loaded {len(pairs):,} pairs: {s_lang} -> {t_lang}")
    except Exception as e:
        print("Skip", fp, "due to", e)

# -------------------- Đọc test để biết hướng cần thiết --------------------
test_path = os.path.join(COMP_DIR, "test.csv")
if not os.path.exists(test_path):
    cand = [p for p in glob(f"{COMP_DIR}/*.csv") if os.path.basename(p).lower().startswith("test")]
    if cand: test_path = cand[0]
test_df_preview = pd.read_csv(test_path)
rename_map = {}
for c in test_df_preview.columns:
    cn = c.strip()
    if cn.lower().replace("_"," ") == "row id":          rename_map[c] = "Row ID"
    elif cn.lower() == "source lang":                    rename_map[c] = "Source Lang"
    elif cn.lower() == "source sentence":                rename_map[c] = "Source Sentence"
    elif cn.lower() == "target lang":                    rename_map[c] = "Target Lang"
    elif cn.lower() == "target sentence":                rename_map[c] = "Target Sentence"
test_df_preview = test_df_preview.rename(columns=rename_map)

needed_dirs = set()
for _, r in test_df_preview.iterrows():
    sl = canon_label(str(r["Source Lang"]))
    tl = canon_label(str(r["Target Lang"]))
    needed_dirs.add((sl, tl))

# -------------------- Dice lexicon --------------------
def build_dice_lexicon(pairs, min_count=2, top_k=3):
    src_docs_contain = defaultdict(set)
    tgt_docs_contain = defaultdict(set)
    co_by_src = defaultdict(lambda: defaultdict(int))
    for i, (s, t) in enumerate(pairs):
        s_tokens = set(simple_tokenize(s.lower()))
        t_tokens = set(simple_tokenize(t.lower()))
        for sw in s_tokens: src_docs_contain[sw].add(i)
        for tw in t_tokens: tgt_docs_contain[tw].add(i)
        for sw in s_tokens:
            for tw in t_tokens:
                co_by_src[sw][tw] += 1
    src_count = {w: len(idx) for w, idx in src_docs_contain.items()}
    tgt_count = {w: len(idx) for w, idx in tgt_docs_contain.items()}
    lex = {}
    for sw, tgt_map in co_by_src.items():
        sc = src_count.get(sw, 0)
        if sc < min_count:
            continue
        cands = []
        for tw, cst in tgt_map.items():
            tc = tgt_count.get(tw, 0)
            if tc < min_count:
                continue
            dice = 2.0 * cst / (sc + tc)
            cands.append((dice, tw))
        if cands:
            top = nlargest(top_k, cands, key=lambda x: x[0])
            lex[sw] = [tw for _, tw in top]
    return lex

lexicons = {}
for (s_lang, t_lang), pairs in bitext_by_dir.items():
    if (s_lang, t_lang) not in needed_dirs: continue
    if not pairs: continue
    print(f"Building lexicon {s_lang} -> {t_lang} on {len(pairs):,} pairs ...")
    lexicons[(s_lang, t_lang)] = build_dice_lexicon(pairs, min_count=2, top_k=3)
    gc.collect()

def translate_tokens(src_tokens, lex):
    return [lex[w][0] if w in lex else w for w in src_tokens]
def dice_translate_sentence(s, src_lang, tgt_lang):
    src_lang = canon_label(src_lang); tgt_lang = canon_label(tgt_lang)
    if (src_lang, tgt_lang) not in lexicons:
        toks = simple_tokenize(s)
        return detokenize(toks) if toks else "."
    toks = simple_tokenize(s.lower())
    hyp_tokens = translate_tokens(toks, lexicons[(src_lang, tgt_lang)])
    hyp = detokenize(hyp_tokens)
    return hyp if hyp.strip() else "."

# ================== Auto load or download NLLB (HF) ==================
HF_REPO = "facebook/nllb-200-distilled-600M"
LOCAL_SCAN_ROOT = "/kaggle/input"
LOCAL_FALLBACK  = "/kaggle/working/nllb200-600m"
NEED_FILES = {"config.json", "tokenizer.json", "pytorch_model.bin"}

def _has_files(dirpath, need=NEED_FILES):
    try:
        names = set(os.listdir(dirpath))
        return need.issubset(names)
    except Exception:
        return False

def _scan_kaggle_input(root=LOCAL_SCAN_ROOT):
    cands = [p for p in glob(f"{root}/*") if os.path.isdir(p)]
    pref = []
    for d in cands:
        if _has_files(d):
            score = 0
            base = os.path.basename(d).lower()
            if "nllb" in base:  score += 2
            if "600m" in base:  score += 1
            pref.append((score, d))
    if pref:
        pref.sort(reverse=True)
        return pref[0][1]
    for d in cands:
        subs = [p for p in glob(f"{d}/*") if os.path.isdir(p)]
        for sd in subs:
            if _has_files(sd):
                return sd
    return None

def _internet_ok(host="huggingface.co", timeout=3):
    try:
        socket.setdefaulttimeout(timeout)
        socket.gethostbyname(host)
        return True
    except Exception:
        return False

def _pip_install(pkgs):
    try:
        if not _internet_ok():
            print("⛔ Internet OFF → cannot pip install:", pkgs)
            return False
        print("🌐 pip install", pkgs)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + pkgs)
        return True
    except Exception as e:
        print("⚠️ pip install failed:", e)
        return False

def _download_from_hf(repo_id, local_dir, max_retries=3):
    if _internet_ok():
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    try:
        import huggingface_hub  # noqa
    except Exception:
        if not _pip_install(["huggingface_hub>=0.22", "hf_transfer>=0.1.5"]):
            return None
    from huggingface_hub import snapshot_download
    os.makedirs(local_dir, exist_ok=True)
    token = os.environ.get("HUGGINGFACE_TOKEN", None)
    for attempt in range(1, max_retries+1):
        try:
            print(f"⬇️  snapshot_download attempt {attempt}/{max_retries} ...")
            path = snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                ignore_patterns=["*.md", "*.gitattributes"],
                token=token,
            )
            if _has_files(path):
                return path
        except Exception as e:
            print("⚠️ snapshot_download failed:", e)
            if attempt < max_retries:
                time.sleep(3 * attempt)
    return None

def get_nllb_path():
    p = _scan_kaggle_input()
    if p:
        print("🔎 Found NLLB in /kaggle/input →", p)
        return p
    if _internet_ok():
        print("🌐 Internet OK → downloading NLLB from HF to", LOCAL_FALLBACK)
        p = _download_from_hf(HF_REPO, LOCAL_FALLBACK)
        if p:
            print("✅ Downloaded NLLB to:", p)
            return p
        print("⚠️ Download attempt failed, will fallback.")
    else:
        print("⛔ Internet OFF (submit mode) → cannot download.")
    return None

# -------------------- NLLB + PEFT LoRA (bật nếu tìm thấy checkpoint) --------------------
USE_MODEL = True
nllb_dir = get_nllb_path()
if not nllb_dir:
    print("➡️ USE_MODEL=False (no checkpoint). Using Dice only.")
    USE_MODEL = False
else:
    print("✅ Using NLLB from:", nllb_dir)

NLLB_LANG_CODE = {"English": "eng_Latn","Hindi": "hin_Deva","Santali": "sat_Olck"}
def nllb_pair_supported(src_lang, tgt_lang):
    return (src_lang in NLLB_LANG_CODE) and (tgt_lang in NLLB_LANG_CODE)

nllb_models, tokenizer = {}, None
_eightbit_ok = False

if USE_MODEL:
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        import torch

        # Thử 8-bit; nếu fail → fallback FP16
        try:
            quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
            tokenizer = AutoTokenizer.from_pretrained(nllb_dir, use_fast=True)
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                nllb_dir, quantization_config=quant_cfg,
                torch_dtype=torch.float16, device_map="auto",
            )
            base_model = prepare_model_for_kbit_training(base_model)
            _eightbit_ok = True
        except Exception as e:
            print("⚠️ 8-bit init failed:", e)
            _eightbit_ok = False
            tokenizer = AutoTokenizer.from_pretrained(nllb_dir, use_fast=True)
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                nllb_dir, torch_dtype=torch.float16, device_map="auto",
            )

        # LoRA targets
        lora_cfg = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj","k_proj","v_proj","o_proj","fc1","fc2"],
            bias="none", task_type="SEQ_2_SEQ_LM"
        )
        base_model = get_peft_model(base_model, lora_cfg)

        # IMPORTANT khi bật checkpointing
        base_model.config.use_cache = False  # yêu cầu bởi HF khi checkpointing
        # dùng kwargs để tránh cảnh báo None requires_grad khi checkpoint re-entrant
        if hasattr(base_model, "gradient_checkpointing_enable"):
            try:
                base_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                base_model.gradient_checkpointing_enable()
        if hasattr(base_model, "enable_input_require_grads"):
            base_model.enable_input_require_grads()

        # Bảo đảm output embedding có grad khi checkpointing
        def _make_inputs_require_grad(module, inp, out):
            if isinstance(out, torch.Tensor):
                out.requires_grad_(True)
        try:
            base_model.get_input_embeddings().register_forward_hook(_make_inputs_require_grad)
        except Exception:
            pass

        # In case tokenizer lacks pad token
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = getattr(tokenizer, "eos_token", None) or tokenizer.unk_token

        try: base_model.print_trainable_parameters()
        except Exception: pass

        nllb_models["shared"] = base_model
        print("✅ NLLB ready (with LoRA). 8-bit:", _eightbit_ok)
        if hasattr(base_model, "hf_device_map"): print("Device map:", base_model.hf_device_map)
    except Exception as e:
        print(f"⚠️ Không khởi tạo được NLLB model: {e} → dùng Dice.")
        USE_MODEL = False

# -------------------- Chuẩn bị dataset multi-direction --------------------
def make_multidir_dataset(bitext_by_dir, max_per_dir=30000, max_len=256):
    try:
        import datasets
    except Exception as e:
        print("⚠️ datasets chưa có, bỏ qua finetune:", e)
        return None
    rows = []
    for (s_lang, t_lang), pairs in bitext_by_dir.items():
        if not nllb_pair_supported(s_lang, t_lang): 
            continue
        take = pairs[:max_per_dir]
        for s, t in take:
            rows.append({"src": s, "tgt": t, "sl": s_lang, "tl": t_lang})
    if not rows: return None
    df = pd.DataFrame(rows)
    ds = datasets.Dataset.from_pandas(df)

    def preprocess(batch):
        input_ids, attn_mask, labels = [], [], []
        for s,t,sl,tl in zip(batch["src"], batch["tgt"], batch["sl"], batch["tl"]):
            tokenizer.src_lang = NLLB_LANG_CODE[sl]
            tokenizer.tgt_lang = NLLB_LANG_CODE[tl]
            enc = tokenizer(str(s), truncation=True, max_length=max_len)
            lab = tokenizer(text_target=str(t), truncation=True, max_length=max_len)
            input_ids.append(enc["input_ids"])
            attn_mask.append(enc["attention_mask"])
            labels.append(lab["input_ids"])
        return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}

    ds = ds.map(preprocess, batched=True, remove_columns=ds.column_names)
    return ds

# -------------------- Custom Trainer: đảm bảo loss là tensor có grad --------------------
from typing import Any
from transformers import Seq2SeqTrainer

class LossTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs: bool=False, **kwargs: Any):
        # giữ nguyên inputs; không .item() để mất grad
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else getattr(outputs, "loss", None)
        if loss is None:
            loss = outputs[0]
        return (loss, outputs) if return_outputs else loss

if USE_MODEL:
    train_ds = make_multidir_dataset(bitext_by_dir, max_per_dir=30000, max_len=256)
    if train_ds is not None and len(train_ds) > 0:
        from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments
        model = nllb_models["shared"]
        data_collator = DataCollatorForSeq2Seq(
            tokenizer, model=model, padding="longest", label_pad_token_id=-100
        )

        print("🔧 BẮT ĐẦU FINE-TUNE LoRA TRÊN NLLB… samples:", len(train_ds))
        args = Seq2SeqTrainingArguments(
            output_dir=f"{WORK_DIR}/nllb-lora",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            num_train_epochs=1,
            warmup_ratio=0.03,
            fp16=True,
            logging_steps=50,
            save_steps=1000,
            save_total_limit=1,
            gradient_checkpointing=True,   # bật phía Trainer
            gradient_checkpointing_kwargs={"use_reentrant": False},  # <<< QUAN TRỌNG
            remove_unused_columns=False,   # <<< giữ 'labels' khi dùng PEFT
            optim=("adamw_bnb_8bit" if _eightbit_ok else "adamw_torch"),
            lr_scheduler_type="cosine",
            report_to="none"
        )
        trainer = LossTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            data_collator=data_collator,
        )
        trainer.train()
        model.save_pretrained(f"{WORK_DIR}/nllb-lora-adapter")
        tokenizer.save_pretrained(f"{WORK_DIR}/nllb-lora-adapter")
        print("✅ DONE TRAIN")
    else:
        print("ℹ️ Không có hướng nào có mã NLLB để train → dùng Dice.")

# -------------------- Inference helpers --------------------
def nllb_generate(texts, src_lang, tgt_lang, max_new_tokens=128):
    if not (USE_MODEL and tokenizer and nllb_models.get("shared")): return None
    if not nllb_pair_supported(src_lang, tgt_lang): return None
    import torch
    model = nllb_models["shared"]
    src_code = NLLB_LANG_CODE[src_lang]
    tgt_code = NLLB_LANG_CODE[tgt_lang]
    tokenizer.src_lang = src_code
    enc = tokenizer(list(map(str, texts)), return_tensors="pt", padding=True, truncation=True, max_length=256)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    try:
        forced_bos_id = tokenizer.lang_code_to_id[tgt_code]
    except Exception:
        forced_bos_id = tokenizer.convert_tokens_to_ids(tgt_code)
    gen = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        num_beams=4, length_penalty=1.0,
        no_repeat_ngram_size=3,
        forced_bos_token_id=forced_bos_id
    )
    outs = tokenizer.batch_decode(gen, skip_special_tokens=True)
    return [normalize_space(o) for o in outs]

def hybrid_translate_sentence(s, src_lang, tgt_lang):
    src_lang = canon_label(src_lang); tgt_lang = canon_label(tgt_lang)
    if USE_MODEL and nllb_pair_supported(src_lang, tgt_lang):
        hyp = nllb_generate([str(s)], src_lang, tgt_lang, max_new_tokens=128)
        if hyp and normalize_space(hyp[0]) != "": return normalize_space(hyp[0])
    return dice_translate_sentence(s, src_lang, tgt_lang)

# -------------------- Inference on test.csv --------------------
test_df = pd.read_csv(test_path).rename(columns=rename_map)
required_cols = ["Row ID","Source Lang","Source Sentence","Target Lang","Target Sentence"]
for col in required_cols:
    if col not in test_df.columns:
        if col == "Target Sentence":
            test_df[col] = ""
        else:
            raise ValueError(f"Missing required column in test.csv: {col}")

preds = []
for _, row in test_df.iterrows():
    src_lang = str(row["Source Lang"]); tgt_lang = str(row["Target Lang"])
    src_sent = str(row["Source Sentence"])
    hyp = hybrid_translate_sentence(src_sent, src_lang, tgt_lang)
    preds.append(hyp if str(hyp).strip() else ".")

test_df["Target Sentence"] = pd.Series(preds).fillna(".").apply(lambda s: s if str(s).strip() else ".")
submission = test_df[required_cols].copy()
out_path = f"{WORK_DIR}/submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
try:
    display(submission.head(10))
except Exception:
    print(submission.head(10).to_string(index=False))

used_model = sum(1 for (s_lang, t_lang) in needed_dirs if nllb_pair_supported(s_lang, t_lang) and USE_MODEL)
used_dice  = sum(1 for (s_lang, t_lang) in needed_dirs if (not nllb_pair_supported(s_lang, t_lang)) or (not USE_MODEL))
print(f"Directions supported by NLLB (used): {used_model}, Dice fallback: {used_dice}")
Using competition directory: /kaggle/input/mm-lo-so-2025
Train files found: ['bhili-train.csv', 'gondi-train.csv', 'mundari-train.csv', 'santali-train.csv']
Loaded 20,000 pairs: Hindi -> Bhilli
Loaded 20,000 pairs: Hindi -> Gondi
Loaded 20,000 pairs: Hindi -> Mundari
Loaded 20,000 pairs: English -> Santali
Building lexicon Hindi -> Bhilli on 20,000 pairs ...
Building lexicon Bhilli -> Hindi on 20,000 pairs ...
Building lexicon Hindi -> Gondi on 20,000 pairs ...
Building lexicon Gondi -> Hindi on 20,000 pairs ...
Building lexicon Hindi -> Mundari on 20,000 pairs ...
Building lexicon Mundari -> Hindi on 20,000 pairs ...
Building lexicon English -> Santali on 20,000 pairs ...
Building lexicon Santali -> English on 20,000 pairs ...
🌐 Internet OK → downloading NLLB from HF to /kaggle/working/nllb200-600m
⬇️  snapshot_download attempt 1/3 ...
config.json: 100%
 846/846 [00:00<00:00, 94.6kB/s]
generation_config.json: 100%
 189/189 [00:00<00:00, 28.1kB/s]
pytorch_model.bin: 100%
 2.46G/2.46G [00:07<00:00, 531MB/s]
sentencepiece.bpe.model: 100%
 4.85M/4.85M [00:00<00:00, 16.4MB/s]
special_tokens_map.json: 
 3.55k/? [00:00<00:00, 198kB/s]
tokenizer.json: 100%
 17.3M/17.3M [00:00<00:00, 60.5MB/s]
tokenizer_config.json: 100%
 564/564 [00:00<00:00, 23.6kB/s]
✅ Downloaded NLLB to: /kaggle/working/nllb200-600m
✅ Using NLLB from: /kaggle/working/nllb200-600m
2025-11-03 09:02:13.844773: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1762160533.987760      20 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1762160534.031175      20 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
⚠️ 8-bit init failed: Using `bitsandbytes` 8-bit quantization requires the latest version of bitsandbytes: `pip install -U bitsandbytes`
trainable params: 7,471,104 || all params: 622,544,896 || trainable%: 1.2001
✅ NLLB ready (with LoRA). 8-bit: False
Device map: {'model.shared': 0, 'lm_head': 0, 'model.decoder.embed_tokens': 0, 'model.encoder.embed_tokens': 0, 'model.encoder.embed_positions': 0, 'model.encoder.layers.0': 0, 'model.encoder.layers.1': 0, 'model.encoder.layers.2': 0, 'model.encoder.layers.3': 0, 'model.encoder.layers.4': 1, 'model.encoder.layers.5': 1, 'model.encoder.layers.6': 1, 'model.encoder.layers.7': 1, 'model.encoder.layers.8': 1, 'model.encoder.layers.9': 1, 'model.encoder.layers.10': 1, 'model.encoder.layers.11': 1, 'model.encoder.layer_norm': 1, 'model.decoder.embed_positions': 1, 'model.decoder.layers': 1, 'model.decoder.layer_norm': 1}
Map: 100%
 40000/40000 [00:18<00:00, 2178.79 examples/s]
No label_names provided for model class `PeftModelForSeq2SeqLM`. Since `PeftModel` hides base models input arguments, if label_names is not given, label_names can't be set automatically within `Trainer`. Note that empty label_names list will be used instead.
🔧 BẮT ĐẦU FINE-TUNE LoRA TRÊN NLLB… samples: 40000
 [1250/1250 57:29, Epoch 1/1]
Step	Training Loss
50	3.176100
100	2.167900
150	2.041500
200	1.969200
250	1.962800
300	1.912800
350	1.880700
400	1.899100
450	1.868500
500	1.845700
550	1.855200
600	1.844700
650	1.829200
700	1.808700
750	1.803300
800	1.810800
850	1.803400
900	1.788500
950	1.809100
1000	1.762200
1050	1.791000
1100	1.774400
1150	1.782100
1200	1.809900
1250	1.769700
✅ DONE TRAIN
/usr/local/lib/python3.11/dist-packages/transformers/generation/utils.py:1739: UserWarning: You have modified the pretrained model configuration to control generation. This is a deprecated strategy to control generation and will be removed in v5. Please use and modify the model generation configuration (see https://huggingface.co/docs/transformers/generation_strategies#default-text-generation-configuration )
  warnings.warn(
Wrote: /kaggle/working/submission.csv
Row ID	Source Lang	Source Sentence	Target Lang	Target Sentence
0	54334	Hindi	उन्होंने कहा कि 2014 के बाद, इस परियोजना को प्...	Bhilli	कि कि कि 2014। बाद। इना परियोजना ने प्रधानमंत्...
1	87641	Hindi	वित्तीय कठिनाइयों को हल करने में सहायक होने के...	Bhilli	वित्तीय मनोविनोद ने हल करवा मा सहायक थावा। हात...
2	32543	Hindi	मेरा सुझाव है कि हमारे सक्रिय दृष्टिकोण, नीतिय...	Bhilli	मारू सुझाव से कि हमारा सक्रिय दृष्टिकोण। नीतिय...
3	26313	Hindi	श्री मोदी ने कहा यह अटल जी ही थे जिन्होंने देश...	Bhilli	श्री मोदी ने कि यो अटल जी ही हता जिहुने देह नी...
4	83303	Hindi	उत्सवादि मनाने के उपलक्ष्य में सुरापान करना सा...	Bhilli	उत्सवादि मनावा। उपलक्ष्य मा सुरापान करवु साधार...
5	131411	Hindi	तुम्‍हारे साथ कभी ऐसा हुआ।	Bhilli	तुमरा हाते कदी ऐवु थायो।
6	101809	Hindi	यह सत्र ग्लासगो, यूनाइटेड किंगडम में आयोजित हुआ।	Bhilli	यो सत्र ग्लासगो। यूनाइटेड किंगडम मा आयोजित थायो।
7	59328	Hindi	यह 9 मार्च 2012 को रिलीज़ हुई थी, जिसे आम तौर ...	Bhilli	यो 9 मार्च 2012 ने रिलीज थाई हती। जिने आम तौर ...
8	57205	Hindi	नियुक्ति मामलों की मंत्रिमंडलीय समिति ने भारती...	Bhilli	नियुक्ति मामलों नी मंत्रिमंडलीय समिति ने भारती...
9	103641	Hindi	भारत को और अधिक साफ-सुथरा बनाने और बेहतर स्वच्...	Bhilli	भारत ने अने वदारे साफ - सुथरो बणावा अने बेहतर ...
Directions supported by NLLB (used): 2, Dice fallback: 6