```
Public Score : 302.17705
Private Score : 172.84104


# ========================= MMLoSo 2025 – NLLB LoRA MAX-BLEU Edition (FIXED) =========================
importos,re,gc,sys,socket,subprocess,time
importpandasaspd
fromcollectionsimport defaultdict
fromglobimport glob
importtorch

# -------------------- Pre-flight: pin protobuf<5 to avoid GetPrototype error --------------------
def_internet_ok(host="pypi.org", timeout=3):
    try:
        socket.setdefaulttimeout(timeout)
        socket.gethostbyname(host)
        return True
    except Exception:
        return False

def_pip_install(pkgs):
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

try:
    importgoogle.protobuf
    frompackagingimport version
    if version.parse(google.protobuf.__version__) >= version.parse("5.0.0"):
        print(f"⚠️ protobuf={google.protobuf.__version__} is >=5 → downgrading to <5 to avoid MessageFactory issue")
        _pip_install(["protobuf<5"])
        importimportlib; importlib.invalidate_caches()
        importgoogle.protobufas_pb; print("protobuf pinned to:", _pb.__version__)
except Exception as _e:
    print("Note: protobuf pre-flight check failed/skipped:", _e)

# -------------------- Paths and Constants --------------------
INPUT_DIR = "/kaggle/input"
WORK_DIR  = "/kaggle/working"
ADAPTER_PATH = f"{WORK_DIR}/nllb-lora-adapter-final"

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
# fix quoting; escape quotes and dash properly
_PUNC_RE = re.compile(r'([\.\,\!\?\;\:\(\)\[\]\{\}"\'।\|/\\\-])')

defnormalize_space(s: str) -> str:
    return _WS_RE.sub(" ", str(s)).strip()

defsimple_tokenize(s: str):
    s = normalize_space(s)
    s = _PUNC_RE.sub(r" \1 ", s)
    s = normalize_space(s)
    return s.split()

defdetokenize(tokens):
    # robust, avoids exotic quote juggling
    txt = " ".join(tokens)
    txt = txt.replace("( ", "(").replace(" )", ")")
    txt = txt.replace(" ' ", "'").replace(" \" ", "\"")
    txt = re.sub(r"\s+([,\.!\?:;])", r"\1", txt)
    return normalize_space(txt)

OFFICIAL_LANGS = {"Bhilli","Hindi","Mundari","Gondi","English","Santali"}
SUB_LANG_CANON = {
    "bhili":"Bhilli","bhilli":"Bhilli",
    "hindi":"Hindi","mundari":"Mundari","gondi":"Gondi",
    "english":"English","santali":"Santali"
}

defcanon_label(lang: str) -> str:
    k = lang.strip().lower()
    return SUB_LANG_CANON.get(k, lang)

# -------------------- Load training CSVs --------------------
defread_train_pairs(csv_path):
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

# -------------------- Create train lookup dictionary --------------------
print("Creating train lookup dictionary for post-processing...")
train_lookup = defaultdict(dict)
for (s_lang, t_lang), pairs in bitext_by_dir.items():
    for s, t in pairs:
        train_lookup[(s_lang, t_lang)][s] = t
print(f"Lookup created. Example: Hindi->Bhilli has {len(train_lookup[('Hindi','Bhilli')]):,} unique sentences.")

# -------------------- Read test file --------------------
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

# ================== Auto load or download NLLB ==================
HF_REPO = "facebook/nllb-200-distilled-600M"
LOCAL_SCAN_ROOT = "/kaggle/input"
LOCAL_FALLBACK  = "/kaggle/working/nllb200-600m"
NEED_FILES = {"config.json", "tokenizer.json", "pytorch_model.bin"}

def_has_files(dirpath, need=NEED_FILES):
    try:
        names = set(os.listdir(dirpath))
        return need.issubset(names)
    except Exception:
        return False

def_scan_kaggle_input(root=LOCAL_SCAN_ROOT):
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

def_internet_ok2(host="huggingface.co", timeout=3):
    try:
        socket.setdefaulttimeout(timeout)
        socket.gethostbyname(host)
        return True
    except Exception:
        return False

def_download_from_hf(repo_id, local_dir, max_retries=3):
    if _internet_ok2():
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    try:
        importhuggingface_hub
    except Exception:
        if not _pip_install(["huggingface_hub>=0.22", "hf_transfer>=0.1.5"]):
            return None
    fromhuggingface_hubimport snapshot_download
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

defget_nllb_path():
    p = _scan_kaggle_input()
    if p:
        print("🔎 Found NLLB in /kaggle/input →", p)
        return p
    if _internet_ok2():
        print("🌐 Internet OK → downloading NLLB from HF to", LOCAL_FALLBACK)
        p = _download_from_hf(HF_REPO, LOCAL_FALLBACK)
        if p:
            print("✅ Downloaded NLLB to:", p)
            return p
        print("⚠️ Download attempt failed, will fallback.")
    else:
        print("⛔ Internet OFF (submit mode) → cannot download.")
    return None

# -------------------- NLLB + PEFT LoRA --------------------
USE_MODEL = True
nllb_dir = get_nllb_path()
if not nllb_dir:
    print("➡️ FATAL: NLLB model not found. Cannot proceed.")
    USE_MODEL = False
else:
    print("✅ Using NLLB from:", nllb_dir)

NLLB_LANG_CODE = {
    "English": "eng_Latn",
    "Hindi":   "hin_Deva",
    "Santali": "sat_Olck",
    "Bhilli":  "hin_Deva",  # proxy to Hindi (no direct Bhilli)
    "Gondi":   "hin_Deva",  # proxy
    "Mundari": "hin_Deva",  # proxy
}

defnllb_pair_supported(src_lang, tgt_lang):
    return (src_lang in NLLB_LANG_CODE) and (tgt_lang in NLLB_LANG_CODE)

model, tokenizer = None, None

if USE_MODEL:
    try:
        fromtransformersimport AutoModelForSeq2SeqLM, AutoTokenizer
        # SAFE: turn off bnb 8-bit path (Kaggle often lacks newest bitsandbytes/accelerate)  ── see refs.
        tokenizer = AutoTokenizer.from_pretrained(nllb_dir, use_fast=True)
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

        model = AutoModelForSeq2SeqLM.from_pretrained(
            nllb_dir, torch_dtype=torch.float16, device_map="auto",
        )

        # Load or train adapter
        frompeftimport PeftModel, LoraConfig, get_peft_model
        if os.path.exists(ADAPTER_PATH):
            print(f"✅ Found trained adapter at {ADAPTER_PATH}. Loading it...")
            model = PeftModel.from_pretrained(model, ADAPTER_PATH)
            print("Adapter loaded successfully.")
        else:
            print(f"ℹ️ No adapter found at {ADAPTER_PATH}. Starting fine-tuning process...")
            lora_cfg = LoraConfig(
                r=32, lora_alpha=64, lora_dropout=0.05,
                target_modules=["q_proj","k_proj","v_proj","o_proj","fc1","fc2"],
                bias="none", task_type="SEQ_2_SEQ_LM"
            )
            model = get_peft_model(model, lora_cfg)
            try: 
                model.print_trainable_parameters()
            except Exception: 
                pass
  
            # --- FIX: dataset creation (no as_target_tokenizer; use text_target=) ---
            importdatasets
            fromtransformersimport DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

            defmake_multidir_dataset(bitext, tkn, max_per_dir=40000, max_len=256):
                rows = []
                for (s_lang, t_lang), pairs in bitext.items():
                    if not nllb_pair_supported(s_lang, t_lang): 
                        continue
                    take = pairs[:max_per_dir]
                    for s, t in take:
                        rows.append({"src": s, "tgt": t, "sl": s_lang, "tl": t_lang})
                if not rows: 
                    return None
  
                ds = datasets.Dataset.from_pandas(pd.DataFrame(rows))

                defpreprocess(examples):
                    # process per-item to respect different lang codes in a batch
                    inputs_ids, attn_masks, labels_ids = [], [], []
                    for s, t, sl, tl in zip(examples["src"], examples["tgt"], examples["sl"], examples["tl"]):
                        src_code = NLLB_LANG_CODE[sl]
                        tgt_code = NLLB_LANG_CODE[tl]
                        tokenizer.src_lang = src_code
                        # input
                        enc = tokenizer(
                            str(s),
                            truncation=True,
                            max_length=max_len,
                            padding=False,
                        )
                        # label with text_target=
                        tokenizer.tgt_lang = tgt_code
                        lab = tokenizer(
                            text_target=[str(t)],
                            truncation=True,
                            max_length=max_len,
                            padding=False,
                        )
                        inputs_ids.append(enc["input_ids"])
                        attn_masks.append(enc["attention_mask"])
                        labels_ids.append(lab["input_ids"][0])
                    return {"input_ids": inputs_ids, "attention_mask": attn_masks, "labels": labels_ids}

                ds = ds.map(preprocess, batched=True, batch_size=128, remove_columns=ds.column_names)
                return ds

            train_ds = make_multidir_dataset(bitext_by_dir, tokenizer)
  
            if train_ds and len(train_ds) > 0:
                data_collator = DataCollatorForSeq2Seq(
                    tokenizer, 
                    model=model, 
                    padding="longest", 
                    label_pad_token_id=-100
                )

                print(f"🔧 STARTING FINE-TUNE ON {len(train_ds)} samples")
                args = Seq2SeqTrainingArguments(
                    output_dir=f"{WORK_DIR}/nllb-lora-checkpoints",
                    per_device_train_batch_size=8,
                    gradient_accumulation_steps=4,
                    learning_rate=1e-4,
                    num_train_epochs=1,
                    warmup_ratio=0.1,
                    fp16=True,
                    logging_steps=25,
                    save_strategy="no",
                    remove_unused_columns=False,
                    lr_scheduler_type="cosine",
                    report_to="none"
                )
  
                # Configure model for training
                try:
                    model.config.use_cache = False
                except Exception:
                    pass
                try:
                    if hasattr(model, "gradient_checkpointing_enable"):
                        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                except Exception:
                    pass
                try:
                    if hasattr(model, "enable_input_require_grads"):
                        model.enable_input_require_grads()
                except Exception:
                    pass
  
                trainer = Seq2SeqTrainer(
                    model=model, 
                    args=args, 
                    train_dataset=train_ds, 
                    data_collator=data_collator
                )
                trainer.train()
  
                print(f"✅ DONE TRAIN. Saving final adapter to {ADAPTER_PATH}")
                model.save_pretrained(ADAPTER_PATH)
                tokenizer.save_pretrained(ADAPTER_PATH)
  
                # Cleanup
                del trainer
                gc.collect()
                torch.cuda.empty_cache()
            else:
                print("ℹ️ No data available for training.")
  
        # Prepare for inference
        try:
            model.config.use_cache = True
        except Exception:
            pass
        model.eval()
        if hasattr(model, "hf_device_map"): 
            print("Device map:", model.hf_device_map)

    except Exception as e:
        print(f"⚠️ FATAL: Could not initialize NLLB model: {e}")
        importtraceback
        traceback.print_exc()
        USE_MODEL = False

# -------------------- Inference helpers --------------------
@torch.no_grad()
defnllb_generate(texts, src_lang, tgt_lang, max_new_tokens=256):
    if not (USE_MODEL and tokenizer and model): 
        return None
    if not nllb_pair_supported(src_lang, tgt_lang): 
        return None
  
    src_code = NLLB_LANG_CODE[src_lang]
    tgt_code = NLLB_LANG_CODE[tgt_lang]
    tokenizer.src_lang = src_code
  
    inputs = tokenizer(
        list(map(str, texts)), 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=256
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
  
    # lang_code_to_id may be missing on some older transformers – provide robust fallback
    try:
        forced_bos_id = tokenizer.lang_code_to_id[tgt_code]
    except Exception:
        forced_bos_id = tokenizer.convert_tokens_to_ids(tgt_code)

    generated_tokens = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=5,
        length_penalty=1.2,
        no_repeat_ngram_size=3,
        forced_bos_token_id=forced_bos_id
    )
  
    # batch_decode không cần as_target_tokenizer
    outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return [normalize_space(o) for o in outputs]

deftranslate_sentence(s, src_lang, tgt_lang):
    src_lang_c = canon_label(src_lang)
    tgt_lang_c = canon_label(tgt_lang)
    hyp = nllb_generate([str(s)], src_lang_c, tgt_lang_c)
    return (hyp[0] if hyp and normalize_space(hyp[0]) else ".")

# -------------------- Inference on test.csv --------------------
required_cols = ["Row ID","Source Lang","Source Sentence","Target Lang","Target Sentence"]
test_df = pd.read_csv(test_path).rename(columns=rename_map)
for col in required_cols:
    if col not in test_df.columns:
        if col == "Target Sentence":
            test_df[col] = ""
        else:
            raise ValueError(f"Missing required column in test.csv: {col}")

preds = []
print("🚀 Starting inference on test data...")
fromtqdm.autoimport tqdm

for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    src_lang = str(row["Source Lang"])
    tgt_lang = str(row["Target Lang"])
    src_sent = str(row["Source Sentence"])
  
    src_lang_c = canon_label(src_lang)
    tgt_lang_c = canon_label(tgt_lang)
  
    # Priority: Copy from train if exact match exists
    if src_sent in train_lookup.get((src_lang_c, tgt_lang_c), {}):
        hyp = train_lookup[(src_lang_c, tgt_lang_c)][src_sent]
    else:
        # Otherwise translate with model
        hyp = translate_sentence(src_sent, src_lang, tgt_lang)
  
    preds.append(hyp if str(hyp).strip() else ".")

test_df["Target Sentence"] = pd.Series(preds).fillna(".").apply(
    lambda s: s if str(s).strip() else "."
)
submission = test_df[required_cols].copy()
out_path = f"{WORK_DIR}/submission.csv"
submission.to_csv(out_path, index=False)
print("✅ Wrote final submission to:", out_path)

try:
    fromIPython.displayimport display
    display(submission.head(10))
except Exception:
    print(submission.head(10).to_string(index=False))

used_model = sum(
    1 for (sl, tl) in set(zip(
        test_df['Source Lang'].apply(canon_label), 
        test_df['Target Lang'].apply(canon_label)
    )) if nllb_pair_supported(sl, tl) and USE_MODEL
)
print(f"\n📈 Final Status: {used_model} directions translated using fine-tuned NLLB model.")
```

```
⚠️ protobuf=6.33.0 is >=5 → downgrading to <5 to avoid MessageFactory issue
🌐 pip install ['protobuf<5']
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 294.9/294.9 kB 6.7 MB/s eta 0:00:00
```

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
bigframes 2.12.0 requires google-cloud-bigquery-storage<3.0.0,>=2.30.0, which is not installed.
opentelemetry-proto 1.37.0 requires protobuf<7.0,>=5.0, but you have protobuf 4.25.8 which is incompatible.
a2a-sdk 0.3.10 requires protobuf>=5.29.5, but you have protobuf 4.25.8 which is incompatible.
ray 2.51.1 requires click!=8.3.0,>=7.0, but you have click 8.3.0 which is incompatible.
bigframes 2.12.0 requires rich<14,>=12.4.4, but you have rich 14.2.0 which is incompatible.
pydrive2 1.21.3 requires cryptography<44, but you have cryptography 46.0.3 which is incompatible.
pydrive2 1.21.3 requires pyOpenSSL<=24.2.1,>=19.1.0, but you have pyopenssl 25.3.0 which is incompatible.
ydf 0.13.0 requires protobuf<7.0.0,>=5.29.1, but you have protobuf 4.25.8 which is incompatible.
grpcio-status 1.71.2 requires protobuf<6.0dev,>=5.26.1, but you have protobuf 4.25.8 which is incompatible.
gcsfs 2025.3.0 requires fsspec==2025.3.0, but you have fsspec 2025.10.0 which is incompatible.
```

```
protobuf pinned to: 6.33.0
Using competition directory: /kaggle/input/mm-lo-so-2025
Train files found: ['bhili-train.csv', 'gondi-train.csv', 'mundari-train.csv', 'santali-train.csv']
Loaded 20,000 pairs: Hindi -> Bhilli
Loaded 20,000 pairs: Hindi -> Gondi
Loaded 20,000 pairs: Hindi -> Mundari
Loaded 20,000 pairs: English -> Santali
Creating train lookup dictionary for post-processing...
Lookup created. Example: Hindi->Bhilli has 19,575 unique sentences.
🌐 Internet OK → downloading NLLB from HF to /kaggle/working/nllb200-600m
⬇️  snapshot_download attempt 1/3 ...
```

config.json: 100%

 846/846 [00:00<00:00, 88.8kB/s]

generation_config.json: 100%

 189/189 [00:00<00:00, 22.7kB/s]

pytorch_model.bin: 100%

 2.46G/2.46G [00:06<00:00, 1.02GB/s]

sentencepiece.bpe.model: 100%

 4.85M/4.85M [00:00<00:00, 17.9MB/s]

special_tokens_map.json: 

 3.55k/? [00:00<00:00, 374kB/s]

tokenizer.json: 100%

 17.3M/17.3M [00:00<00:00, 74.8MB/s]

tokenizer_config.json: 100%

 564/564 [00:00<00:00, 59.7kB/s]

```
✅ Downloaded NLLB to: /kaggle/working/nllb200-600m
✅ Using NLLB from: /kaggle/working/nllb200-600m
```

```
2025-11-12 10:31:58.007099: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1762943518.218441      20 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1762943518.279321      20 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
```

```
ℹ️ No adapter found at /kaggle/working/nllb-lora-adapter-final. Starting fine-tuning process...
trainable params: 14,942,208 || all params: 630,016,000 || trainable%: 2.3717
```

Map: 100%

 160000/160000 [01:18<00:00, 1896.30 examples/s]

```
No label_names provided for model class `PeftModelForSeq2SeqLM`. Since `PeftModel` hides base models input arguments, if label_names is not given, label_names can't be set automatically within `Trainer`. Note that empty label_names list will be used instead.
```

```
🔧 STARTING FINE-TUNE ON 160000 samples
```

 [5000/5000 2:07:47, Epoch 1/1]

| Step | Training Loss |
| ---- | ------------- |
| 25   | 4.560900      |
| 50   | 4.510900      |
| 75   | 4.453100      |
| 100  | 4.333300      |
| 125  | 3.977400      |
| 150  | 3.807500      |
| 175  | 3.781600      |
| 200  | 3.759900      |
| 225  | 3.696900      |
| 250  | 3.681400      |
| 275  | 3.469600      |
| 300  | 3.534300      |
| 325  | 3.492500      |
| 350  | 3.325600      |
| 375  | 3.298700      |
| 400  | 3.290100      |
| 425  | 3.250400      |
| 450  | 3.362600      |
| 475  | 3.127600      |
| 500  | 3.160700      |
| 525  | 3.141600      |
| 550  | 3.071500      |
| 575  | 3.087100      |
| 600  | 3.140200      |
| 625  | 3.078900      |
| 650  | 3.010300      |
| 675  | 3.044700      |
| 700  | 3.016100      |
| 725  | 3.117000      |
| 750  | 2.998100      |
| 775  | 2.901100      |
| 800  | 2.928400      |
| 825  | 2.904900      |
| 850  | 2.839800      |
| 875  | 2.914600      |
| 900  | 2.812600      |
| 925  | 2.904300      |
| 950  | 2.933400      |
| 975  | 2.924200      |
| 1000 | 2.857200      |
| 1025 | 2.967900      |
| 1050 | 2.826700      |
| 1075 | 2.812300      |
| 1100 | 2.905400      |
| 1125 | 2.878600      |
| 1150 | 2.786900      |
| 1175 | 2.846200      |
| 1200 | 2.814900      |
| 1225 | 2.851400      |
| 1250 | 2.875200      |
| 1275 | 2.744300      |
| 1300 | 2.795600      |
| 1325 | 2.762500      |
| 1350 | 2.808500      |
| 1375 | 2.885700      |
| 1400 | 2.679200      |
| 1425 | 2.835300      |
| 1450 | 2.790500      |
| 1475 | 2.848200      |
| 1500 | 2.751700      |
| 1525 | 2.709700      |
| 1550 | 2.784100      |
| 1575 | 2.798100      |
| 1600 | 2.758000      |
| 1625 | 2.696200      |
| 1650 | 2.724300      |
| 1675 | 2.629800      |
| 1700 | 2.666100      |
| 1725 | 2.787000      |
| 1750 | 2.770100      |
| 1775 | 2.801700      |
| 1800 | 2.692700      |
| 1825 | 2.726300      |
| 1850 | 2.676000      |
| 1875 | 2.806400      |
| 1900 | 2.782400      |
| 1925 | 2.727200      |
| 1950 | 2.639200      |
| 1975 | 2.714100      |
| 2000 | 2.656800      |
| 2025 | 2.695700      |
| 2050 | 2.757700      |
| 2075 | 2.738200      |
| 2100 | 2.713600      |
| 2125 | 2.753900      |
| 2150 | 2.752800      |
| 2175 | 2.611300      |
| 2200 | 2.690500      |
| 2225 | 2.732100      |
| 2250 | 2.717800      |
| 2275 | 2.715900      |
| 2300 | 2.702900      |
| 2325 | 2.718400      |
| 2350 | 2.621800      |
| 2375 | 2.678900      |
| 2400 | 2.654800      |
| 2425 | 2.575200      |
| 2450 | 2.695600      |
| 2475 | 2.716200      |
| 2500 | 2.607900      |
| 2525 | 2.675100      |
| 2550 | 2.674600      |
| 2575 | 2.626500      |
| 2600 | 2.701300      |
| 2625 | 2.581500      |
| 2650 | 2.580300      |
| 2675 | 2.648100      |
| 2700 | 2.659800      |
| 2725 | 2.679100      |
| 2750 | 2.657300      |
| 2775 | 2.684500      |
| 2800 | 2.651000      |
| 2825 | 2.655700      |
| 2850 | 2.621500      |
| 2875 | 2.680200      |
| 2900 | 2.589100      |
| 2925 | 2.737300      |
| 2950 | 2.669100      |
| 2975 | 2.544800      |
| 3000 | 2.645300      |
| 3025 | 2.719000      |
| 3050 | 2.590200      |
| 3075 | 2.639000      |
| 3100 | 2.615800      |
| 3125 | 2.604800      |
| 3150 | 2.588300      |
| 3175 | 2.662800      |
| 3200 | 2.567700      |
| 3225 | 2.728200      |
| 3250 | 2.611300      |
| 3275 | 2.634700      |
| 3300 | 2.662500      |
| 3325 | 2.640500      |
| 3350 | 2.640100      |
| 3375 | 2.555500      |
| 3400 | 2.592000      |
| 3425 | 2.597800      |
| 3450 | 2.531800      |
| 3475 | 2.578300      |
| 3500 | 2.606700      |
| 3525 | 2.682100      |
| 3550 | 2.592400      |
| 3575 | 2.588900      |
| 3600 | 2.609900      |
| 3625 | 2.594900      |
| 3650 | 2.556000      |
| 3675 | 2.613600      |
| 3700 | 2.624600      |
| 3725 | 2.546500      |
| 3750 | 2.544400      |
| 3775 | 2.623500      |
| 3800 | 2.626500      |
| 3825 | 2.587900      |
| 3850 | 2.622300      |
| 3875 | 2.607700      |
| 3900 | 2.638700      |
| 3925 | 2.599600      |
| 3950 | 2.572700      |
| 3975 | 2.643700      |
| 4000 | 2.579800      |
| 4025 | 2.581600      |
| 4050 | 2.627400      |
| 4075 | 2.600700      |
| 4100 | 2.572800      |
| 4125 | 2.525600      |
| 4150 | 2.609000      |
| 4175 | 2.607900      |
| 4200 | 2.559100      |
| 4225 | 2.642100      |
| 4250 | 2.577100      |
| 4275 | 2.517900      |
| 4300 | 2.579700      |
| 4325 | 2.495600      |
| 4350 | 2.530200      |
| 4375 | 2.542300      |
| 4400 | 2.580000      |
| 4425 | 2.634600      |
| 4450 | 2.604500      |
| 4475 | 2.603000      |
| 4500 | 2.614800      |
| 4525 | 2.579600      |
| 4550 | 2.574100      |
| 4575 | 2.552900      |
| 4600 | 2.534400      |
| 4625 | 2.609800      |
| 4650 | 2.553500      |
| 4675 | 2.481700      |
| 4700 | 2.617200      |
| 4725 | 2.550900      |
| 4750 | 2.563800      |
| 4775 | 2.563900      |
| 4800 | 2.651600      |
| 4825 | 2.606300      |
| 4850 | 2.624600      |
| 4875 | 2.536300      |
| 4900 | 2.566200      |
| 4925 | 2.609700      |
| 4950 | 2.667600      |
| 4975 | 2.568800      |
| 5000 | 2.537600      |

```
✅ DONE TRAIN. Saving final adapter to /kaggle/working/nllb-lora-adapter-final
Device map: {'model.shared': 0, 'lm_head': 0, 'model.decoder.embed_tokens': 0, 'model.encoder.embed_tokens': 0, 'model.encoder.embed_positions': 0, 'model.encoder.layers.0': 0, 'model.encoder.layers.1': 0, 'model.encoder.layers.2': 0, 'model.encoder.layers.3': 0, 'model.encoder.layers.4': 1, 'model.encoder.layers.5': 1, 'model.encoder.layers.6': 1, 'model.encoder.layers.7': 1, 'model.encoder.layers.8': 1, 'model.encoder.layers.9': 1, 'model.encoder.layers.10': 1, 'model.encoder.layers.11': 1, 'model.encoder.layer_norm': 1, 'model.decoder.embed_positions': 1, 'model.decoder.layers': 1, 'model.decoder.layer_norm': 1}
🚀 Starting inference on test data...
```

100%

 15999/15999 [4:31:26<00:00, 10.51it/s]

```
✅ Wrote final submission to: /kaggle/working/submission.csv
```

|   | Row ID | Source Lang | Source Sentence                                                                           | Target Lang | Target Sentence                                                                           |
| - | ------ | ----------- | ----------------------------------------------------------------------------------------- | ----------- | ----------------------------------------------------------------------------------------- |
| 0 | 54334  | Hindi       | उन्होंने कहा कि 2014 के बाद, इस परियोजना को प्...         | Bhilli      | तिनाये केदु की 2014 ने बाद मा इनो काम ने प्रधा...         |
| 1 | 87641  | Hindi       | वित्तीय कठिनाइयों को हल करने में सहायक होने के...   | Bhilli      | वित्तीय कठिनाइयों ने हल करवा मा मदद थावा नी हा...    |
| 2 | 32543  | Hindi       | मेरा सुझाव है कि हमारे सक्रिय दृष्टिकोण, नीतिय...   | Bhilli      | मावा सुझाव से कि मावा सक्रिय दृष्टिकोण नीतियां...  |
| 3 | 26313  | Hindi       | श्री मोदी ने कहा यह अटल जी ही थे जिन्होंने देश...     | Bhilli      | श्री मोदी यी केदु इद अटल जी ही से जिना देश नी ...       |
| 4 | 83303  | Hindi       | उत्सवादि मनाने के उपलक्ष्य में सुरापान करना सा...  | Bhilli      | त्यौहार मनावा ना उपलक्ष्य मा सुरापान करवा आम -...    |
| 5 | 131411 | Hindi       | तुम्‍हारे साथ कभी ऐसा हुआ।                                          | Bhilli      | तमुर संग कदी इदेक आयता।                                                |
| 6 | 101809 | Hindi       | यह सत्र ग्लासगो, यूनाइटेड किंगडम में आयोजित हुआ।  | Bhilli      | इद सत्र ग्लासगो, युनायटेड किंगडम ते आयोजित आयर।    |
| 7 | 59328  | Hindi       | यह 9 मार्च 2012 को रिलीज़ हुई थी, जिसे आम तौर ...            | Bhilli      | ये 9 मार्च 2012 ने रिलीज़ हुई थी, जिनकी आम तौर...           |
| 8 | 57205  | Hindi       | नियुक्ति मामलों की मंत्रिमंडलीय समिति ने भारती... | Bhilli      | नियुक्ति मामलों नी मंत्रिमंडलीय समिति ने भारती... |
| 9 | 103641 | Hindi       | भारत को और अधिक साफ-सुथरा बनाने और बेहतर स्वच्...    | Bhilli      | भारत ने अने वदु साफ-सुथरा बणावा अने बेहतर स्वच...    |

```
📈 Final Status: 8 directions translated using fine-tuned NLLB model.
```
