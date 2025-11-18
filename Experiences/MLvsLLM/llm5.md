Private Score : 184

Public Score : 308

```
# ========================= MMLoSo 2025 – NLLB LoRA MAX-BLEU Edition (FIXED) =========================
import os, re, gc, sys, socket, subprocess, time
import pandas as pd
from collections import defaultdict
from glob import glob
import torch

# -------------------- Pre-flight: pin protobuf<5 to avoid GetPrototype error --------------------
def _internet_ok(host="pypi.org", timeout=3):
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

try:
    import google.protobuf
    from packaging import version
    if version.parse(google.protobuf.__version__) >= version.parse("5.0.0"):
        print(f"⚠️ protobuf={google.protobuf.__version__} is >=5 → downgrading to <5 to avoid MessageFactory issue")
        _pip_install(["protobuf<5"])
        import importlib; importlib.invalidate_caches()
        import google.protobuf as _pb; print("protobuf pinned to:", _pb.__version__)
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

def normalize_space(s: str) -> str:
    return _WS_RE.sub(" ", str(s)).strip()

def simple_tokenize(s: str):
    s = normalize_space(s)
    s = _PUNC_RE.sub(r" \1 ", s)
    s = normalize_space(s)
    return s.split()

def detokenize(tokens):
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

# -------------------- Create train lookup dictionary --------------------
print("Creating train lookup dictionary for post-processing...")
train_lookup = defaultdict(dict)
for (s_lang, t_lang), pairs in bitext_by_dir.items():
    for s, t in pairs:
        train_lookup[(s_lang, t_lang)][s] = t
print(f"Lookup created. Example: Hindi->Bhilli has {len(train_lookup[('Hindi', 'Bhilli')]):,} unique sentences.")

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
HF_REPO = "facebook/nllb-200-distilled-1.3B"
LOCAL_SCAN_ROOT = "/kaggle/input"
LOCAL_FALLBACK  = "/kaggle/working/nllb-200-distilled-1.3B"
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

def _internet_ok2(host="huggingface.co", timeout=3):
    try:
        socket.setdefaulttimeout(timeout)
        socket.gethostbyname(host)
        return True
    except Exception:
        return False

def _download_from_hf(repo_id, local_dir, max_retries=3):
    if _internet_ok2():
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    try:
        import huggingface_hub
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

def nllb_pair_supported(src_lang, tgt_lang):
    return (src_lang in NLLB_LANG_CODE) and (tgt_lang in NLLB_LANG_CODE)

model, tokenizer = None, None

if USE_MODEL:
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        # SAFE: turn off bnb 8-bit path (Kaggle often lacks newest bitsandbytes/accelerate)  ── see refs.
        tokenizer = AutoTokenizer.from_pretrained(nllb_dir, use_fast=True)
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

        model = AutoModelForSeq2SeqLM.from_pretrained(
            nllb_dir, torch_dtype=torch.float16, device_map="auto",
        )

        # Load or train adapter
        from peft import PeftModel, LoraConfig, get_peft_model
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
            import datasets
            from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

            def make_multidir_dataset(bitext, tkn, max_per_dir=40000, max_len=256):
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

                def preprocess(examples):
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
        import traceback
        traceback.print_exc()
        USE_MODEL = False

# -------------------- Inference helpers --------------------
@torch.no_grad()
def nllb_generate(texts, src_lang, tgt_lang, max_new_tokens=256):
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

def translate_sentence(s, src_lang, tgt_lang):
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
from tqdm.auto import tqdm

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
    from IPython.display import display
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
protobuf pinned to: 6.33.0
Using competition directory: /kaggle/input/mm-lo-so-2025
Train files found: ['bhili-train.csv', 'gondi-train.csv', 'mundari-train.csv', 'santali-train.csv']
Loaded 20,000 pairs: Hindi -> Bhilli
Loaded 20,000 pairs: Hindi -> Gondi
Loaded 20,000 pairs: Hindi -> Mundari
Loaded 20,000 pairs: English -> Santali
Creating train lookup dictionary for post-processing...
Lookup created. Example: Hindi->Bhilli has 19,575 unique sentences.
🌐 Internet OK → downloading NLLB from HF to /kaggle/working/nllb-200-distilled-1.3B
⬇️  snapshot_download attempt 1/3 ...
```

```
No label_names provided for model class `PeftModelForSeq2SeqLM`. Since `PeftModel` hides base models input arguments, if label_names is not given, label_names can't be set automatically within `Trainer`. Note that empty label_names list will be used instead.
🔧 STARTING FINE-TUNE ON 160000 samples
 [5000/5000 5:44:16, Epoch 1/1]
Step	Training Loss
25	4.130000
50	4.094300
75	4.033900
100	3.933000
125	3.612400
150	3.504200
175	3.484800
200	3.440100
225	3.365100
250	3.368900
275	3.126800
300	3.180100
325	3.122700
350	2.961300
375	2.940400
400	2.921100
425	2.888000
450	2.991100
475	2.745900
500	2.791300
525	2.758100
550	2.679500
575	2.696800
600	2.747700
625	2.672400
650	2.616900
675	2.655600
700	2.622600
725	2.707900
750	2.609800
775	2.504600
800	2.535700
825	2.521900
850	2.463300
875	2.531300
900	2.402800
925	2.520100
950	2.538400
975	2.539100
1000	2.466500
1025	2.552900
1050	2.431100
1075	2.425500
1100	2.504400
1125	2.494300
1150	2.399600
1175	2.446400
1200	2.410500
1225	2.447400
1250	2.457400
1275	2.367600
1300	2.410900
1325	2.375000
1350	2.416300
1375	2.487800
1400	2.292100
1425	2.435700
1450	2.386900
1475	2.436600
1500	2.347400
1525	2.321700
1550	2.394900
1575	2.401500
1600	2.331500
1625	2.301600
1650	2.323200
1675	2.225000
1700	2.263600
1725	2.367500
1750	2.372800
1775	2.405900
1800	2.303500
1825	2.309400
1850	2.280100
1875	2.395600
1900	2.371800
1925	2.322600
1950	2.249100
1975	2.326800
2000	2.250800
2025	2.287800
2050	2.333800
2075	2.319400
2100	2.300500
2125	2.345100
2150	2.338400
2175	2.203600
2200	2.270400
2225	2.317900
2250	2.305000
2275	2.309500
2300	2.310900
2325	2.314800
2350	2.221000
2375	2.288500
2400	2.262700
2425	2.167100
2450	2.286700
2475	2.294100
2500	2.231400
2525	2.274500
2550	2.276600
2575	2.217100
2600	2.287100
2625	2.173000
2650	2.188500
2675	2.218100
2700	2.251400
2725	2.265600
2750	2.243500
2775	2.275400
2800	2.229600
2825	2.248700
2850	2.217100
2875	2.265700
2900	2.195400
2925	2.314700
2950	2.257800
2975	2.124700
3000	2.235700
3025	2.298500
3050	2.178200
3075	2.224600
3100	2.219400
3125	2.189700
3150	2.173100
3175	2.258800
3200	2.161600
3225	2.323600
3250	2.198300
3275	2.221300
3300	2.239500
3325	2.229100
3350	2.246000
3375	2.147400
3400	2.186800
3425	2.201800
3450	2.128300
3475	2.153600
3500	2.190400
3525	2.263000
3550	2.187600
3575	2.170500
3600	2.203800
3625	2.179600
3650	2.118800
3675	2.194800
3700	2.204700
3725	2.117200
3750	2.146400
3775	2.201300
3800	2.214000
3825	2.159300
3850	2.198700
3875	2.173000
3900	2.234900
3925	2.200500
3950	2.162600
3975	2.231600
4000	2.176200
4025	2.164000
4050	2.208300
4075	2.176400
4100	2.162500
4125	2.107800
4150	2.189900
4175	2.179800
4200	2.142100
4225	2.243800
4250	2.171500
4275	2.105300
4300	2.176300
4325	2.099000
4350	2.124000
4375	2.129500
4400	2.171300
4425	2.208700
4450	2.181400
4475	2.188500
4500	2.183800
4525	2.160900
4550	2.166900
4575	2.135000
4600	2.122000
4625	2.194200
4650	2.154100
4675	2.069000
4700	2.207700
4725	2.149600
4750	2.135200
4775	2.150300
4800	2.233200
4825	2.189500
4850	2.210900
4875	2.140900
4900	2.152900
4925	2.184300
4950	2.246200
4975	2.160300
5000	2.124900
✅ DONE TRAIN. Saving final adapter to /kaggle/working/nllb-lora-adapter-final
Device map: {'': 0}
🚀 Starting inference on test data...
  0%|          | 0/15999 [00:00<?, ?it/s]
✅ Wrote final submission to: /kaggle/working/submission.csv
Row ID	Source Lang	Source Sentence	Target Lang	Target Sentence
0	54334	Hindi	उन्होंने कहा कि 2014 के बाद, इस परियोजना को प्...	Bhilli	तिनाये केदु की 2014 ने बाद मा इनो काम ने प्रधा...
1	87641	Hindi	वित्तीय कठिनाइयों को हल करने में सहायक होने के...	Bhilli	वित्तीय कठिनाईयों को हल करने में सहायक होने के...
2	32543	Hindi	मेरा सुझाव है कि हमारे सक्रिय दृष्टिकोण, नीतिय...	Bhilli	मेरा सुझाव से कि हमारा सक्रिय दृष्टिकोण, नीतिय...
3	26313	Hindi	श्री मोदी ने कहा यह अटल जी ही थे जिन्होंने देश...	Bhilli	श्री मोदी ये केदू ये अटल जी ही हता जिने देश नी...
4	83303	Hindi	उत्सवादि मनाने के उपलक्ष्य में सुरापान करना सा...	Bhilli	तिहार मनावा ना उपलक्ष् य मा सुरापान करवानु साद...
5	131411	Hindi	तुम्‍हारे साथ कभी ऐसा हुआ।	Bhilli	अमअः लोः चिमिन नेकन होबा जना।
6	101809	Hindi	यह सत्र ग्लासगो, यूनाइटेड किंगडम में आयोजित हुआ।	Bhilli	यो सत्र ग्लासगो, यूनाइटेड किंगडम मा आयोजित थायो।
7	59328	Hindi	यह 9 मार्च 2012 को रिलीज़ हुई थी, जिसे आम तौर ...	Bhilli	यो 9 मार्च 2012 मा रिलिज थाई थी, जिना आम तौर प...
8	57205	Hindi	नियुक्ति मामलों की मंत्रिमंडलीय समिति ने भारती...	Bhilli	नियुक्ति मामलो नी मंत्रिमंडलीय समिति यी भारतीय...
9	103641	Hindi	भारत को और अधिक साफ-सुथरा बनाने और बेहतर स्वच्...	Bhilli	भारत ने अने वदु साफ-सुथरा बणावा अने असल स्वच्छ...

📈 Final Status: 8 directions translated using fine-tuned NLLB model.
```
