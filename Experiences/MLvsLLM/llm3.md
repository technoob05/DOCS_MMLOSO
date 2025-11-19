```
Private Score : 168.26741

Public  Score :  286.30303


importos,re,gc,sys,socket,subprocess,time
importpandasaspd
fromcollectionsimport defaultdict
fromglobimport glob
importtorch
importunicodedata  # Cho NFKC Normalization
importnumpyasnp   # Cho MBR
fromtqdm.autoimport tqdm

# -------------------- Pre-flight: Cài đặt thư viện cần thiết --------------------
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

# Cài sacrebleu cho MBR và protobuf tương thích
print("Cài đặt/Kiểm tra sacrebleu và protobuf...")
if _pip_install(["sacrebleu", "protobuf<5"]):
    print("Cài đặt thành công.")
else:
    print("Không thể cài đặt, tiếp tục với thư viện hệ thống.")

try:
    importsacrebleu
    fromtransformersimport AutoModelForSeq2SeqLM, AutoTokenizer
    frompeftimport PeftModel
    print(f"Sacrebleu (for MBR) loaded, version: {sacrebleu.__version__}")
except ImportError as e:
    print(f"⛔ FATAL: Không thể import thư viện. Lỗi: {e}")
    print("Vui lòng bật Internet và chạy lại notebook để cài đặt.")
    # Dừng script nếu không có thư viện
    sys.exit(1)

# -------------------- [QUAN TRỌNG] Định nghĩa đường dẫn --------------------
# Đường dẫn dữ liệu thi (test.csv, train.csv)
COMP_DIR = "/kaggle/input/mm-lo-so-2025" 
# Đường dẫn model/adapter bạn đã cung cấp
MODEL_INPUT_DIR = "/kaggle/input/number-1-noone-canbehere"
# Đường dẫn đến NLLB 600M base
NLLB_BASE_PATH = os.path.join(MODEL_INPUT_DIR, "nllb200-600m")
# Đường dẫn đến LoRA adapter đã train
ADAPTER_PATH = os.path.join(MODEL_INPUT_DIR, "nllb-lora-adapter-final")
# Nơi lưu file submission
WORK_DIR  = "/kaggle/working"

print(f"Loading Base Model from: {NLLB_BASE_PATH}")
print(f"Loading Adapter from: {ADAPTER_PATH}")
print(f"Loading Competition Data from: {COMP_DIR}")

# -------------------- Utilities (Với NFKC Normalization) --------------------
_WS_RE   = re.compile(r"\s+")
_PUNC_RE = re.compile(r'([\.\,\!\?\;\:\(\)\[\]\{\}"\'।\|/\\\-])')

defnormalize_space(s: str) -> str:
    s = str(s)
    # [TỐI ƯU 1] Thêm NFKC Normalization
    s = unicodedata.normalize('NFKC', s)
    return _WS_RE.sub(" ", s).strip()

OFFICIAL_LANGS = {"Bhilli","Hindi","Mundari","Gondi","English","Santali"}
SUB_LANG_CANON = {
    "bhili":"Bhilli","bhilli":"Bhilli",
    "hindi":"Hindi","mundari":"Mundari","gondi":"Gondi",
    "english":"English","santali":"Santali"
}
defcanon_label(lang: str) -> str:
    k = lang.strip().lower()
    return SUB_LANG_CANON.get(k, lang)

# -------------------- [TỐI ƯU 2] Tạo Train Lookup (Copy-paste) --------------------
defread_train_pairs(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Warning: Cannot find {csv_path}, skipping.")
        return None, None, []
  
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

print("Creating train lookup dictionary (for exact match copy)...")
train_lookup = defaultdict(dict)
bitext_by_dir = {}  # Khởi tạo bitext_by_dir tại đây
train_files = sorted(glob(f"{COMP_DIR}/*.csv"))
train_files = [f for f in train_files if 'test' not in os.path.basename(f).lower()]

for fp in train_files:
    s_lang, t_lang, pairs = read_train_pairs(fp)
    if s_lang:
        bitext_by_dir[(s_lang, t_lang)] = pairs
        bitext_by_dir[(t_lang, s_lang)] = [(t, s) for (s, t) in pairs]
        print(f"Loaded {len(pairs):,} pairs: {s_lang} <-> {t_lang}")

for (s_lang, t_lang), pairs in bitext_by_dir.items():
    for s, t in pairs:
        train_lookup[(s_lang, t_lang)][s] = t
print(f"Lookup created. Example: Hindi->Bhilli has {len(train_lookup[('Hindi','Bhilli')]):,} unique sentences.")

# -------------------- Đọc file Test --------------------
test_path = os.path.join(COMP_DIR, "test.csv")
if not os.path.exists(test_path):
    cand = [p for p in glob(f"{COMP_DIR}/*.csv") if os.path.basename(p).lower().startswith("test")]
    if cand: 
        test_path = cand[0]
    else: 
        print("⛔ FATAL: Không tìm thấy file test.csv!")
        sys.exit(1)

print(f"Loading test file from: {test_path}")
test_df = pd.read_csv(test_path)
rename_map = {}
for c in test_df.columns:
    cn = c.strip()
    if cn.lower().replace("_"," ") == "row id":          rename_map[c] = "Row ID"
    elif cn.lower() == "source lang":                    rename_map[c] = "Source Lang"
    elif cn.lower() == "source sentence":                rename_map[c] = "Source Sentence"
    elif cn.lower() == "target lang":                    rename_map[c] = "Target Lang"
    elif cn.lower() == "target sentence":                rename_map[c] = "Target Sentence"
test_df = test_df.rename(columns=rename_map)

required_cols = ["Row ID","Source Lang","Source Sentence","Target Lang"]
if "Target Sentence" not in test_df.columns:
    test_df["Target Sentence"] = ""

# -------------------- Load Model và Adapter (Chỉ Inference) --------------------
NLLB_LANG_CODE = {
    "English": "eng_Latn",
    "Hindi":   "hin_Deva",
    "Santali": "sat_Olck",
    "Bhilli":  "hin_Deva",
    "Gondi":   "hin_Deva",
    "Mundari": "hin_Deva",
}
defnllb_pair_supported(src_lang, tgt_lang):
    return (src_lang in NLLB_LANG_CODE) and (tgt_lang in NLLB_LANG_CODE)

model, tokenizer = None, None
try:
    print("Loading Tokenizer from adapter directory...")
    # Load tokenizer từ ADAPTER_PATH là an toàn nhất
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, use_fast=True)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    print("Loading Base Model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        NLLB_BASE_PATH, torch_dtype=torch.float16, device_map="auto",
    )

    print(f"Attaching LoRA Adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    print("✅ Adapter loaded successfully.")
  
    # Tối ưu cho inference
    model.config.use_cache = True
    model = model.eval()
    if hasattr(model, "hf_device_map"): 
        print("Device map:", model.hf_device_map)

except Exception as e:
    print(f"⚠️ FATAL: Could not initialize NLLB model: {e}")
    importtraceback; traceback.print_exc()
    USE_MODEL = False
else:
    USE_MODEL = True

# -------------------- [TỐI ƯU 3] MBR Reranking Helpers --------------------
MBR_GAIN_BLEU_W = 0.6
MBR_GAIN_CHRF_W = 0.4
MBR_N_BEST = 10  # Số lượng câu N-best để tạo và rerank

defcalculate_utility(hyp, ref_list):
    if not sacrebleu or not hyp or not ref_list:
        return 0.0
  
    bleu_score = sacrebleu.sentence_bleu(
        hyp, ref_list, smooth_method='add-k', smooth_value=1
    ).score
    chrf_score = sacrebleu.sentence_chrf(
        hyp, ref_list, char_order=6, beta=3
    ).score
  
    gain = (MBR_GAIN_BLEU_W * bleu_score) + (MBR_GAIN_CHRF_W * chrf_score)
    return gain

defmbr_rerank(n_best_list):
    if not n_best_list: 
        return "."
    if len(n_best_list) == 1: 
        return n_best_list[0]
  
    unique_hyps = sorted(list(set(n_best_list)))
    if len(unique_hyps) == 1: 
        return unique_hyps[0]

    scores = np.zeros(len(unique_hyps))
    for i, hyp_i in enumerate(unique_hyps):
        total_utility = calculate_utility(hyp_i, unique_hyps)
        scores[i] = total_utility
  
    best_index = np.argmax(scores)
    return unique_hyps[best_index]

# -------------------- Inference helpers (Tích hợp MBR) --------------------
@torch.no_grad()
defnllb_generate_n_best(texts, src_lang, tgt_lang, max_new_tokens=256):
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
  
    try:
        forced_bos_id = tokenizer.lang_code_to_id[tgt_code]
    except Exception:
        forced_bos_id = tokenizer.convert_tokens_to_ids(tgt_code)

    generated_tokens = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=MBR_N_BEST,
        num_return_sequences=MBR_N_BEST,
        length_penalty=1.2,
        no_repeat_ngram_size=3,
        forced_bos_token_id=forced_bos_id
    )
  
    outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return [normalize_space(o) for o in outputs if normalize_space(o)]

deftranslate_sentence(s, src_lang, tgt_lang):
    src_lang_c = canon_label(src_lang)
    tgt_lang_c = canon_label(tgt_lang)
  
    n_best_hyp_list = nllb_generate_n_best([str(s)], src_lang_c, tgt_lang_c)
    if not n_best_hyp_list:
        return "."
  
    best_hyp = mbr_rerank(n_best_hyp_list)
    return (best_hyp if normalize_space(best_hyp) else ".")

# -------------------- Chạy Inference trên test.csv --------------------
if USE_MODEL:
    preds = []
    print("\n🚀 Starting inference on test data (with NFKC + Train-Lookup + MBR Reranking)...")
  
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Tắt cảnh báo

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        src_lang = str(row["Source Lang"])
        tgt_lang = str(row["Target Lang"])
        src_sent = str(row["Source Sentence"])
  
        src_lang_c = canon_label(src_lang)
        tgt_lang_c = canon_label(tgt_lang)
  
        # TỐI ƯU 2: Copy từ train nếu khớp
        if src_sent in train_lookup.get((src_lang_c, tgt_lang_c), {}):
            hyp = train_lookup[(src_lang_c, tgt_lang_c)][src_sent]
        else:
            # TỐI ƯU 1+3: Dịch bằng NLLB + MBR (đã có NFKC)
            hyp = translate_sentence(src_sent, src_lang, tgt_lang)
  
        preds.append(hyp if str(hyp).strip() else ".")

    test_df["Target Sentence"] = pd.Series(preds).fillna(".").apply(
        lambda s: s if str(s).strip() else ".")
    submission = test_df[required_cols + ["Target Sentence"]].copy()
  
    out_path = f"{WORK_DIR}/submission.csv"
    # ✅ SỬA LỖI Ở ĐÂY: dùng out_path chứ không phải out_BAPATH
    submission.to_csv(out_path, index=False)
    print(f"✅ Wrote final submission to: {out_path}")
  
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
    print(f"\n📈 Final Status: {used_model} directions translated using loaded NLLB + MBR Reranking.")
else:
    print("⛔ Model không được tải. Không thể chạy inference.")
```

```
Cài đặt/Kiểm tra sacrebleu và protobuf...
🌐 pip install ['sacrebleu', 'protobuf<5']
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 51.8/51.8 kB 1.8 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 104.1/104.1 kB 4.0 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 294.9/294.9 kB 11.0 MB/s eta 0:00:00
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
Cài đặt thành công.
```

```
2025-11-13 06:25:59.839157: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1763015160.063376      19 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1763015160.135128      19 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
```

```
Sacrebleu (for MBR) loaded, version: 2.5.1
Loading Base Model from: /kaggle/input/number-1-noone-canbehere/nllb200-600m
Loading Adapter from: /kaggle/input/number-1-noone-canbehere/nllb-lora-adapter-final
Loading Competition Data from: /kaggle/input/mm-lo-so-2025
Creating train lookup dictionary (for exact match copy)...
Loaded 20,000 pairs: Hindi <-> Bhilli
Loaded 20,000 pairs: Hindi <-> Gondi
Loaded 20,000 pairs: Hindi <-> Mundari
Loaded 20,000 pairs: English <-> Santali
Lookup created. Example: Hindi->Bhilli has 19,575 unique sentences.
Loading test file from: /kaggle/input/mm-lo-so-2025/test.csv
Loading Tokenizer from adapter directory...
Loading Base Model...
Attaching LoRA Adapter from /kaggle/input/number-1-noone-canbehere/nllb-lora-adapter-final...
✅ Adapter loaded successfully.
Device map: {'model.shared': 0, 'lm_head': 0, 'model.decoder.embed_tokens': 0, 'model.encoder.embed_tokens': 0, 'model.encoder.embed_positions': 0, 'model.encoder.layers.0': 0, 'model.encoder.layers.1': 0, 'model.encoder.layers.2': 0, 'model.encoder.layers.3': 0, 'model.encoder.layers.4': 1, 'model.encoder.layers.5': 1, 'model.encoder.layers.6': 1, 'model.encoder.layers.7': 1, 'model.encoder.layers.8': 1, 'model.encoder.layers.9': 1, 'model.encoder.layers.10': 1, 'model.encoder.layers.11': 1, 'model.encoder.layer_norm': 1, 'model.decoder.embed_positions': 1, 'model.decoder.layers': 1, 'model.decoder.layer_norm': 1}

🚀 Starting inference on test data (with NFKC + Train-Lookup + MBR Reranking)...
```

100%

 15999/15999 [5:09:55<00:00,  2.03it/s]

```
✅ Wrote final submission to: /kaggle/working/submission.csv
```

|   | Row ID | Source Lang | Source Sentence                                                                           | Target Lang | Target Sentence                                                                           |
| - | ------ | ----------- | ----------------------------------------------------------------------------------------- | ----------- | ----------------------------------------------------------------------------------------- |
| 0 | 54334  | Hindi       | उन्होंने कहा कि 2014 के बाद, इस परियोजना को प्...         | Bhilli      | तिनाये केदु की 2014 ने बाद मा इनो काम ने प्रधा...         |
| 1 | 87641  | Hindi       | वित्तीय कठिनाइयों को हल करने में सहायक होने के...   | Bhilli      | वित्तीय कठिनाइयों को हल करने में सहायक होने के...   |
| 2 | 32543  | Hindi       | मेरा सुझाव है कि हमारे सक्रिय दृष्टिकोण, नीतिय...   | Bhilli      | नाअः सुझाव मेनअः चि मावा सक्रिय दृष्टिकोण, नीत...   |
| 3 | 26313  | Hindi       | श्री मोदी ने कहा यह अटल जी ही थे जिन्होंने देश...     | Bhilli      | श्री मोदी यी केदु इद अटल जी ही से जिना देश नी ...       |
| 4 | 83303  | Hindi       | उत्सवादि मनाने के उपलक्ष्य में सुरापान करना सा...  | Bhilli      | उत्सवादि मनावा ना उपलक्ष्य मा सुरापान करवा आम ...   |
| 5 | 131411 | Hindi       | तुम्‍हारे साथ कभी ऐसा हुआ।                                          | Bhilli      | तमुन संग कदी इदेक आयता।                                                |
| 6 | 101809 | Hindi       | यह सत्र ग्लासगो, यूनाइटेड किंगडम में आयोजित हुआ।  | Bhilli      | इद सत्र ग्लासगो युनायटेड किंगडम ते आयोजित आयता।   |
| 7 | 59328  | Hindi       | यह 9 मार्च 2012 को रिलीज़ हुई थी, जिसे आम तौर ...            | Bhilli      | ये 9 मार्च 2012 ने रिलीज़ हुई थी, जिनकी आम तौर...           |
| 8 | 57205  | Hindi       | नियुक्ति मामलों की मंत्रिमंडलीय समिति ने भारती... | Bhilli      | नियुक्ति मामला नी मंत्रिमंडलीय समिति ने भारतीय... |
| 9 | 103641 | Hindi       | भारत को और अधिक साफ-सुथरा बनाने और बेहतर स्वच्...    | Bhilli      | भारत ने अने घणी साफ-सुथरा बणावा अने बेहतर स्वच...    |

```
📈 Final Status: 8 directions translated using loaded NLLB + MBR Reranking.
```
