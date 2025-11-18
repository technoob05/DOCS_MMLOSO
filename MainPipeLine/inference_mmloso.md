```
Public Score :306.55737
Private Score : 174.53126


importos,re,gc,sys,socket,subprocess,time,math,pickle
importpandasaspd
importnumpyasnp
importtorch

fromcollectionsimport defaultdict, Counter
fromglobimport glob
fromtqdm.autoimport tqdm

# ============================================================
# 0. PRE-FLIGHT: CÀI THƯ VIỆN
# ============================================================
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

print("Cài đặt/Kiểm tra sacrebleu và protobuf...")
if _pip_install(["sacrebleu", "protobuf<5"]):
    print("Cài đặt sacrebleu/protobuf thành công.")
else:
    print("Không thể cài đặt, tiếp tục với thư viện hệ thống.")

try:
    importsacrebleu
    fromtransformersimport AutoModelForSeq2SeqLM, AutoTokenizer
    frompeftimport PeftModel
    print(f"Sacrebleu loaded, version: {sacrebleu.__version__}")
except ImportError as e:
    print(f"⛔ FATAL: Không thể import thư viện. Lỗi: {e}")
    sys.exit(1)

# ============================================================
# 1. ĐƯỜNG DẪN
# ============================================================
COMP_DIR        = "/kaggle/input/mm-lo-so-2025"
MODEL_INPUT_DIR = "/kaggle/input/number-1-noone-canbehere"
NLLB_BASE_PATH  = os.path.join(MODEL_INPUT_DIR, "nllb200-600m")
ADAPTER_PATH    = os.path.join(MODEL_INPUT_DIR, "nllb-lora-adapter-final")
SMT_MODEL_PATH  = "/kaggle/input/best-ml/smt_model_bt_mbr.pkl"
WORK_DIR        = "/kaggle/working"

print(f"Loading Base Model from: {NLLB_BASE_PATH}")
print(f"Loading Adapter    from: {ADAPTER_PATH}")
print(f"Loading SMT model  from: {SMT_MODEL_PATH}")
print(f"Loading Data       from: {COMP_DIR}")

# ============================================================
# 2. UTILITIES: NORMALIZE + TOKENIZE (DÙNG CHUNG NLLB + SMT)
# ============================================================
defnormalize_space(s: str) -> str:
"""
    KHÔNG DÙNG NFKC.
    Chỉ gom whitespace về 1 khoảng trắng.
    """
    s = str(s)
    return re.sub(r"\s+", " ", s).strip()

defsimple_tokenize(s: str):
    s = normalize_space(s)
    s = re.sub(r"(\d+)", r" \1 ", s)
    s = re.sub(r"([.,!?;:()\[\]{}\"'“”‘’।|/\\\-])", r" \1 ", s)
    s = normalize_space(s)
    return s.split()

defdetokenize(tokens):
    out = []
    for i, t in enumerate(tokens):
        if i > 0 and t in {".", ",", "!", "?", ";", ":", ")", "”", "’", "।"}:
            if out:
                out[-1] = out[-1] + t
        elif t in {"(", "“", "‘"} and len(out) > 0:
            out.append(t)
        else:
            out.append(t)
    txt = " ".join(out)
    txt = txt.replace("( ", "(").replace(" )", ")")
    txt = txt.replace("“ ", "“").replace(" ”", "”").replace("‘ ", "‘").replace(" ’", "’")
    return normalize_space(txt)

# --- Fuzzy key: dùng cho fuzzy copy-from-train ---
PUNCT_CHARS = r"\.\,\!\?\;\:\(\)\[\]\{\}\"'“”‘’।\|/\\\-"

defnormalize_for_fuzzy(s: str) -> str:
"""
    Cực conservative fuzzy:
      - normalize_space
      - lowercase
      - remove punctuation
    → chỉ khác nhau ở dấu câu / spacing mới match.
    """
    s = normalize_space(s).lower()
    s = re.sub(f"[{PUNCT_CHARS}]", " ", s)
    s = normalize_space(s)
    return s

SUB_LANG_CANON = {
    "bhili":"Bhilli","bhilli":"Bhilli",
    "hindi":"Hindi","mundari":"Mundari","gondi":"Gondi",
    "english":"English","santali":"Santali",
}
defcanon_label(lang: str) -> str:
    return SUB_LANG_CANON.get(lang.strip().lower(), lang)

FORWARD_DIRS = {("Hindi","Bhilli"), ("Hindi","Mundari"), ("Hindi","Gondi"), ("English","Santali")}
REVERSE_DIRS = {("Bhilli","Hindi"), ("Mundari","Hindi"), ("Gondi","Hindi"), ("Santali","English")}

# Seed cho ổn định một chút
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ============================================================
# 3. TRAIN-LOOKUP (COPY-FROM-TRAIN + FUZZY COPY)
# ============================================================
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

print("Creating train lookup dictionary (for exact + fuzzy copy)...")
train_lookup  = defaultdict(dict)                      # exact
fuzzy_lookup  = defaultdict(lambda: defaultdict(list)) # fuzzy[(s_lang,t_lang)][fuzzy_key] = [(len, src, tgt), ...]

bitext_by_dir = {}
train_files = sorted(glob(f"{COMP_DIR}/*.csv"))
train_files = [f for f in train_files if 'test' not in os.path.basename(f).lower()]

for fp in train_files:
    s_lang, t_lang, pairs = read_train_pairs(fp)
    if s_lang:
        bitext_by_dir[(s_lang, t_lang)] = pairs
        bitext_by_dir[(t_lang, s_lang)] = [(t, s) for (s, t) in pairs]
        print(f"Loaded {len(pairs):,} pairs: {s_lang} <-> {t_lang}")

for (s_lang, t_lang), pairs in bitext_by_dir.items():
    d_exact = train_lookup[(s_lang, t_lang)]
    d_fuzzy = fuzzy_lookup[(s_lang, t_lang)]
    for s, t in pairs:
        # exact key
        d_exact[s] = t

        # fuzzy key + length
        fk   = normalize_for_fuzzy(s)
        toks = simple_tokenize(s)
        d_fuzzy[fk].append((len(toks), s, t))

print(f"Exact lookup example: Hindi->Bhilli has {len(train_lookup[('Hindi','Bhilli')]):,} entries.")

# Hàm fuzzy copy conservative
deffuzzy_copy_from_train(src_norm, src_lang_c, tgt_lang_c, max_len_diff=1):
"""
    src_norm: câu đã normalize_space
    Cực conservative:
      - fuzzy_key trùng (bỏ dấu, lowercase)
      - |len_train - len_src| <= max_len_diff (default=1)
    """
    key = (src_lang_c, tgt_lang_c)
    bucket = fuzzy_lookup.get(key, None)
    if not bucket:
        return None

    fk = normalize_for_fuzzy(src_norm)
    cands = bucket.get(fk, None)
    if not cands:
        return None

    src_len = len(simple_tokenize(src_norm))
    best_tgt = None
    best_diff = 10**9

    for L, s_train, t_train in cands:
        diff = abs(L - src_len)
        if diff <= max_len_diff and diff < best_diff:
            best_diff = diff
            best_tgt = t_train
            if diff == 0:
                break  # quá đẹp rồi, dừng

    return best_tgt

# ============================================================
# 4. ĐỌC test.csv
# ============================================================
test_path = os.path.join(COMP_DIR, "test.csv")
if not os.path.exists(test_path):
    cand = [p for p in glob(f"{COMP_DIR}/*.csv") if os.path.basename(p).lower().startswith("test")]
    if cand:
        test_path = cand[0]
    else:
        print("⛔ FATAL: Không tìm thấy test.csv")
        sys.exit(1)

print(f"Loading test file from: {test_path}")
test_df = pd.read_csv(test_path)
rename_map = {}
for c in test_df.columns:
    cn = c.strip()
    if cn.lower().replace("_"," ") == "row id":       rename_map[c] = "Row ID"
    elif cn.lower() == "source lang":                 rename_map[c] = "Source Lang"
    elif cn.lower() == "source sentence":             rename_map[c] = "Source Sentence"
    elif cn.lower() == "target lang":                 rename_map[c] = "Target Lang"
    elif cn.lower() == "target sentence":             rename_map[c] = "Target Sentence"
test_df = test_df.rename(columns=rename_map)

required_cols = ["Row ID","Source Lang","Source Sentence","Target Lang"]
if "Target Sentence" not in test_df.columns:
    test_df["Target Sentence"] = ""

# ============================================================
# 5. LOAD NLLB + LORA
# ============================================================
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
    print("Loading Tokenizer from ADAPTER_PATH...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, use_fast=True)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    print("Loading Base NLLB model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        NLLB_BASE_PATH, torch_dtype=torch.float16, device_map="auto",
    )

    print("Attaching LoRA adapter...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    print("✅ NLLB + LoRA loaded successfully.")
    model.config.use_cache = True
    model = model.eval()
    if hasattr(model, "hf_device_map"):
        print("Device map:", model.hf_device_map)

except Exception as e:
    print(f"⚠️ FATAL: Could not initialize NLLB model: {e}")
    importtraceback; traceback.print_exc()
    USE_NLLB = False
else:
    USE_NLLB = True

# ============================================================
# 6. LOAD SMT MODEL (TỪ PICKLE)
# ============================================================
classSMT_Model:
    pass

USE_SMT = False
smt_model = None
if os.path.exists(SMT_MODEL_PATH):
    print(f"Loading SMT model from: {SMT_MODEL_PATH}")
    try:
        with open(SMT_MODEL_PATH, "rb") as f:
            smt_model = pickle.load(f)
        print("✅ SMT model loaded successfully.")
        USE_SMT = True
    except Exception as e:
        print(f"⚠️ Error loading SMT model: {e}")
        USE_SMT = False
else:
    print("⚠️ SMT model file not found, skip SMT.")
    USE_SMT = False

PUNCT_SET = {".",",","!","?",";",":","(",")","[","]","{","}","–","—","-","—","|","/","\\","'","\"","“","”","‘","’","।"}
NUM_RE   = re.compile(r"^[+\-]?\d+([.,:/-]\d+)*$")
URL_RE   = re.compile(r"(https?://|www\.)", re.I)
EMAIL_RE = re.compile(r".+@.+\..+")

defis_copy_token(w: str) -> bool:
    w_ = w.strip()
    return (w_ in PUNCT_SET or NUM_RE.match(w_) is not None or
            URL_RE.search(w_) is not None or EMAIL_RE.match(w_) is not None)

defcreate_lm_score_fn(lm_data):
    if lm_data is None:
        return lambda w, p2, p1: 0.0
    ngram, context_count, add_k = lm_data
    defscore(next_word, prev2, prev1):
        ctx = (prev2, prev1)
        c = ngram[ctx][next_word]
        total = context_count[ctx]
        return math.log((c + add_k) / (total + add_k * 1e5))
    return score

defbeam_decode_n_best(src_tokens, lex_prob, lm_score_fn,
                       beam_size=5,        # beam cho SMT
                       lm_w=0.85,
                       rep_pen_w=0.75, len_ratio=1.0, len_w=0.12):
    Beam = [([], "<s>", "<s>", 0.0, 0)]  # (hyp, prev2, prev1, score, L)
    for w in src_tokens:
        newB = []
        if is_copy_token(w):
            cand = [(w, 0.0)]
        else:
            dist = lex_prob.get(w, None)
            if not dist:
                cand = [(w, math.log(1e-6))]
            else:
                lst = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:5]
                cand = [(tw, math.log(max(1e-12, p))) for (tw,p) in lst]
        hyp_cache = {}
        for hyp, p2, p1, score, L in Beam:
            for tw, lp in cand:
                lm  = lm_score_fn(tw.lower(), p2.lower(), p1.lower())
                rep = 0.0
                if len(hyp)>=2 and hyp[-1]==tw and hyp[-2]==tw:
                    rep -= 3.0*rep_pen_w
                elif len(hyp)>=1 and hyp[-1]==tw:
                    rep -= 1.0*rep_pen_w
                tgt_len = L+1
                src_len = len(src_tokens)
                goal    = len_ratio * src_len
                len_pen = -len_w * abs(tgt_len - goal)/max(1.0, goal)
                new_score = score + lp + lm_w*lm + rep + len_pen
                new_state = (p1, tw)
                if new_state not in hyp_cache or new_score > hyp_cache[new_state][3]:
                    hyp_cache[new_state] = (hyp+[tw], p1, tw, new_score, tgt_len)
        newB = list(hyp_cache.values())
        newB.sort(key=lambda x: x[3], reverse=True)
        Beam = newB[:beam_size]

    deffinal_score(entry):
        hyp, p2, p1, sc, L = entry
        goal = len_ratio * len(src_tokens)
        final_sc = sc + len_w * math.log(max(1.0, L)/max(1.0, goal))
        return (hyp, p2, p1, final_sc, L)

    final_beam = [final_score(entry) for entry in Beam]
    final_beam.sort(key=lambda x: x[3], reverse=True)
    return final_beam[:beam_size]

defsmt_generate_n_best(smt_model, s, src_lang, tgt_lang, n_best=5):
"""
    Trả về list các câu SMT (string) từ beam search.
    Dùng N-best để MBR có thêm đa dạng.
    """
    if not smt_model:
        return []
    src_lang_c = canon_label(src_lang)
    tgt_lang_c = canon_label(tgt_lang)
    key = (src_lang_c, tgt_lang_c)
    if key not in getattr(smt_model, "lex_prob_by_dir", {}):
        return []
    s_norm = normalize_space(s)
    toks   = simple_tokenize(s_norm)
    if not toks:
        return []

    lower = [w.lower() for w in toks]
    lex_prob = smt_model.lex_prob_by_dir[key]
    lm_data  = smt_model.lm_data_by_tgt.get(tgt_lang_c, None)
    lm_score = create_lm_score_fn(lm_data)
    len_r    = smt_model.len_ratio_by_dir.get(key, 1.0)

    params = smt_model.params_fwd if key in FORWARD_DIRS else smt_model.params_rev
    beam_size = min(params.get("beam", 10), n_best)

    n_best_list = beam_decode_n_best(
        lower, lex_prob, lm_score,
        beam_size=beam_size,
        lm_w=params.get("lm_w", 0.9),
        rep_pen_w=params.get("rep_pen", 0.8),
        len_ratio=len_r,
        len_w=params.get("len_w", 0.12),
    )
    if not n_best_list:
        return []

    outs = []
    for entry in n_best_list[:n_best]:
        hyp_tokens = entry[0]
        if hyp_tokens:
            hyp = detokenize(hyp_tokens)
            if hyp.strip():
                outs.append(hyp)
    return outs

# ============================================================
# 7. MBR RERANKING (SACREBLEU BLEU + CHRF3) – DÙNG CHUNG
# ============================================================
MBR_GAIN_BLEU_W = 0.6
MBR_GAIN_CHRF_W = 0.4
NLLB_N_BEST     = 10   # dùng làm num_beams & num_return_sequences
SMT_N_BEST      = 5    # SMT N-best cho đa dạng

defcalculate_utility(hyp, ref_list):
    if not hyp or not ref_list:
        return 0.0
    bleu_score = sacrebleu.sentence_bleu(
        hyp, ref_list, smooth_method="add-k", smooth_value=1
    ).score
    chrf_score = sacrebleu.sentence_chrf(
        hyp, ref_list, char_order=6, beta=3
    ).score
    return MBR_GAIN_BLEU_W * bleu_score + MBR_GAIN_CHRF_W * chrf_score

defmbr_rerank_texts(candidates):
"""
    candidates: list[str] (đã unique, normalized)
    """
    if not candidates:
        return "."
    if len(candidates) == 1:
        return candidates[0]
    unique_hyps = sorted(list(set(candidates)))
    if len(unique_hyps) == 1:
        return unique_hyps[0]

    scores = np.zeros(len(unique_hyps), dtype=np.float64)
    for i, hyp_i in enumerate(unique_hyps):
        scores[i] = calculate_utility(hyp_i, unique_hyps)
    best_idx = int(np.argmax(scores))
    return unique_hyps[best_idx]

# ============================================================
# 8. NLLB GENERATION – BEAM SEARCH + N-BEST
# ============================================================
@torch.no_grad()
defnllb_generate_n_best(texts, src_lang, tgt_lang, max_new_tokens=256):
"""
    Dùng beam search với:
      - num_beams = NLLB_N_BEST (=10)
      - num_return_sequences = NLLB_N_BEST (=10)
      → lấy full N-best cho MBR.
    """
    if not (USE_NLLB and tokenizer and model):
        return []
    if not nllb_pair_supported(src_lang, tgt_lang):
        return []

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
        num_beams=NLLB_N_BEST,
        num_return_sequences=NLLB_N_BEST,
        length_penalty=1.2,
        no_repeat_ngram_size=3,
        forced_bos_token_id=forced_bos_id
    )
    outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    outs = []
    for o in outputs:
        s = normalize_space(o)
        if s:
            outs.append(s)
    return outs

# ============================================================
# 9. ENSEMBLE TRANSLATION (TRAIN-LOOKUP + FUZZY + SMT + NLLB + MBR)
# ============================================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if USE_NLLB or USE_SMT:
    preds = []
    fuzzy_hits = 0
    exact_hits = 0

    print("\n🚀 Ensemble inference (Exact + Fuzzy Train-Lookup + SMT + NLLB + MBR Union)...")

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        src_lang = str(row["Source Lang"])
        tgt_lang = str(row["Target Lang"])
        src_sent = str(row["Source Sentence"])

        src_lang_c = canon_label(src_lang)
        tgt_lang_c = canon_label(tgt_lang)
        src_norm   = normalize_space(src_sent)

        # 1) Exact-match lookup từ train
        exact_dict = train_lookup.get((src_lang_c, tgt_lang_c), {})
        if src_norm in exact_dict:
            hyp = exact_dict[src_norm]
            preds.append(hyp if str(hyp).strip() else ".")
            exact_hits += 1
            continue

        # 1b) Fuzzy copy-from-train (cực conservative)
        fuzzy_hyp = fuzzy_copy_from_train(src_norm, src_lang_c, tgt_lang_c, max_len_diff=1)
        if fuzzy_hyp is not None and str(fuzzy_hyp).strip():
            preds.append(fuzzy_hyp)
            fuzzy_hits += 1
            continue

        # 2) SMT N-best
        smt_cands = []
        if USE_SMT:
            try:
                smt_cands = smt_generate_n_best(
                    smt_model, src_sent, src_lang, tgt_lang, n_best=SMT_N_BEST
                )
            except Exception as e:
                smt_cands = []

        # 3) NLLB N-best
        nllb_cands = []
        if USE_NLLB and nllb_pair_supported(src_lang_c, tgt_lang_c):
            try:
                nllb_cands = nllb_generate_n_best(
                    [src_sent], src_lang_c, tgt_lang_c
                )
            except Exception as e:
                nllb_cands = []

        # 4) Pool hợp nhất + MBR rerank
        pool = []
        for cand in smt_cands + nllb_cands:
            s = normalize_space(cand)
            if s:
                pool.append(s)
        pool = list(dict.fromkeys(pool))  # unique, giữ order

        if not pool:
            # fallback: copy source (hoặc ".")
            fallback = src_norm if src_norm.strip() else "."
            preds.append(fallback)
        else:
            best = mbr_rerank_texts(pool)
            preds.append(best if best.strip() else ".")

    test_df["Target Sentence"] = pd.Series(preds).fillna(".").apply(
        lambda s: s if str(s).strip() else "."
    )
    submission = test_df[required_cols + ["Target Sentence"]].copy()
    out_path = f"{WORK_DIR}/submission.csv"
    submission.to_csv(out_path, index=False)
    print(f"\n✅ Wrote final ENSEMBLE submission to: {out_path}")
    print(f"✅ Exact train-copy hits : {exact_hits}")
    print(f"✅ Fuzzy train-copy hits : {fuzzy_hits}")

    try:
        fromIPython.displayimport display
        display(submission.head(10))
    except Exception:
        print(submission.head(10).to_string(index=False))

    used_dirs = set(zip(
        test_df["Source Lang"].apply(canon_label),
        test_df["Target Lang"].apply(canon_label)
    ))
    used_nllb = sum(1 for (sl, tl) in used_dirs if nllb_pair_supported(sl, tl) and USE_NLLB)
    used_smt  = sum(1 for (sl, tl) in used_dirs
                    if USE_SMT and (sl, tl) in getattr(smt_model, "lex_prob_by_dir", {}))
    print(f"\n📈 Directions using NLLB: {used_nllb}")
    print(f"📈 Directions using SMT : {used_smt}")
else:
    print("⛔ Neither NLLB nor SMT model is available. Cannot run ensemble.")
```

```
Cài đặt/Kiểm tra sacrebleu và protobuf...
🌐 pip install ['sacrebleu', 'protobuf<5']
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 51.8/51.8 kB 1.7 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 104.1/104.1 kB 3.8 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 294.9/294.9 kB 10.1 MB/s eta 0:00:00
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
Cài đặt sacrebleu/protobuf thành công.
```

```
2025-11-13 12:55:57.294582: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1763038557.711973      19 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1763038557.818874      19 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
```

```
Sacrebleu loaded, version: 2.5.1
Loading Base Model from: /kaggle/input/number-1-noone-canbehere/nllb200-600m
Loading Adapter    from: /kaggle/input/number-1-noone-canbehere/nllb-lora-adapter-final
Loading SMT model  from: /kaggle/input/best-ml/smt_model_bt_mbr.pkl
Loading Data       from: /kaggle/input/mm-lo-so-2025
Creating train lookup dictionary (for exact + fuzzy copy)...
Loaded 20,000 pairs: Hindi <-> Bhilli
Loaded 20,000 pairs: Hindi <-> Gondi
Loaded 20,000 pairs: Hindi <-> Mundari
Loaded 20,000 pairs: English <-> Santali
Exact lookup example: Hindi->Bhilli has 19,575 entries.
Loading test file from: /kaggle/input/mm-lo-so-2025/test.csv
Loading Tokenizer from ADAPTER_PATH...
Loading Base NLLB model...
Attaching LoRA adapter...
✅ NLLB + LoRA loaded successfully.
Device map: {'model.shared': 0, 'lm_head': 0, 'model.decoder.embed_tokens': 0, 'model.encoder.embed_tokens': 0, 'model.encoder.embed_positions': 0, 'model.encoder.layers.0': 0, 'model.encoder.layers.1': 0, 'model.encoder.layers.2': 0, 'model.encoder.layers.3': 0, 'model.encoder.layers.4': 1, 'model.encoder.layers.5': 1, 'model.encoder.layers.6': 1, 'model.encoder.layers.7': 1, 'model.encoder.layers.8': 1, 'model.encoder.layers.9': 1, 'model.encoder.layers.10': 1, 'model.encoder.layers.11': 1, 'model.encoder.layer_norm': 1, 'model.decoder.embed_positions': 1, 'model.decoder.layers': 1, 'model.decoder.layer_norm': 1}
Loading SMT model from: /kaggle/input/best-ml/smt_model_bt_mbr.pkl
✅ SMT model loaded successfully.

🚀 Ensemble inference (Exact + Fuzzy Train-Lookup + SMT + NLLB + MBR Union)...
```

100%

 15999/15999 [4:39:43<00:00, 765.83it/s]

```
✅ Wrote final ENSEMBLE submission to: /kaggle/working/submission.csv
✅ Exact train-copy hits : 3534
✅ Fuzzy train-copy hits : 14
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
📈 Directions using NLLB: 8
📈 Directions using SMT : 8
```
