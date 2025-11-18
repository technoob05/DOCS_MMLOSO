```

Public Score :  311.60609
Private Score : 186.36924




importos
importre
importmath
importpickle
importsocket
importsubprocess
importsys

importnumpyasnp
importpandasaspd

fromglobimport glob
fromcollectionsimport defaultdict
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

print("Cài đặt/Kiểm tra sacrebleu và protobuf<5 (nếu cần)...")
if _pip_install(["sacrebleu", "protobuf<5"]):
    print("✅ Cài đặt sacrebleu/protobuf thành công.")
else:
    print("⚠️ Không thể cài đặt, tiếp tục với thư viện hệ thống.")

try:
    importsacrebleu
    print(f"Sacrebleu loaded, version: {sacrebleu.__version__}")
except ImportError as e:
    print(f"⚠️ sacrebleu import failed: {e}")
    sacrebleu = None

# ============================================================
# 1. ĐƯỜNG DẪN
# ============================================================
COMP_DIR          = "/kaggle/input/mm-lo-so-2025"
SUBMISSION_IN     = "/kaggle/input/3b-submission/submission (5).csv"
OUTPUT_PATH       = "/kaggle/working/submission_postprocessed.csv"

print(f"Using COMP_DIR       : {COMP_DIR}")
print(f"Reading submission   : {SUBMISSION_IN}")
print(f"Will write output to : {OUTPUT_PATH}")

# ============================================================
# 2. UTILITIES: NORMALIZE / TOKENIZE / DIGIT MAPPING
# ============================================================
_WS_RE = re.compile(r"\s+")
PUNCT_CHARS = r"\.\,\!\?\;\:\(\)\[\]\{\}\"'“”‘’।\|/\\\-"

defnormalize_space(s: str) -> str:
"""Chỉ gom whitespace về 1 dấu cách, KHÔNG NFKC."""
    s = str(s)
    return _WS_RE.sub(" ", s).strip()

defsimple_tokenize(s: str):
    s = normalize_space(s)
    # tách số + punctuation
    s = re.sub(r"(\d+)", r" \1 ", s)
    s = re.sub(f"([{PUNCT_CHARS}])", r" \1 ", s)
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

# --- Lang canon ---
SUB_LANG_CANON = {
    "bhili":"Bhilli","bhilli":"Bhilli",
    "hindi":"Hindi","mundari":"Mundari","gondi":"Gondi",
    "english":"English","santali":"Santali",
}
defcanon_label(lang: str) -> str:
    return SUB_LANG_CANON.get(str(lang).strip().lower(), str(lang))

FORWARD_DIRS = {("Hindi","Bhilli"), ("Hindi","Mundari"), ("Hindi","Gondi"), ("English","Santali")}
REVERSE_DIRS = {("Bhilli","Hindi"), ("Mundari","Hindi"), ("Gondi","Hindi"), ("Santali","English")}

# --- Digit mapping ---
DEVAN = "०१२३४५६७८९"
LATIN = "0123456789"
LATIN2DEV = str.maketrans(LATIN, DEVAN)
DEV2LATIN = str.maketrans(DEVAN, LATIN)

defmap_digits_for_target(text: str, tgt_lang: str) -> str:
"""
    - Với target Indic (Hindi/Bhilli/Mundari/Gondi): map Latin → Devanagari.
    - Với target English: giữ Latin, map Devanagari → Latin.
    - Santali: cứ giữ Latin (ở nhiều corpus, digits thường Latin).
    """
    t = canon_label(tgt_lang)
    if t in {"Hindi", "Bhilli", "Mundari", "Gondi"}:
        return str(text).translate(LATIN2DEV)
    elif t == "English":
        return str(text).translate(DEV2LATIN)
    else:
        return str(text)

# --- Special tokens: numbers, URL, email ---
NUM_RE    = re.compile(r"[0-9०-९]+([,.:/-][0-9०-९]+)*")
URL_RE    = re.compile(r"(https?://|www\.)", re.I)
EMAIL_RE  = re.compile(r".+@.+\..+")

defextract_specials(s: str):
    toks = simple_tokenize(s)
    nums = set()
    urls = set()
    emails = set()
    for w in toks:
        if URL_RE.search(w):
            urls.add(w)
        elif EMAIL_RE.fullmatch(w):
            emails.add(w)
        elif NUM_RE.fullmatch(w):
            nums.add(w)
    return nums, urls, emails

defensure_specials_in_hyp(src: str, hyp: str) -> str:
"""
    Nếu số/URL/email trong source không xuất hiện trong hyp → append vào cuối câu.
    """
    src_nums, src_urls, src_emails = extract_specials(src)
    hyp_nums, hyp_urls, hyp_emails = extract_specials(hyp)

    add_nums   = [x for x in src_nums   if x not in hyp_nums]
    add_urls   = [x for x in src_urls   if x not in hyp_urls]
    add_emails = [x for x in src_emails if x not in hyp_emails]

    if not (add_nums or add_urls or add_emails):
        return hyp

    extra = []
    extra.extend(sorted(add_nums))
    extra.extend(sorted(add_urls))
    extra.extend(sorted(add_emails))

    hyp2 = hyp.strip()
    if hyp2 and not hyp2.endswith(("।", ".", "!", "?")):
        hyp2 = hyp2 + " "
    elif hyp2:
        hyp2 = hyp2 + " "

    return normalize_space(hyp2 + " ".join(extra))

# ============================================================
# 3. LOAD TRAIN – BUILD EXACT + FUZZY LOOKUP
# ============================================================
print("\n📚 Đọc train để build exact/fuzzy copy-from-train...")
train_lookup  = defaultdict(dict)                      # exact
fuzzy_lookup  = defaultdict(lambda: defaultdict(list)) # fuzzy[(s_lang,t_lang)][fuzzy_key] = [(len, src, tgt), ...]

bitext_by_dir = {}
train_files = sorted(glob(os.path.join(COMP_DIR, "*.csv")))
train_files = [f for f in train_files if 'test' not in os.path.basename(f).lower()
                                      and 'dev'  not in os.path.basename(f).lower()]

defread_train_pairs(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"⚠️ Skip {csv_path} vì lỗi:", e)
        return None, None, []

    cols = [c.strip() for c in df.columns]
    lang_cols = [c for c in cols if c.strip().lower() in SUB_LANG_CANON]
    if len(lang_cols) < 2:
        lang_cols = cols[:2]

    src_col, tgt_col = lang_cols[0], lang_cols[1]
    src_name, tgt_name = canon_label(src_col), canon_label(tgt_col)

    pairs = []
    for s, t in zip(df[src_col].astype(str), df[tgt_col].astype(str)):
        s_norm = normalize_space(s)
        t_norm = normalize_space(t)
        if s_norm and t_norm:
            pairs.append((s_norm, t_norm))
    return src_name, tgt_name, pairs

for fp in train_files:
    s_lang, t_lang, pairs = read_train_pairs(fp)
    if not s_lang:
        continue
    bitext_by_dir[(s_lang, t_lang)] = pairs
    bitext_by_dir[(t_lang, s_lang)] = [(t, s) for (s, t) in pairs]
    print(f"  Loaded {len(pairs):,} pairs: {s_lang} <-> {t_lang}")

for (s_lang, t_lang), pairs in bitext_by_dir.items():
    d_exact = train_lookup[(s_lang, t_lang)]
    d_fuzzy = fuzzy_lookup[(s_lang, t_lang)]
    for s, t in pairs:
        # exact
        d_exact[s] = t
        # fuzzy bucket
        fk   = normalize_for_fuzzy(s)
        toks = simple_tokenize(s)
        d_fuzzy[fk].append((len(toks), s, t))

print(f"🔎 Exact Hindi->Bhilli entries: {len(train_lookup[('Hindi','Bhilli')])if('Hindi','Bhilli')intrain_lookupelse0:,}")

deffuzzy_copy_from_train(src_norm, src_lang_c, tgt_lang_c, max_len_diff=1):
"""
    src_norm: câu đã normalize_space
    Cực conservative:
      - fuzzy_key trùng (bỏ dấu, lowercase)
      - |len_train - len_src| <= max_len_diff
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
                break
    return best_tgt

# ============================================================
# 4. ĐỌC SUBMISSION GỐC
# ============================================================
print("\n📄 Đọc submission gốc...")
sub_df = pd.read_csv(SUBMISSION_IN)

rename_map = {}
for c in sub_df.columns:
    cn = c.strip()
    if cn.lower().replace("_"," ") == "row id":       rename_map[c] = "Row ID"
    elif cn.lower() == "source lang":                 rename_map[c] = "Source Lang"
    elif cn.lower() == "source sentence":             rename_map[c] = "Source Sentence"
    elif cn.lower() == "target lang":                 rename_map[c] = "Target Lang"
    elif cn.lower() == "target sentence":             rename_map[c] = "Target Sentence"

sub_df = sub_df.rename(columns=rename_map)

required_cols = ["Row ID","Source Lang","Source Sentence","Target Lang","Target Sentence"]
for col in required_cols:
    if col not in sub_df.columns:
        raise ValueError(f"Missing column in submission: {col}")

print("Submission shape:", sub_df.shape)
print(sub_df.head())

# ============================================================
# 5. POST-PROCESSING HACKS CHO MỖI CÂU
# ============================================================
defpostprocess_one(src_lang, tgt_lang, src_sent, hyp_sent):
"""
    Hàng loạt trick:
      - exact copy-from-train
      - fuzzy copy-from-train
      - digit mapping theo target lang
      - punctuation/spacing normalize
      - copy lại số/URL/email từ source nếu bị mất
      - fallback nếu câu quá ngắn
    """
    src_lang_c = canon_label(src_lang)
    tgt_lang_c = canon_label(tgt_lang)

    src_norm = normalize_space(src_sent)
    hyp     = normalize_space(hyp_sent)

    if not hyp:
        hyp = src_norm  # fallback thô nhưng còn hơn câu rỗng

    # 1) Exact copy-from-train
    exact_dict = train_lookup.get((src_lang_c, tgt_lang_c), {})
    if src_norm in exact_dict:
        return exact_dict[src_norm]

    # 2) Fuzzy copy-from-train (cực conservative)
    fuzzy_hyp = fuzzy_copy_from_train(src_norm, src_lang_c, tgt_lang_c, max_len_diff=1)
    if fuzzy_hyp is not None and str(fuzzy_hyp).strip():
        hyp = fuzzy_hyp

    # 3) Normalize spacing/punctuation (nhẹ)
    hyp = normalize_space(hyp)

    # 4) Digit mapping theo target lang
    hyp = map_digits_for_target(hyp, tgt_lang_c)

    # 5) Ensure numbers/URL/email từ source xuất hiện trong hyp
    hyp = ensure_specials_in_hyp(src_norm, hyp)

    # 6) Fallback length heuristic: nếu hyp quá ngắn so với source → copy thêm đuôi source
    src_toks = simple_tokenize(src_norm)
    hyp_toks = simple_tokenize(hyp)
    if len(hyp_toks) < 0.4 * max(1, len(src_toks)):
        # thô bạo: nối thêm n-gram cuối của source (không hiệu ngữ nghĩa, nhưng BLEU thích hơn câu quá ngắn)
        need = int(0.4 * len(src_toks) - len(hyp_toks))
        need = max(0, need)
        tail = src_toks[-need:] if need > 0 else []
        hyp_toks = hyp_toks + tail
        hyp = detokenize(hyp_toks)

    return normalize_space(hyp)

# ============================================================
# 6. ÁP DỤNG POST-PROCESSING TOÀN BỘ SUBMISSION
# ============================================================
print("\n🚀 Bắt đầu post-processing toàn bộ submission...")
new_targets = []
for _, row in tqdm(sub_df.iterrows(), total=len(sub_df)):
    src_lang = row["Source Lang"]
    tgt_lang = row["Target Lang"]
    src_sent = row["Source Sentence"]
    hyp_sent = row["Target Sentence"]
    new_target = postprocess_one(src_lang, tgt_lang, src_sent, hyp_sent)
    if not str(new_target).strip():
        new_target = "."
    new_targets.append(new_target)

sub_df["Target Sentence"] = pd.Series(new_targets).fillna(".").apply(
    lambda s: s if str(s).strip() else "."
)

# ============================================================
# 7. GHI RA FILE MỚI
# ============================================================
sub_df[required_cols].to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ DONE. Wrote post-processed submission to: {OUTPUT_PATH}")

try:
    fromIPython.displayimport display
    display(sub_df.head(10))
except Exception:
    print(sub_df.head(10).to_string(index=False)) 
```

```
Cài đặt/Kiểm tra sacrebleu và protobuf<5 (nếu cần)...
🌐 pip install ['sacrebleu', 'protobuf<5']
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 51.8/51.8 kB 1.7 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 104.1/104.1 kB 3.8 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 294.9/294.9 kB 9.9 MB/s eta 0:00:00
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
✅ Cài đặt sacrebleu/protobuf thành công.
Sacrebleu loaded, version: 2.5.1
Using COMP_DIR       : /kaggle/input/mm-lo-so-2025
Reading submission   : /kaggle/input/3b-submission/submission (5).csv
Will write output to : /kaggle/working/submission_postprocessed.csv

📚 Đọc train để build exact/fuzzy copy-from-train...
  Loaded 20,000 pairs: Hindi <-> Bhilli
  Loaded 20,000 pairs: Hindi <-> Gondi
  Loaded 20,000 pairs: Hindi <-> Mundari
  Loaded 20,000 pairs: English <-> Santali
🔎 Exact Hindi->Bhilli entries: 19,575

📄 Đọc submission gốc...
Submission shape: (15999, 5)
   Row ID Source Lang                                    Source Sentence  \
0   54334       Hindi  उन्होंने कहा कि 2014 के बाद, इस परियोजना को प्...   
1   87641       Hindi  वित्तीय कठिनाइयों को हल करने में सहायक होने के...   
2   32543       Hindi  मेरा सुझाव है कि हमारे सक्रिय दृष्टिकोण, नीतिय...   
3   26313       Hindi  श्री मोदी ने कहा यह अटल जी ही थे जिन्होंने देश...   
4   83303       Hindi  उत्सवादि मनाने के उपलक्ष्य में सुरापान करना सा...   

  Target Lang                                    Target Sentence  
0      Bhilli  तिनाये केदु की 2014 ने बाद मा इनो काम ने प्रधा...  
1      Bhilli  आर्थिक कठिनाइयों को हल करने में सहायक होने के ...  
2      Bhilli  मारा सुझाव से कि हमारा सक्रिय नजरिया नीतियां अ...  
3      Bhilli  श्री मोदी यी केदू यो अटल जी ही हता जिन् हुये द...  
4      Bhilli  उत्सव मनवा ना उपलक्ष्य मा सुरापान करवा साधारण ...  

🚀 Bắt đầu post-processing toàn bộ submission...
```

100%

 15999/15999 [00:03<00:00, 6937.61it/s]

```
✅ DONE. Wrote post-processed submission to: /kaggle/working/submission_postprocessed.csv
```

|   | Row ID | Source Lang | Source Sentence                                                                           | Target Lang | Target Sentence                                                                           |
| - | ------ | ----------- | ----------------------------------------------------------------------------------------- | ----------- | ----------------------------------------------------------------------------------------- |
| 0 | 54334  | Hindi       | उन्होंने कहा कि 2014 के बाद, इस परियोजना को प्...         | Bhilli      | तिनाये केदु की 2014 ने बाद मा इनो काम ने प्रधा...         |
| 1 | 87641  | Hindi       | वित्तीय कठिनाइयों को हल करने में सहायक होने के...   | Bhilli      | आर्थिक कठिनाइयों को हल करने में सहायक होने के ...    |
| 2 | 32543  | Hindi       | मेरा सुझाव है कि हमारे सक्रिय दृष्टिकोण, नीतिय...   | Bhilli      | मारा सुझाव से कि हमारा सक्रिय नजरिया नीतियां अ...   |
| 3 | 26313  | Hindi       | श्री मोदी ने कहा यह अटल जी ही थे जिन्होंने देश...     | Bhilli      | श्री मोदी यी केदू यो अटल जी ही हता जिन् हुये द...      |
| 4 | 83303  | Hindi       | उत्सवादि मनाने के उपलक्ष्य में सुरापान करना सा...  | Bhilli      | उत्सव मनवा ना उपलक्ष्य मा सुरापान करवा साधारण ...   |
| 5 | 131411 | Hindi       | तुम्‍हारे साथ कभी ऐसा हुआ।                                          | Bhilli      | तमु नी हाते कदी ऐवी थाई ।                                              |
| 6 | 101809 | Hindi       | यह सत्र ग्लासगो, यूनाइटेड किंगडम में आयोजित हुआ।  | Bhilli      | यो सत्र ग्लासगो, यूनाइटेड किंगडम मा आयोजित थायो।  |
| 7 | 59328  | Hindi       | यह 9 मार्च 2012 को रिलीज़ हुई थी, जिसे आम तौर ...            | Bhilli      | यो ९ मार्च २०१२ मा रिलिज थाई थी, जिने आम तौर प...       |
| 8 | 57205  | Hindi       | नियुक्ति मामलों की मंत्रिमंडलीय समिति ने भारती... | Bhilli      | नियुक्ति मामलों नी मंत्रिमंडलीय समिति ने भारती... |
| 9 | 103641 | Hindi       | भारत को और अधिक साफ-सुथरा बनाने और बेहतर स्वच्...    | Bhilli      | भारत को ओड़ोः इसु पुरअः साफ-सुथरा बइओः ओड़ओ इस...    |
