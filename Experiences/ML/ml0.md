Public Score  : 158.84476
Private  Score: 140.32171


# MMLoSo 2025 – Offline Baseline (word-by-word via Dice lexicon)
# ---------------------------------------------------------------
# Works with: /kaggle/input/mm-lo-so-2025/{bhili-train.csv,gondi-train.csv,mundari-train.csv,santali-train.csv,test.csv}
# Produces:   /kaggle/working/submission.csv

import os, re, math, gc, sys
import pandas as pd
from collections import defaultdict, Counter
from glob import glob

INPUT_DIR = "/kaggle/input"
# Try to find the competition folder heuristically
cand_dirs = [p for p in glob(f"{INPUT_DIR}/*") if os.path.isdir(p)]
COMP_DIR = None
for d in cand_dirs:
    if any(os.path.basename(fp).lower().startswith(("bhili","bhiili","gondi","mundari","santali")) 
           for fp in glob(f"{d}/*.csv")):
        COMP_DIR = d
        break
if COMP_DIR is None:
    # Fallback to a common folder name
    COMP_DIR = "/kaggle/input/mm-lo-so-2025"

print("Using competition directory:", COMP_DIR)

# ------------- Utilities -------------
def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()

def simple_tokenize(s: str):
    # keep punctuation as separate tokens; lowercase for lexical matching
    s = normalize_space(s)
    # Split on spaces but also separate punctuation
    s = re.sub(r"([.,!?;:()\[\]{}\"'“”‘’।|/\\\-])", r" \1 ", s)
    s = normalize_space(s)
    return s.split()

def detokenize(tokens):
    # Re-attach simple punctuation spacing heuristics
    out = []
    for i, t in enumerate(tokens):
        if i > 0 and t in {".", ",", "!", "?", ";", ":", ")", "”", "’", "।"}:
            # no space before these
            out[-1] = out[-1] + t
        elif t in {"(", "“", "‘"} and len(out) > 0:
            # no space after these
            out.append(t)
        else:
            out.append(t)
    # Fix ( ... and quotes spacing roughly
    txt = " ".join(out)
    txt = txt.replace("( ", "(").replace(" )", ")")
    txt = txt.replace("“ ", "“").replace(" ”", "”")
    txt = txt.replace("‘ ", "‘").replace(" ’", "’")
    return normalize_space(txt)

# Official label set (exact case/spelling required in submission)
OFFICIAL_LANGS = {"Bhilli","Hindi","Mundari","Gondi","English","Santali"}

# Some datasets may use 'Bhili' (one 'l'); map to official 'Bhilli' at submission time,
# but keep original for training file detection.
SUB_LANG_CANON = {
    "bhili":"Bhilli","bhilli":"Bhilli",
    "hindi":"Hindi","mundari":"Mundari","gondi":"Gondi",
    "english":"English","santali":"Santali"
}

def canon_label(lang: str) -> str:
    k = lang.strip().lower()
    return SUB_LANG_CANON.get(k, lang)

# ------------- Load training CSVs -------------
def read_train_pairs(csv_path):
    """Return tuple: (src_lang_name, tgt_lang_name, list of (src,tgt))"""
    df = pd.read_csv(csv_path)
    cols = [c.strip() for c in df.columns]
    # Heuristics: files have exactly two language columns (e.g., 'Hindi' & 'Bhili')
    # Find two text columns by name overlap with known languages (case-insensitive)
    lang_cols = []
    for c in cols:
        if c.strip().lower() in SUB_LANG_CANON:
            lang_cols.append(c)
    if len(lang_cols) < 2:
        # If headers are odd, fallback to first 2 columns
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
# Filter likely train files
train_files = [f for f in train_files if 'test' not in os.path.basename(f).lower()]
print("Train files found:", [os.path.basename(x) for x in train_files])

# Build a dictionary of training pairs for all directions
bitext_by_dir = defaultdict(list)  # key: (SRC,TGT) -> list[(src,tgt)]
for fp in train_files:
    try:
        s_lang, t_lang, pairs = read_train_pairs(fp)
        bitext_by_dir[(s_lang, t_lang)].extend(pairs)
        # Also store reversed direction
        bitext_by_dir[(t_lang, s_lang)].extend([(t, s) for (s, t) in pairs])
        print(f"Loaded {len(pairs):,} pairs: {s_lang} -> {t_lang}")
    except Exception as e:
        print("Skip", fp, "due to", e)

# ------------- Learn word lexicons with Dice coefficient -------------
def build_dice_lexicon(pairs, min_count=2, top_k=3):
    """
    Build word-to-word dictionary with Dice(A,B) = 2*C(A,B)/(C(A)+C(B)),
    where C counts sentence co-occurrence. Very fast & alignment-free.
    Returns: dict[src_word] = list of target words (ranked)
    """
    src_docs_contain = defaultdict(set)
    tgt_docs_contain = defaultdict(set)
    cooccur = defaultdict(int)
    # Document-level (sentence-level) presence
    for i, (s, t) in enumerate(pairs):
        s_tokens = set(simple_tokenize(s))
        t_tokens = set(simple_tokenize(t))
        for sw in s_tokens:
            src_docs_contain[sw].add(i)
        for tw in t_tokens:
            tgt_docs_contain[tw].add(i)
        for sw in s_tokens:
            for tw in t_tokens:
                cooccur[(sw, tw)] += 1

    src_count = {w: len(idx) for w, idx in src_docs_contain.items()}
    tgt_count = {w: len(idx) for w, idx in tgt_docs_contain.items()}

    lex = {}
    for sw, sc in src_count.items():
        if sc < min_count:
            continue
        cand = []
        # Iterate only tw that actually co-occurred with sw
        # To speed up, scan keys once
        # Build a local list
        for (w1, tw), cst in cooccur.items():
            if w1 != sw: 
                continue
            tc = tgt_count.get(tw, 0)
            if tc < min_count: 
                continue
            dice = 2.0 * cst / (sc + tc)
            cand.append((dice, tw))
        cand.sort(reverse=True)
        if cand:
            lex[sw] = [tw for _, tw in cand[:top_k]]
    return lex

lexicons = {}
for (s_lang, t_lang), pairs in bitext_by_dir.items():
    if len(pairs) == 0: 
        continue
    print(f"Building lexicon {s_lang} -> {t_lang} on {len(pairs):,} pairs ...")
    lexicons[(s_lang, t_lang)] = build_dice_lexicon(pairs, min_count=2, top_k=3)
    gc.collect()

# ------------- Translation -------------
def translate_tokens(src_tokens, lex):
    out = []
    for w in src_tokens:
        if w in lex:
            # choose most frequent/first candidate
            out.append(lex[w][0])
        else:
            # keep source token (copy) as a simple fallback
            out.append(w)
    return out

def translate_sentence(s, src_lang, tgt_lang):
    src_lang = canon_label(src_lang)
    tgt_lang = canon_label(tgt_lang)
    if (src_lang, tgt_lang) not in lexicons:
        # No lexicon learned (unexpected) -> copy
        toks = simple_tokenize(s)
        hyp = detokenize(toks) if toks else "."
        return hyp
    toks = simple_tokenize(s.lower())
    hyp_tokens = translate_tokens(toks, lexicons[(src_lang, tgt_lang)])
    hyp = detokenize(hyp_tokens)
    # Ensure not empty for Kaggle evaluator
    return hyp if hyp.strip() else "."

# ------------- Inference on test.csv -------------
test_path = os.path.join(COMP_DIR, "test.csv")
if not os.path.exists(test_path):
    # try alternative name
    cand = [p for p in glob(f"{COMP_DIR}/*.csv") if os.path.basename(p).lower().startswith("test")]
    if cand:
        test_path = cand[0]

test_df = pd.read_csv(test_path)
# Normalize header variations to exactly required names
# Accept common variants then rename
rename_map = {}
for c in test_df.columns:
    cn = c.strip()
    if cn.lower().replace("_"," ") == "row id":
        rename_map[c] = "Row ID"
    elif cn.lower() == "source lang":
        rename_map[c] = "Source Lang"
    elif cn.lower() == "source sentence":
        rename_map[c] = "Source Sentence"
    elif cn.lower() == "target lang":
        rename_map[c] = "Target Lang"
    elif cn.lower() == "target sentence":
        rename_map[c] = "Target Sentence"
test_df = test_df.rename(columns=rename_map)

required_cols = ["Row ID","Source Lang","Source Sentence","Target Lang","Target Sentence"]
for col in required_cols:
    if col not in test_df.columns:
        # create missing target sentence col
        if col == "Target Sentence":
            test_df[col] = ""
        else:
            raise ValueError(f"Missing required column in test.csv: {col}")

# Translate
preds = []
for idx, row in test_df.iterrows():
    src_lang = str(row["Source Lang"])
    tgt_lang = str(row["Target Lang"])
    src_sent = str(row["Source Sentence"])
    # Make sure we output canonical target language labels exactly as in spec
    # (but DO NOT change the test CSV values in these columns — submission will keep them)
    hyp = translate_sentence(src_sent, src_lang, tgt_lang)
    preds.append(hyp if hyp.strip() else ".")

test_df["Target Sentence"] = preds

# Final sanity: fill null/empty with "."
test_df["Target Sentence"] = test_df["Target Sentence"].fillna(".").apply(lambda s: s if str(s).strip() else ".")

# IMPORTANT: Keep exact column order and names
submission = test_df[required_cols].copy()

# Certain organizers are strict about valid language label values.
# If the test provided 'Bhili' instead of 'Bhilli', we still keep the original labels from test file
# because auto-eval checks the columns, not us transforming labels.

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
display(submission.head(10))
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
Wrote: /kaggle/working/submission.csv
Row ID	Source Lang	Source Sentence	Target Lang	Target Sentence
0	54334	Hindi	उन्होंने कहा कि 2014 के बाद, इस परियोजना को प्...	Bhilli	कि कि कि 2014। बाद। इना परियोजना ने प्रधानमंत्...
1	87641	Hindi	वित्तीय कठिनाइयों को हल करने में सहायक होने के...	Bhilli	वित्तीय वखा ने हल करवा मा सहायक थावा। हाते - ह...
2	32543	Hindi	मेरा सुझाव है कि हमारे सक्रिय दृष्टिकोण, नीतिय...	Bhilli	मारू सुझाव से कि हमारा सक्रिय दृष्टिकोण। नीतिय...
3	26313	Hindi	श्री मोदी ने कहा यह अटल जी ही थे जिन्होंने देश...	Bhilli	श्री मोदी ने कि यो अटल जी ही हता जिहुने देह नी...
4	83303	Hindi	उत्सवादि मनाने के उपलक्ष्य में सुरापान करना सा...	Bhilli	उत्सवादि मनावा। उपलक्ष्य मा सुरापान करवु साधार...
5	131411	Hindi	तुम्‍हारे साथ कभी ऐसा हुआ।	Bhilli	तुमरा हाते कदी ऐवु थायो।
6	101809	Hindi	यह सत्र ग्लासगो, यूनाइटेड किंगडम में आयोजित हुआ।	Bhilli	यो सत्र ग्लासगो। यूनाइटेड किंगडम मा आयोजित थायो।
7	59328	Hindi	यह 9 मार्च 2012 को रिलीज़ हुई थी, जिसे आम तौर ...	Bhilli	यो 9 मार्च 2012 ने रिलीज थाई हती। जिने आम तौर ...
8	57205	Hindi	नियुक्ति मामलों की मंत्रिमंडलीय समिति ने भारती...	Bhilli	नियुक्ति मामलों नी मंत्रिमंडलीय समिति ने भारती...
9	103641	Hindi	भारत को और अधिक साफ-सुथरा बनाने और बेहतर स्वच्...	Bhilli	भारत ने अने वदारे साफ - सुथरो बणावा अने बेहतर ...
