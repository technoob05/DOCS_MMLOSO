```
Public Score :  193.36209
Private Score : 153.90990

importos,re,math,gc,sys,random
importpandasaspd
fromcollectionsimport defaultdict, Counter
fromglobimport glob
importunicodedata  # Để chuẩn hóa Unicode
importpickle       # [MỚI] Thêm để lưu/tải model

# -------------------- Paths --------------------
INPUT_DIR = "/kaggle/input"
WORK_DIR  = "/kaggle/working"
cand_dirs = [p for p in glob(f"{INPUT_DIR}/*") if os.path.isdir(p)]
COMP_DIR = None
for d in cand_dirs:
    if any(os.path.basename(fp).lower().startswith(("bhili","bhiili","gondi","mundari","santali"))
           for fp in glob(f"{d}/*.csv")):
        COMP_DIR = d; break
if COMP_DIR is None:
    COMP_DIR = "/kaggle/input/mm-lo-so-2025"
print("Using competition directory:", COMP_DIR)

# -------------------- Utilities (Với NFKC + Tách số) --------------------
defnormalize_space(s: str) -> str:
    s = str(s)
    try:
        s = unicodedata.normalize('NFKC', s)
    except Exception:
        pass 
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
            if out: out[-1] = out[-1] + t
        elif t in {"(", "“", "‘"} and len(out) > 0:
            out.append(t)
        else:
            out.append(t)
    txt = " ".join(out)
    txt = txt.replace("( ", "(").replace(" )", ")")
    txt = txt.replace("“ ", "“").replace(" ”", "”").replace("‘ ", "‘").replace(" ’", "’")
    return normalize_space(txt)

SUB_LANG_CANON = {
    "bhili":"Bhilli","bhilli":"Bhilli","hindi":"Hindi","mundari":"Mundari",
    "gondi":"Gondi","english":"English","santali":"Santali"}
defcanon_label(lang: str) -> str:
    return SUB_LANG_CANON.get(lang.strip().lower(), lang)

FORWARD_DIRS = {("Hindi","Bhilli"), ("Hindi","Mundari"), ("Hindi","Gondi"), ("English","Santali")}
REVERSE_DIRS = {("Bhilli","Hindi"), ("Mundari","Hindi"), ("Gondi","Hindi"), ("Santali","English")}

# -------------------- Load training --------------------
defread_train_pairs(csv_path):
    df = pd.read_csv(csv_path)
    cols = [c.strip() for c in df.columns]
    lang_cols = [c for c in cols if c.strip().lower() in SUB_LANG_CANON]
    if len(lang_cols) < 2: lang_cols = cols[:2]
    src_col, tgt_col = lang_cols[0], lang_cols[1]
    src_name, tgt_name = canon_label(src_col), canon_label(tgt_col)
    pairs = []
    for s, t in zip(df[src_col].astype(str), df[tgt_col].astype(str)):
        s, t = normalize_space(s), normalize_space(t)
        if s and t: pairs.append((s, t))
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

# -------------------- Lexicon (IBM1 Diagonal) --------------------
NUM_RE   = re.compile(r"^[+\-]?\d+([.,:/-]\d+)*$")
URL_RE   = re.compile(r"(https?://|www\.)", re.I)
EMAIL_RE = re.compile(r".+@.+\..+")
PUNCT_SET = {".",",","!","?",";",":","(",")","[","]","{","}","–","—","-","—","|","/","\\","'","\"","“","”","‘","’","।"}
COMMON_SRC_STOPS = set([
    "के","की","का","और","कि","तो","में","पर","से","था","थे","है","Hूँ","है हैं","हो","ही","ने",
    "the","a","an","and","or","to","of","in","on","for","is","are","was","were"])

defis_copy_token(w: str) -> bool:
    w_ = w.strip()
    return (w_ in PUNCT_SET or NUM_RE.match(w_) is not None or
            URL_RE.search(w_) is not None or EMAIL_RE.match(w_) is not None)

defbuild_dice_tables(pairs, min_count=2):
    src_docs_contain, tgt_docs_contain, cooccur = defaultdict(set), defaultdict(set), defaultdict(int)
    for i,(s,t) in enumerate(pairs):
        s_set = set(simple_tokenize(s.lower()))
        t_set = set(simple_tokenize(t.lower()))
        for sw in s_set: src_docs_contain[sw].add(i)
        for tw in t_set: tgt_docs_contain[tw].add(i)
        for sw in s_set:
            for tw in t_set:
                cooccur[(sw,tw)] += 1
    src_count = {w: len(idx) for w,idx in src_docs_contain.items()}
    tgt_count = {w: len(idx) for w,idx in tgt_docs_contain.items()}
    dice_by_src = defaultdict(list)
    for (sw,tw),cst in cooccur.items():
        sc, tc = src_count.get(sw,0), tgt_count.get(tw,0)
        if sc>=min_count and tc>=min_count:
            dice = 2.0*cst/(sc+tc)
            dice_by_src[sw].append((dice,tw))
    for sw in dice_by_src:
        dice_by_src[sw].sort(reverse=True)
    return dice_by_src

definit_from_dice(dice_by_src, top_k=40):
    t_given_s = {}
    for sw, lst in dice_by_src.items():
        cands = [tw for _,tw in lst[:top_k]]
        if not cands: continue
        p = 1.0/len(cands)
        t_given_s[sw] = {tw:p for tw in cands}
    return t_given_s

defibm1_em_with_diag(pairs, t_given_s, iters=6, floor=1e-9, lambda_diag=4.0):
    for _ in range(iters):
        count = defaultdict(lambda: defaultdict(float))
        total_s = defaultdict(float)
        for s,t in pairs:
            s_tok = [w for w in simple_tokenize(s.lower())]
            t_tok = [w for w in simple_tokenize(t.lower())]
            s_ext = ["<NULL>"] + s_tok
            m, l = len(s_ext), len(t_tok)
            for j, tw in enumerate(t_tok, start=1):
                z = 0.0
                weights = []
                for i, sw in enumerate(s_ext, start=1):
                    base = t_given_s.get(sw, {}).get(tw, floor)
                    pos  = math.exp(-lambda_diag * abs((j / max(1,l)) - (i / max(1,m))))
                    wght = base * pos
                    weights.append((sw, wght))
                    z += wght
                if z == 0.0: z = floor * m
                for sw, wght in weights:
                    p = wght / z
                    count[sw][tw] += p
                    total_s[sw]   += p
        new_t = {}
        for sw, d in count.items():
            denom = total_s[sw] if total_s[sw] > 0 else 1.0
            dd = {tw: max(c/denom, floor) for tw, c in d.items()}
            new_t[sw] = dict(sorted(dd.items(), key=lambda x: x[1], reverse=True)[:100])
        t_given_s = new_t
    return t_given_s

defsymmetrize(t_ab, t_ba, thresh=1e-7):
    lex = {}
    for sw in set(t_ab.keys()):
        cand = set(t_ab.get(sw, {}).keys())
        bi = {tw for tw in cand if t_ba.get(tw, {}).get(sw, 0.0) > thresh}
        if bi:
            lex[sw] = {tw: 0.5*(t_ab[sw].get(tw,0)+t_ba.get(tw,{}).get(sw,0)) for tw in bi}
        else:
            tops = dict(sorted(t_ab.get(sw,{}).items(), key=lambda x: x[1], reverse=True)[:5])
            if tops: lex[sw] = tops
    return lex

defbuild_lexicon_ibm1_diag(pairs, dice_top=40, em_iter=6, min_count=2, lambda_diag=4.0):
    dice_src = build_dice_tables(pairs, min_count=min_count)
    t_ab = init_from_dice(dice_src, top_k=dice_top)
    rev_pairs = [(t,s) for (s,t) in pairs]
    dice_tgt = build_dice_tables(rev_pairs, min_count=min_count)
    t_ba = init_from_dice(dice_tgt, top_k=dice_top)
    t_ab = ibm1_em_with_diag(pairs,     t_ab, iters=em_iter, lambda_diag=lambda_diag)
    t_ba = ibm1_em_with_diag(rev_pairs, t_ba, iters=em_iter, lambda_diag=lambda_diag)
    lex_prob = symmetrize(t_ab, t_ba)
    for sw in list(lex_prob.keys()):
        if sw in COMMON_SRC_STOPS:
            top3 = dict(sorted(lex_prob[sw].items(), key=lambda x: x[1], reverse=True)[:3])
            s = sum(top3.values()) or 1.0
            lex_prob[sw] = {tw: (p/s)*0.5 for tw,p in top3.items()}
    return lex_prob

# -------------------- Word 3-gram LM (add-k) --------------------
# [SỬA] Trả về data (ngrams, counts) thay vì hàm (closure)
deftrain_word_trigram(sentences, add_k=0.1):
    ngram = defaultdict(Counter); context_count = Counter()
    for s in sentences:
        toks = ["<s>"] + simple_tokenize(s.lower()) + ["</s>"]
        for i in range(2, len(toks)):
            ctx = (toks[i-2], toks[i-1]); ngram[ctx][toks[i]] += 1; context_count[ctx] += 1
    # Trả về data, không phải hàm score
    return (ngram, context_count, add_k) 

# [MỚI] Hàm trợ giúp để tạo lại hàm score từ data đã lưu
defcreate_lm_score_fn(lm_data):
    if lm_data is None:
        # Trả về LM rỗng nếu không có data
        return lambda w, p2, p1: 0.0 
  
    ngram, context_count, add_k = lm_data
  
    defscore(next_word, prev2, prev1):
        ctx = (prev2, prev1)
        c = ngram[ctx][next_word]; total = context_count[ctx]
        # Dùng 1e5 làm vocab_size_guess giống code gốc
        return math.log((c + add_k) / (total + add_k*1e5))
  
    return score

# -------------------- Length ratio --------------------
defestimate_len_ratio(pairs):
    ratios = []
    for s,t in pairs:
        ls = max(1, len(simple_tokenize(s)))
        lt = max(1, len(simple_tokenize(t)))
        ratios.append(lt/ls)
    ratios.sort()
    return ratios[len(ratios)//2] if ratios else 1.0

# ---
# --- MBR Reranking Functions (BLEU + chrF3) ---
# ---
N_BEST_MBR = 10 # Số lượng N-best dùng cho MBR

defsoftmax(log_probs):
    if not log_probs: return []
    max_log = max(log_probs)
    exps = [math.exp(lp - max_log) for lp in log_probs]
    sum_exps = sum(exps)
    if sum_exps == 0:
        return [1.0 / len(log_probs)] * len(log_probs)
    return [e / sum_exps for e in exps]

defget_char_ngrams(s: str, n: int) -> Counter:
    return Counter(s[i:i+n] for i in range(len(s) - n + 1))

defsentence_chrf(hyp_tokens, ref_tokens, n_char=6, beta=3):
    hyp_str = " ".join(hyp_tokens)
    ref_str = " ".join(ref_tokens)
    hyp_ngrams = get_char_ngrams(hyp_str, n_char)
    ref_ngrams = get_char_ngrams(ref_str, n_char)
    common_ngrams = hyp_ngrams & ref_ngrams
    common_count = sum(common_ngrams.values())
    hyp_count = sum(hyp_ngrams.values())
    ref_count = sum(ref_ngrams.values())
    if hyp_count == 0 or ref_count == 0 or common_count == 0:
        return 0.0
    prec = common_count / hyp_count
    rec = common_count / ref_count
    f_beta = (1 + beta**2) * (prec * rec) / ((beta**2 * prec) + rec)
    return f_beta

defget_word_ngrams(tokens: list, n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

defsentence_bleu_smooth1(hyp_tokens, ref_tokens, max_n=4):
    log_bleu = 0.0
    for n in range(1, max_n + 1):
        hyp_ngrams = get_word_ngrams(hyp_tokens, n)
        ref_ngrams = get_word_ngrams(ref_tokens, n)
        common_count = 0
        for ng, count in hyp_ngrams.items():
            common_count += min(count, ref_ngrams[ng])
        hyp_len = max(0, len(hyp_tokens) - n + 1)
        prec = (common_count + 1) / (hyp_len + 1)
        log_bleu += math.log(prec)
    log_bleu /= max_n
    hyp_len = len(hyp_tokens)
    ref_len = len(ref_tokens)
    bp = 1.0
    if hyp_len < ref_len:
        bp = math.exp(1.0 - ref_len / max(1, hyp_len))
    return bp * math.exp(log_bleu)

defutility_gain(hyp_toks, ref_toks):
    chrf3 = sentence_chrf(hyp_toks, ref_toks, n_char=6, beta=3)
    bleu = sentence_bleu_smooth1(hyp_toks, ref_toks)
    return 0.6 * bleu + 0.4 * chrf3

defmbr_rerank(n_best_list):
    if not n_best_list: return []
    if len(n_best_list) == 1: return n_best_list[0][0]
    hyps = [entry[0] for entry in n_best_list]
    log_probs = [entry[3] for entry in n_best_list]
    probs = softmax(log_probs)
    mbr_scores = []
    for i, cand_hyp in enumerate(hyps):
        expected_gain = 0.0
        for j, ref_hyp in enumerate(hyps):
            gain = utility_gain(cand_hyp, ref_hyp)
            expected_gain += probs[j] * gain 
        mbr_scores.append(expected_gain)
    best_index = mbr_scores.index(max(mbr_scores))
    return hyps[best_index]

# ---
# --- N-best Beam Decoder ---
# ---
defbeam_decode_n_best(src_tokens, lex_prob, lm_score_fn, beam_size=10, lm_w=0.85,
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
                lm  = lm_score_fn(tw.lower(), p2.lower(), p1.lower()) # lm_score_fn đã được tạo
                rep = 0.0
                if len(hyp)>=2 and hyp[-1]==tw and hyp[-2]==tw: rep -= 3.0*rep_pen_w
                elif len(hyp)>=1 and hyp[-1]==tw: rep -= 1.0*rep_pen_w
                tgt_len = L+1; src_len = len(src_tokens); goal = len_ratio*src_len
                len_pen = -len_w * abs(tgt_len - goal)/max(1.0, goal)
                new_score = score + lp + lm_w*lm + rep + len_pen
                new_state = (p1, tw)
  
                if new_state not in hyp_cache or new_score > hyp_cache[new_state][3]:
                    hyp_cache[new_state] = (hyp+[tw], p1, tw, new_score, tgt_len)

        newB = list(hyp_cache.values())
        newB.sort(key=lambda x: x[3], reverse=True)
        Beam = newB[:beam_size]
  
    deffinal_score(entry):
        hyp, _, _, sc, L = entry
        goal = len_ratio*len(src_tokens)
        final_sc = sc + len_w * math.log(max(1.0, L) / max(1.0, goal))
        return (hyp, entry[1], entry[2], final_sc, L)

    final_beam = [final_score(entry) for entry in Beam]
    final_beam.sort(key=lambda x: x[3], reverse=True)
    return final_beam[:beam_size]

# -------------------- [MỚI] SMT Model Class --------------------
# Đóng gói mọi thứ vào một class để dễ dàng lưu/tải

classSMT_Model:
    def__init__(self):
        self.lex_prob_by_dir = {}
        self.len_ratio_by_dir = {}
        self.lm_data_by_tgt = {} # Sẽ lưu (ngram, context_count, add_k)
        self.params_fwd = {}
        self.params_rev = {}
        print("SMT_Model object initialized.")

    deftrain(self, bitext_by_dir, params_fwd, params_rev):
        self.params_fwd = params_fwd
        self.params_rev = params_rev
  
        fwd_pairs = {}
        rev_pairs = {}
        for (s_lang, t_lang), pairs in bitext_by_dir.items():
            if (s_lang, t_lang) in FORWARD_DIRS:
                fwd_pairs[(s_lang, t_lang)] = pairs
            elif (s_lang, t_lang) in REVERSE_DIRS:
                rev_pairs[(s_lang, t_lang)] = pairs

        # --- Vòng 1: Huấn luyện REVERSE ---
        print("\n--- [Vòng 1] Training REVERSE models (e.g., Bhilli -> Hindi) ---")
        for (s_lang, t_lang), pairs in rev_pairs.items():
            if not pairs: continue
            p = self.params_rev
            print(f"[LEX-DIAG-R1] {s_lang}->{t_lang} | EM={p['em_iter']} dice_top={p['dice_top']} lambda={p['lambda_diag']} pairs={len(pairs):,}")
  
            self.lex_prob_by_dir[(s_lang, t_lang)] = build_lexicon_ibm1_diag(
                pairs, dice_top=p['dice_top'], em_iter=p['em_iter'], min_count=2, lambda_diag=p['lambda_diag']
            )
  
            if t_lang not in self.lm_data_by_tgt:
                print(f"Training LM for: {t_lang}")
                tgt_sents = [t for _, t in pairs]
                for (s2, t2), p2 in bitext_by_dir.items():
                    if t2 == t_lang and s2 != s_lang:
                        tgt_sents.extend([t for _, t in p2])
                print(f"  -> LM for {t_lang} trained on {len(tgt_sents):,} sentences.")
                # [SỬA] Lưu data (ngram, count)
                self.lm_data_by_tgt[t_lang] = train_word_trigram(tgt_sents, add_k=0.1)
  
            self.len_ratio_by_dir[(s_lang, t_lang)] = estimate_len_ratio(pairs)
            gc.collect()

        # --- Vòng 2: Back-Translation ---
        print("\n--- [Vòng 2] Generating Back-Translated Data (Augmentation) ---")
        augmented_fwd_pairs = {}
        for (s_lang, t_lang), real_pairs in fwd_pairs.items():
            rev_key = (t_lang, s_lang)
            if rev_key not in self.lex_prob_by_dir:
                print(f"Warning: No reverse model {rev_key}. Skipping augmentation for {s_lang}->{t_lang}")
                augmented_fwd_pairs[(s_lang, t_lang)] = real_pairs
                continue
  
            print(f"Augmenting {s_lang}->{t_lang} using back-translation from {t_lang}->{s_lang}...")
            real_tgt_sents = [t for s, t in real_pairs]
            synthetic_src_sents = []
  
            for i, t_sent in enumerate(real_tgt_sents):
                # [SỬA] Gọi hàm dịch nội bộ (không có MBR)
                synth_s_sent = self._translate_for_backtrans(
                    t_sent, t_lang, s_lang, self.params_rev
                )
                synthetic_src_sents.append(synth_s_sent)

            synthetic_pairs = list(zip(synthetic_src_sents, real_tgt_sents))
            augmented_fwd_pairs[(s_lang, t_lang)] = real_pairs + synthetic_pairs
            print(f"  -> Augmented {s_lang}->{t_lang}: {len(real_pairs):,} real -> {len(augmented_fwd_pairs[(s_lang,t_lang)]):,} total pairs")
            gc.collect()

        # --- Vòng 3: Huấn luyện FORWARD ---
        print("\n--- [Vòng 3] Training FORWARD models on Augmented Data ---")
        for (s_lang, t_lang), aug_pairs in augmented_fwd_pairs.items():
            if not aug_pairs: continue
            p = self.params_fwd
            print(f"[LEX-DIAG-R2] {s_lang}->{t_lang} | EM={p['em_iter']} dice_top={p['dice_top']} lambda={p['lambda_diag']} pairs={len(aug_pairs):,}")
  
            self.lex_prob_by_dir[(s_lang, t_lang)] = build_lexicon_ibm1_diag(
                aug_pairs, 
                dice_top=p['dice_top'], em_iter=p['em_iter'], min_count=2, lambda_diag=p['lambda_diag']
            )
  
            if t_lang not in self.lm_data_by_tgt:
                print(f"Training LM for: {t_lang}")
                real_tgt_sents = [t for s,t in fwd_pairs.get((s_lang,t_lang), [])]
                print(f"  -> LM for {t_lang} trained on {len(real_tgt_sents):,} sentences.")
                # [SỬA] Lưu data (ngram, count)
                self.lm_data_by_tgt[t_lang] = train_word_trigram(real_tgt_sents, add_k=0.1)
  
            self.len_ratio_by_dir[(s_lang, t_lang)] = estimate_len_ratio(aug_pairs)
            gc.collect()
        print("--- SMT Model Training Complete ---")

    # Hàm dịch nội bộ (chỉ lấy top-1) cho Vòng 2
    def_translate_for_backtrans(self, s, src_lang, tgt_lang, params):
        toks = simple_tokenize(s); lower = [w.lower() for w in toks]
        key = (src_lang, tgt_lang)
        if key not in self.lex_prob_by_dir: return "."
  
        rev_lex = self.lex_prob_by_dir[key]
        # [SỬA] Tạo LM score fn từ data đã lưu
        lm_data = self.lm_data_by_tgt.get(tgt_lang)
        rev_lm = create_lm_score_fn(lm_data)
  
        rev_len_ratio = self.len_ratio_by_dir.get(key, 1.0)
  
        n_best = beam_decode_n_best(
            lower, rev_lex, rev_lm,
            beam_size=params['beam'], lm_w=params['lm_w'],
            rep_pen_w=params['rep_pen'], len_ratio=rev_len_ratio,
            len_w=params['len_w']
        )
        if not n_best: return "."
        best_hyp_tokens = n_best[0][0] # Lấy top-1
        return detokenize(best_hyp_tokens)

    # Hàm dịch chính (public) với N-best + MBR
    deftranslate_sentence(self, s, src_lang, tgt_lang):
        src_lang = canon_label(src_lang); tgt_lang = canon_label(tgt_lang)
        s = normalize_space(s)
        toks = simple_tokenize(s); lower = [w.lower() for w in toks]
        key = (src_lang, tgt_lang)
  
        if key not in self.lex_prob_by_dir:
            hyp = detokenize(toks) if toks else "."
            return hyp if hyp.strip() else "."
  
        lex_prob = self.lex_prob_by_dir[key]
        # [SỬA] Tạo LM score fn từ data đã lưu
        lm_data = self.lm_data_by_tgt.get(tgt_lang)
        lm_score = create_lm_score_fn(lm_data)
  
        len_r = self.len_ratio_by_dir.get(key, 1.0)
        params = self.params_fwd if key in FORWARD_DIRS else self.params_rev
  
        # 1. Tạo N-best list
        n_best_list = beam_decode_n_best(
            lower, lex_prob, lm_score,
            beam_size=params['beam'], # beam_size bây giờ là N_BEST_MBR
            lm_w=params['lm_w'],
            rep_pen_w=params['rep_pen'], len_ratio=len_r,
            len_w=params['len_w']
        )
  
        if not n_best_list:
            return "." # Không tìm thấy gì
  
        # 2. Rerank N-best list bằng MBR
        best_hyp_tokens = mbr_rerank(n_best_list)
  
        if len(best_hyp_tokens)>0 and toks and toks[0][:1].isupper():
            best_hyp_tokens[0] = best_hyp_tokens[0][:1].upper() + best_hyp_tokens[0][1:]
        hyp = detokenize(best_hyp_tokens)
        return hyp if hyp.strip() else "."

    # [MỚI] Hàm Save/Load
    defsave(self, filepath):
"""Lưu toàn bộ object SMT_Model vào file pickle."""
        print(f"Saving SMT model to {filepath}...")
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            print(f"✅ Model saved successfully to {filepath}")
        except Exception as e:
            print(f"⚠️ Error saving model: {e}")

    @staticmethod
    defload(filepath):
"""Tải SMT_Model từ file pickle."""
        print(f"Loading SMT model from {filepath}...")
        if not os.path.exists(filepath):
            print(f"⛔ Error: Model file not found at {filepath}")
            return None
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            print("✅ Model loaded successfully.")
            return model
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")
            return None

# -------------------- Main Execution --------------------

# Định nghĩa các tham số
PARAMS_FWD = dict(beam=N_BEST_MBR, lm_w=0.90, rep_pen=0.80, len_w=0.12, em_iter=7, dice_top=40, lambda_diag=4.0)
PARAMS_REV = dict(beam=N_BEST_MBR, lm_w=0.70, rep_pen=0.60, len_w=0.10, em_iter=5, dice_top=30, lambda_diag=4.0)

# Khởi tạo và Huấn luyện model
smt_model = SMT_Model()
smt_model.train(bitext_by_dir, PARAMS_FWD, PARAMS_REV)

# [MỚI] Lưu model đã huấn luyện ra output
MODEL_SAVE_PATH = f"{WORK_DIR}/smt_model_bt_mbr.pkl"
smt_model.save(MODEL_SAVE_PATH)

# -------------------- Read test.csv and translate --------------------
test_path = os.path.join(COMP_DIR, "test.csv")
if not os.path.exists(test_path):
    cand = [p for p in glob(f"{COMP_DIR}/*.csv") if os.path.basename(p).lower().startswith("test")]
    if cand: test_path = cand[0]

test_df = pd.read_csv(test_path)
rename_map = {}
for c in test_df.columns:
    cn = c.strip().lower().replace("_"," ")
    if cn == "row id": rename_map[c] = "Row ID"
    elif cn == "source lang": rename_map[c] = "Source Lang"
    elif cn == "source sentence": rename_map[c] = "Source Sentence"
    elif cn == "target lang": rename_map[c] = "Target Lang"
    elif cn == "target sentence": rename_map[c] = "Target Sentence"
test_df = test_df.rename(columns=rename_map)
required_cols = ["Row ID","Source Lang","Source Sentence","Target Lang","Target Sentence"]
for col in required_cols:
    if col not in test_df.columns:
        if col == "Target Sentence": test_df[col] = ""
        else: raise ValueError(f"Missing required column in test.csv: {col}")

print("\n--- Starting Final Inference (with MBR Reranking) ---")
preds = []
# [SỬA] Dùng method của object model đã huấn luyện
for i, row in test_df.iterrows():
    hyp = smt_model.translate_sentence(str(row["Source Sentence"]), str(row["Source Lang"]), str(row["Target Lang"]))
    preds.append(hyp if hyp.strip() else ".")
    if (i+1) % 500 == 0:
        print(f"Processed {i+1} / {len(test_df)}")

test_df["Target Sentence"] = preds
test_df["Target Sentence"] = test_df["Target Sentence"].fillna(".").apply(lambda s: s if str(s).strip() else ".")
submission = test_df[required_cols].copy()
out_path = f"{WORK_DIR}/submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)

try:
    fromIPython.displayimport display
    display(submission.head(10))
except Exception:
    print(submission.head(10).to_string(index=False))

print(f"\nSMT Model (cho Ensemble) đã được lưu tại: {MODEL_SAVE_PATH}")
```

```
Using competition directory: /kaggle/input/mm-lo-so-2025
Train files found: ['bhili-train.csv', 'gondi-train.csv', 'mundari-train.csv', 'santali-train.csv']
Loaded 20,000 pairs: Hindi -> Bhilli
Loaded 20,000 pairs: Hindi -> Gondi
Loaded 20,000 pairs: Hindi -> Mundari
Loaded 20,000 pairs: English -> Santali
SMT_Model object initialized.

--- [Vòng 1] Training REVERSE models (e.g., Bhilli -> Hindi) ---
[LEX-DIAG-R1] Bhilli->Hindi | EM=5 dice_top=30 lambda=4.0 pairs=20,000
Training LM for: Hindi
  -> LM for Hindi trained on 60,000 sentences.
[LEX-DIAG-R1] Gondi->Hindi | EM=5 dice_top=30 lambda=4.0 pairs=20,000
[LEX-DIAG-R1] Mundari->Hindi | EM=5 dice_top=30 lambda=4.0 pairs=20,000
[LEX-DIAG-R1] Santali->English | EM=5 dice_top=30 lambda=4.0 pairs=20,000
Training LM for: English
  -> LM for English trained on 20,000 sentences.

--- [Vòng 2] Generating Back-Translated Data (Augmentation) ---
Augmenting Hindi->Bhilli using back-translation from Bhilli->Hindi...
  -> Augmented Hindi->Bhilli: 20,000 real -> 40,000 total pairs
Augmenting Hindi->Gondi using back-translation from Gondi->Hindi...
  -> Augmented Hindi->Gondi: 20,000 real -> 40,000 total pairs
Augmenting Hindi->Mundari using back-translation from Mundari->Hindi...
  -> Augmented Hindi->Mundari: 20,000 real -> 40,000 total pairs
Augmenting English->Santali using back-translation from Santali->English...
  -> Augmented English->Santali: 20,000 real -> 40,000 total pairs

--- [Vòng 3] Training FORWARD models on Augmented Data ---
[LEX-DIAG-R2] Hindi->Bhilli | EM=7 dice_top=40 lambda=4.0 pairs=40,000
Training LM for: Bhilli
  -> LM for Bhilli trained on 20,000 sentences.
[LEX-DIAG-R2] Hindi->Gondi | EM=7 dice_top=40 lambda=4.0 pairs=40,000
Training LM for: Gondi
  -> LM for Gondi trained on 20,000 sentences.
[LEX-DIAG-R2] Hindi->Mundari | EM=7 dice_top=40 lambda=4.0 pairs=40,000
Training LM for: Mundari
  -> LM for Mundari trained on 20,000 sentences.
[LEX-DIAG-R2] English->Santali | EM=7 dice_top=40 lambda=4.0 pairs=40,000
Training LM for: Santali
  -> LM for Santali trained on 20,000 sentences.
--- SMT Model Training Complete ---
Saving SMT model to /kaggle/working/smt_model_bt_mbr.pkl...
✅ Model saved successfully to /kaggle/working/smt_model_bt_mbr.pkl

--- Starting Final Inference (with MBR Reranking) ---
Processed 500 / 15999
Processed 1000 / 15999
Processed 1500 / 15999
Processed 2000 / 15999
Processed 2500 / 15999
Processed 3000 / 15999
Processed 3500 / 15999
Processed 4000 / 15999
Processed 4500 / 15999
Processed 5000 / 15999
Processed 5500 / 15999
Processed 6000 / 15999
Processed 6500 / 15999
Processed 7000 / 15999
Processed 7500 / 15999
Processed 8000 / 15999
Processed 8500 / 15999
Processed 9000 / 15999
Processed 9500 / 15999
Processed 10000 / 15999
Processed 10500 / 15999
Processed 11000 / 15999
Processed 11500 / 15999
Processed 12000 / 15999
Processed 12500 / 15999
Processed 13000 / 15999
Processed 13500 / 15999
Processed 14000 / 15999
Processed 14500 / 15999
Processed 15000 / 15999
Processed 15500 / 15999
Wrote: /kaggle/working/submission.csv
```

|   | Row ID | Source Lang | Source Sentence                                                                           | Target Lang | Target Sentence                                                                            |
| - | ------ | ----------- | ----------------------------------------------------------------------------------------- | ----------- | ------------------------------------------------------------------------------------------ |
| 0 | 54334  | Hindi       | उन्होंने कहा कि 2014 के बाद, इस परियोजना को प्...         | Bhilli      | तिहुने किदु कि 2014 ना बाद, इनी परियोजना ने प्...          |
| 1 | 87641  | Hindi       | वित्तीय कठिनाइयों को हल करने में सहायक होने के...   | Bhilli      | वित्तीय काहटियु ने हल करवा मां मदतगार थावा ना ...     |
| 2 | 32543  | Hindi       | मेरा सुझाव है कि हमारे सक्रिय दृष्टिकोण, नीतिय...   | Bhilli      | मारो सुझाव से कि हमारा सक्रिय दृष्टिकोण, नीतिय...    |
| 3 | 26313  | Hindi       | श्री मोदी ने कहा यह अटल जी ही थे जिन्होंने देश...     | Bhilli      | श्री मोदी यी केदु यो अटल जी ही हता जिहुने देह ...       |
| 4 | 83303  | Hindi       | उत्सवादि मनाने के उपलक्ष्य में सुरापान करना सा...  | Bhilli      | उत्सवादि मनावा ना उपलक्ष्य मा सुरापान करवानु स...   |
| 5 | 131411 | Hindi       | तुम्‍हारे साथ कभी ऐसा हुआ।                                          | Bhilli      | तुमरा हाते कदी ऐवु थायो।                                               |
| 6 | 101809 | Hindi       | यह सत्र ग्लासगो, यूनाइटेड किंगडम में आयोजित हुआ।  | Bhilli      | यो सत्र ग्लासगो, युनाइटेड किंगडम मे आयोजित थायों। |
| 7 | 59328  | Hindi       | यह 9 मार्च 2012 को रिलीज़ हुई थी, जिसे आम तौर ...            | Bhilli      | यो 9 मार्च 2012 ने रिलीज थाई हती, जीने आम तोर ...             |
| 8 | 57205  | Hindi       | नियुक्ति मामलों की मंत्रिमंडलीय समिति ने भारती... | Bhilli      | नियुक्ति मामलों नी मंत्रिमंडलीय समिति यी भारती...  |
| 9 | 103641 | Hindi       | भारत को और अधिक साफ-सुथरा बनाने और बेहतर स्वच्...    | Bhilli      | भारत ने अने वदारे साफ - सुथरो बणावा अने बेहतर ...       |

```
SMT Model (cho Ensemble) đã được lưu tại: /kaggle/working/smt_model_bt_mbr.pkl
```
