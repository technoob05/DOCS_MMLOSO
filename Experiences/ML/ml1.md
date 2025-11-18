```
Public Score : 175.82615

Private Score : 143.91296

# ========================= MMLoSo 2025 – IBM1(+Diagonal Prior) + KN word LM + char-LM + Length Bonus =========================
# Mục tiêu: Kết hợp bản điểm cao (IBM1 + diagonal prior) với char-LM fusion + length bonus + KN 3-gram LM
# Tùy chọn: DEV tuner (grid) cho char_lm_w ∈ [0.1,0.4], len_w ∈ [0.05,0.25]
# -----------------------------------------------------------------------------------------------------------------------------

importos,re,math,gc,sys,random
importpandasaspd
fromcollectionsimport defaultdict, Counter
fromglobimport glob

# -------------------- Switches --------------------
DEV_TUNE = False               # True: chạy tuner dev rồi infer test bằng best (char_lm_w, len_w)
DEV_MAX_PER_DIR = 2000         # tối đa cặp/hướng dùng cho dev
GRID_CHAR_W = [0.10, 0.20, 0.30, 0.40]
GRID_LEN_W  = [0.05, 0.10, 0.15, 0.20, 0.25]

# Diagonal prior (fast_align-style)
USE_DIAG_PRIOR = True
LAMBDA_DIAG    = 4.0           # thử {2,4,6}

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

# -------------------- Tokenization / Detok --------------------
defnormalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()

defsimple_tokenize(s: str):
    s = normalize_space(s)
    s = re.sub(r"([.,!?;:()\[\]{}\"'“”‘’।|/\\\-])", r" \1 ", s)
    s = normalize_space(s)
    return s.split()

defdetokenize(tokens):
    out=[]
    for i,t in enumerate(tokens):
        if i>0 and t in {".",",","!","?",";",":",")","”","’","।"}:
            out[-1]=out[-1]+t
        elif t in {"(","“","‘"} and len(out)>0:
            out.append(t)
        else:
            out.append(t)
    txt=" ".join(out)
    txt=txt.replace("( ","(").replace(" )",")")
    txt=txt.replace("“ ","“").replace(" ”","”").replace("‘ ","‘").replace(" ’","’")
    return normalize_space(txt)

# -------------------- Lang labels --------------------
SUB_LANG_CANON = {
    "bhili":"Bhilli","bhilli":"Bhilli","hindi":"Hindi","mundari":"Mundari",
    "gondi":"Gondi","english":"English","santali":"Santali"
}
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
    pairs=[]
    for s,t in zip(df[src_col].astype(str), df[tgt_col].astype(str)):
        s,t = normalize_space(s), normalize_space(t)
        if s and t: pairs.append((s,t))
    return src_name, tgt_name, pairs

train_files = sorted(glob(f"{COMP_DIR}/*.csv"))
train_files = [f for f in train_files if 'test' not in os.path.basename(f).lower()]
print("Train files found:", [os.path.basename(x) for x in train_files])

bitext_by_dir = defaultdict(list)
for fp in train_files:
    try:
        s_lang, t_lang, pairs = read_train_pairs(fp)
        bitext_by_dir[(s_lang, t_lang)].extend(pairs)
        bitext_by_dir[(t_lang, s_lang)].extend([(t,s) for (s,t) in pairs])
        print(f"Loaded {len(pairs):,} pairs: {s_lang} -> {t_lang}")
    except Exception as e:
        print("Skip", fp, "due to", e)

# -------------------- Lexicon (Dice -> IBM1 (diag) two-way + symmetrize) --------------------
NUM_RE   = re.compile(r"^[+\-]?\d+([.,:/-]\d+)*$")
URL_RE   = re.compile(r"(https?://|www\.)", re.I)
EMAIL_RE = re.compile(r".+@.+\..+")
PUNCT_SET= {".",",","!","?",";",":","(",")","[","]","{","}","–","—","-","—","|","/","\\","'","\"","“","”","‘","’","।"}
COMMON_SRC_STOPS = set([
    "के","की","का","और","कि","तो","में","पर","से","था","थे","है","हूँ","हैं","हो","ही","ने",
    "the","a","an","and","or","to","of","in","on","for","is","are","was","were"
])

defis_copy_token(w: str) -> bool:
    w_=w.strip()
    return (w_ in PUNCT_SET or NUM_RE.match(w_) is not None or
            URL_RE.search(w_) is not None or EMAIL_RE.match(w_) is not None)

defbuild_dice_tables(pairs, min_count=2):
    src_docs_contain, tgt_docs_contain, cooccur = defaultdict(set), defaultdict(set), defaultdict(int)
    for i,(s,t) in enumerate(pairs):
        s_set=set(simple_tokenize(s.lower())); t_set=set(simple_tokenize(t.lower()))
        for sw in s_set: src_docs_contain[sw].add(i)
        for tw in t_set: tgt_docs_contain[tw].add(i)
        for sw in s_set:
            for tw in t_set:
                cooccur[(sw,tw)] += 1
    src_count={w:len(idx) for w,idx in src_docs_contain.items()}
    tgt_count={w:len(idx) for w,idx in tgt_docs_contain.items()}
    dice_by_src=defaultdict(list)
    for (sw,tw),cst in cooccur.items():
        sc,tc=src_count.get(sw,0),tgt_count.get(tw,0)
        if sc>=min_count and tc>=min_count:
            dice=2.0*cst/(sc+tc)
            dice_by_src[sw].append((dice,tw))
    for sw in dice_by_src: dice_by_src[sw].sort(reverse=True)
    return dice_by_src

definit_from_dice(dice_by_src, top_k=40):
    t_given_s={}
    for sw,lst in dice_by_src.items():
        cands=[tw for _,tw in lst[:top_k]]
        if not cands: continue
        p=1.0/len(cands)
        t_given_s[sw]={tw:p for tw in cands}
    return t_given_s

defibm1_em_with_diag(pairs, t_given_s, iters=6, floor=1e-9, lambda_diag=4.0):
    # fast_align-style diagonal prior: exp(-λ * | j/l - i/m |)
    for _ in range(iters):
        count=defaultdict(lambda: defaultdict(float))
        total_s=defaultdict(float)
        for s,t in pairs:
            s_tok=[w for w in simple_tokenize(s.lower())]
            t_tok=[w for w in simple_tokenize(t.lower())]
            s_ext=["<NULL>"]+s_tok
            m,l=len(s_ext),len(t_tok)
            for j,tw in enumerate(t_tok, start=1):
                z=0.0; weights=[]
                for i,sw in enumerate(s_ext, start=1):
                    base=t_given_s.get(sw,{}).get(tw,floor)
                    pos=math.exp(-lambda_diag*abs((j/max(1,l))-(i/max(1,m))))
                    wght=base*pos
                    weights.append((sw,wght)); z+=wght
                if z==0.0: z=floor*m
                for sw,wght in weights:
                    p=wght/z
                    count[sw][tw]+=p; total_s[sw]+=p
        new_t={}
        for sw,d in count.items():
            denom=total_s[sw] if total_s[sw]>0 else 1.0
            dd={tw:max(c/denom, floor) for tw,c in d.items()}
            new_t[sw]=dict(sorted(dd.items(), key=lambda x:x[1], reverse=True)[:100])
        t_given_s=new_t
    return t_given_s

defsymmetrize(t_ab, t_ba, thresh=1e-7):
    lex={}
    for sw in set(t_ab.keys()):
        cand=set(t_ab.get(sw,{}).keys())
        bi={tw for tw in cand if t_ba.get(tw,{}).get(sw,0.0)>thresh}
        if bi:
            lex[sw]={tw:0.5*(t_ab[sw].get(tw,0)+t_ba.get(tw,{}).get(sw,0)) for tw in bi}
        else:
            tops=dict(sorted(t_ab.get(sw,{}).items(), key=lambda x:x[1], reverse=True)[:5])
            if tops: lex[sw]=tops
    return lex

defbuild_lexicon_ibm1_diag(pairs, dice_top=40, em_iter=6, min_count=2, lambda_diag=4.0):
    dice_src=build_dice_tables(pairs, min_count=min_count)
    t_ab=init_from_dice(dice_src, top_k=dice_top)
    rev_pairs=[(t,s) for (s,t) in pairs]
    dice_tgt=build_dice_tables(rev_pairs, min_count=min_count)
    t_ba=init_from_dice(dice_tgt, top_k=dice_top)
    t_ab=ibm1_em_with_diag(pairs,     t_ab, iters=em_iter, lambda_diag=lambda_diag)
    t_ba=ibm1_em_with_diag(rev_pairs, t_ba, iters=em_iter, lambda_diag=lambda_diag)
    lex_prob=symmetrize(t_ab, t_ba)
    # nhẹ hoá stopwords
    for sw in list(lex_prob.keys()):
        if sw in COMMON_SRC_STOPS:
            top3=dict(sorted(lex_prob[sw].items(), key=lambda x:x[1], reverse=True)[:3])
            s=sum(top3.values()) or 1.0
            lex_prob[sw]={tw:(p/s)*0.5 for tw,p in top3.items()}
    return lex_prob

# -------------------- KN-interpolated trigram LM --------------------
D_DISCOUNT = 0.75

classKNTrigramLM:
    def__init__(self):
        self.c2 = defaultdict(Counter)     # bigram: w_{i-1} -> w_i
        self.c3 = defaultdict(Counter)     # trigram: (w_{i-2}, w_{i-1}) -> w_i
        self.c1 = Counter()
        self.total_tokens = 0
        self.continuation = Counter()      # #unique left histories for w_i
        self.bigram_continuation = Counter()

    deftrain(self, sentences):
        seen_uni_hist=set(); seen_bi_hist=set()
        for s in sentences:
            toks=["<s>","<s>"]+simple_tokenize(s.lower())+["</s>"]
            for i in range(2, len(toks)):
                w2,w1,wi = toks[i-2], toks[i-1], toks[i]
                self.c3[(w2,w1)][wi]+=1
                self.c2[(w1)][wi]+=1
                self.c1[wi]+=1
                self.total_tokens+=1
                seen_uni_hist.add((wi, w1))
                seen_bi_hist.add(((w1,wi), w2))
        for (wi,_h) in seen_uni_hist:
            self.continuation[wi]+=1
        for ((w1,wi),_h2) in seen_bi_hist:
            self.bigram_continuation[(w1,wi)]+=1

    defprob_kn2(self, wi, w1):
        c_hist = sum(self.c2[(w1)].values())
        c = self.c2[(w1)][wi]
        left = max(c - D_DISCOUNT, 0.0) / max(1, c_hist)
        n_types = len(self.c2[(w1)])
        lam = (D_DISCOUNT * n_types) / max(1, c_hist)
        cont = self.continuation[wi] / max(1, len(self.c2))
        return left + lam * cont

    defprob_kn3(self, wi, w2, w1):
        hist = (w2,w1)
        c_hist = sum(self.c3[hist].values())
        c = self.c3[hist][wi]
        left = max(c - D_DISCOUNT, 0.0) / max(1, c_hist)
        n_types = len(self.c3[hist])
        lam = (D_DISCOUNT * n_types) / max(1, c_hist)
        return left + lam * self.prob_kn2(wi, w1)

    defscore(self, wi, w2, w1):
        p = self.prob_kn3(wi.lower(), w2.lower(), w1.lower())
        return math.log(max(p, 1e-12))

deftrain_word_trigram_kn(sentences):
    lm = KNTrigramLM()
    lm.train(sentences)
    return lm.score

# -------------------- Char 5-gram LM --------------------
deftrain_char_ngram(sentences, n=5, add_k=0.05):
    fromcollectionsimport defaultdict, Counter
    ngram = defaultdict(Counter); ctx_count = Counter()
    for s in sentences:
        chars = ["<s>"]*(n-1) + list(s.lower()) + ["</s>"]
        for i in range(n-1, len(chars)):
            ctx = tuple(chars[i-n+1:i])
            ngram[ctx][chars[i]] += 1
            ctx_count[ctx] += 1
    defscore_word(word):
        chars = ["<s>"]*(n-1) + list(word.lower())
        sc = 0.0
        for i in range(n-1, len(chars)):
            ctx = tuple(chars[i-n+1:i])
            c = ngram[ctx].get(chars[i], 0)
            total = ctx_count[ctx]
            sc += math.log((c + add_k) / (total + add_k*1e5))
        return sc
    return score_word

# -------------------- Length ratio --------------------
defestimate_len_ratio(pairs):
    ratios=[]
    for s,t in pairs:
        ls=max(1,len(simple_tokenize(s))); lt=max(1,len(simple_tokenize(t)))
        ratios.append(lt/ls)
    ratios.sort()
    return ratios[len(ratios)//2] if ratios else 1.0

# -------------------- Beam decoder (char-LM fusion + length bonus) --------------------
defbeam_decode(src_tokens, lex_prob, word_lm_fn, char_lm_fn,
                beam_size=6, lm_w=0.90, char_lm_w=0.20, rep_pen_w=0.80,
                len_ratio=1.0, len_w=0.12):
    Beam=[([], "<s>", "<s>", 0.0, 0)]  # (hyp, prev2, prev1, score, len)
    for w in src_tokens:
        newB=[]
        if is_copy_token(w):
            cand=[(w, 0.0)]
        else:
            dist = lex_prob.get(w, None)
            if not dist:
                cand=[(w, math.log(1e-6))]
            else:
                lst = sorted(dist.items(), key=lambda x:x[1], reverse=True)[:5]
                cand = [(tw, math.log(max(1e-12,p))) for (tw,p) in lst]
        for hyp,p2,p1,score,L in Beam:
            for tw,lp in cand:
                wlm = word_lm_fn(tw.lower(), p2.lower(), p1.lower()) if word_lm_fn else 0.0
                clm = char_lm_fn(tw) if char_lm_fn else 0.0
                rep = 0.0
                if len(hyp)>=2 and hyp[-1]==tw and hyp[-2]==tw: rep -= 3.0*rep_pen_w
                elif len(hyp)>=1 and hyp[-1]==tw:               rep -= 1.0*rep_pen_w
                tgt_len=L+1; src_len=len(src_tokens); goal=len_ratio*src_len
                len_pen = -len_w * abs(tgt_len - goal)/max(1.0, goal)
                newB.append((hyp+[tw], p1, tw, score + lp + lm_w*wlm + char_lm_w*clm + rep + len_pen, tgt_len))
        newB.sort(key=lambda x:x[3], reverse=True)
        Beam=newB[:beam_size]
    # final length bonus nhẹ
    deffinal_score(entry):
        _,_,_,sc,L = entry
        goal=len_ratio*len(src_tokens)
        return sc + len_w * math.log(max(1.0, L)/max(1.0, goal))
    best = max(Beam, key=final_score)[0]
    return best

# -------------------- Train per-direction models --------------------
lex_prob_by_dir = {}
len_ratio_by_dir = {}
word_lm_by_tgt  = {}
char_lm_by_tgt  = {}

for (s_lang,t_lang), pairs in bitext_by_dir.items():
    if not pairs: continue
    is_forward = (s_lang,t_lang) in FORWARD_DIRS
    em_iter  = 7 if is_forward else 5
    dice_top = 40 if is_forward else 30
    print(f"[LEX-DIAG] {s_lang}->{t_lang} | EM={em_iter} dice_top={dice_top} λ={LAMBDA_DIAG} pairs={len(pairs):,}")
    lex_prob_by_dir[(s_lang,t_lang)] = build_lexicon_ibm1_diag(
        pairs, dice_top=dice_top, em_iter=em_iter, min_count=2, lambda_diag=LAMBDA_DIAG
    )
    if t_lang not in word_lm_by_tgt:
        tgt_sents=[t for _,t in pairs]
        word_lm_by_tgt[t_lang] = train_word_trigram_kn(tgt_sents)   # KN 3-gram
        char_lm_by_tgt[t_lang] = train_char_ngram(tgt_sents, n=5, add_k=0.05)
    len_ratio_by_dir[(s_lang,t_lang)] = estimate_len_ratio(pairs)
    gc.collect()

# -------------------- Approx metrics for dev tuning (fast) --------------------
defbleu_corpus_approx(refs, hyps):
    # BLEU-2 + BP (proxy)
    defngram_counts(toks, n):
        return Counter([tuple(toks[i:i+n]) for i in range(0, max(0,len(toks)-n+1))])
    inter1=inter2=tot1=tot2=0; len_h=len_r=0
    for r,h in zip(refs, hyps):
        rt=r.split(); ht=h.split()
        R1,H1 = Counter(rt), Counter(ht)
        inter1 += sum((R1 & H1).values()); tot1 += max(1,len(ht))
        R2,H2 = ngram_counts(rt,2), ngram_counts(ht,2)
        inter2 += sum((R2 & H2).values()); tot2 += max(1,len(ht)-1)
        len_h += len(ht); len_r += len(rt)
    p1 = inter1/max(1,tot1); p2 = inter2/max(1,tot2)
    logp = 0.5*(math.log(max(p1,1e-9)) + math.log(max(p2,1e-9)))
    bp = math.exp(min(0.0, 1 - (len_r/max(1,len_h))))
    return bp * math.exp(logp)

defchrf3_corpus_approx(refs, hyps):
    defc3(s):
        t=list(s); 
        return Counter([tuple(t[i:i+3]) for i in range(0, max(0,len(t)-2))])
    inter=totH=totR=0
    for r,h in zip(refs, hyps):
        R,H = c3(r), c3(h)
        inter += sum((R & H).values()); totH += sum(H.values()); totR += sum(R.values())
        if totH==0 or totR==0: return 0.0
    p = inter/max(1,totH); r = inter/max(1,totR)
    return 0.0 if (p+r)==0 else 2*p*r/(p+r)

defobjective_score(blocks):
    # blocks: {'f_bleu','r_bleu','f_chrf','r_chrf'}
    return 0.6*(blocks['f_bleu']) + 0.4*(blocks['r_bleu']) + 0.4*(0.6*blocks['f_chrf'] + 0.4*blocks['r_chrf'])

defdev_split(pairs, max_n=2000, seed=1337):
    random.Random(seed).shuffle(pairs)
    n = min(max_n, len(pairs))
    cut = n//5 if n>=100 else max(1, n//5)   # ~20% dev
    return pairs[cut:n], pairs[:cut]

# -------------------- Decode helpers --------------------
BASE_PARAMS_FWD = dict(beam=6, lm_w=0.90, char_lm_w=0.20, rep_pen=0.80, len_w=0.12)
BASE_PARAMS_REV = dict(beam=4, lm_w=0.70, char_lm_w=0.15, rep_pen=0.60, len_w=0.10)

deftranslate_tokens(src_tokens, key, word_lm_fn, char_lm_fn, params):
    lex = lex_prob_by_dir.get(key, {})
    if not lex: return src_tokens
    hyp_tokens = beam_decode(
        [w.lower() for w in src_tokens], lex, word_lm_fn, char_lm_fn,
        beam_size=params['beam'], lm_w=params['lm_w'], char_lm_w=params['char_lm_w'],
        rep_pen_w=params['rep_pen'], len_ratio=len_ratio_by_dir.get(key,1.0), len_w=params['len_w']
    )
    return hyp_tokens

deftranslate_sentence(s, src_lang, tgt_lang, params_fwd=BASE_PARAMS_FWD, params_rev=BASE_PARAMS_REV):
    src_lang = canon_label(src_lang); tgt_lang = canon_label(tgt_lang)
    toks = simple_tokenize(s)
    key = (src_lang, tgt_lang)
    if key not in lex_prob_by_dir:
        return detokenize(toks) if toks else "."
    params = params_fwd if key in FORWARD_DIRS else params_rev
    hyp_tokens = translate_tokens(
        toks, key, word_lm_by_tgt.get(tgt_lang), char_lm_by_tgt.get(tgt_lang), params
    )
    if len(hyp_tokens)>0 and toks and toks[0][:1].isupper():
        hyp_tokens[0] = hyp_tokens[0][:1].upper() + hyp_tokens[0][1:]
    hyp = detokenize(hyp_tokens)
    return hyp if hyp.strip() else "."

# -------------------- DEV tuner --------------------
defrun_dev_tune():
    dev_sets = {'f':[], 'r':[]}
    for key, pairs in bitext_by_dir.items():
        if not pairs: continue
        is_fwd = key in FORWARD_DIRS
        tr, dv = dev_split(pairs[:], max_n=DEV_MAX_PER_DIR)
        dev_sets['f' if is_fwd else 'r'].append((key, dv))

    best=None; best_score=-1
    for char_w in GRID_CHAR_W:
        for len_w in GRID_LEN_W:
            params_f = BASE_PARAMS_FWD.copy(); params_f['char_lm_w']=char_w; params_f['len_w']=len_w
            params_r = BASE_PARAMS_REV.copy(); params_r['char_lm_w']=char_w; params_r['len_w']=len_w

            f_refs,f_hyps,r_refs,r_hyps = [],[],[],[]
            for key, dv in dev_sets['f']:
                for s,t in dv:
                    f_hyps.append(translate_sentence(s, key[0], key[1], params_f, params_r)); f_refs.append(t)
            for key, dv in dev_sets['r']:
                for s,t in dv:
                    r_hyps.append(translate_sentence(s, key[0], key[1], params_f, params_r)); r_refs.append(t)

            blocks = dict(
                f_bleu=bleu_corpus_approx(f_refs,f_hyps) if f_refs else 0.0,
                r_bleu=bleu_corpus_approx(r_refs,r_hyps) if r_refs else 0.0,
                f_chrf=chrf3_corpus_approx(f_refs,f_hyps) if f_refs else 0.0,
                r_chrf=chrf3_corpus_approx(r_refs,r_hyps) if r_refs else 0.0
            )
            obj = objective_score(blocks)
            print(f"[TUNE] char_w={char_w:.2f} len_w={len_w:.2f} -> score={obj:.4f} | blocks={blocks}")
            if obj>best_score:
                best_score=obj; best=(char_w,len_w)
    if best is None:
        print("[TUNE] Fallback to base params.")
        return BASE_PARAMS_FWD['char_lm_w'], BASE_PARAMS_FWD['len_w']
    print(f"[TUNE] Best: char_lm_w={best[0]:.2f}, len_w={best[1]:.2f} | score={best_score:.4f}")
    return best

# -------------------- Inference --------------------
defrun_inference(char_w=None, len_w=None):
    if char_w is not None and len_w is not None:
        BASE_PARAMS_FWD['char_lm_w']=char_w; BASE_PARAMS_REV['char_lm_w']=char_w
        BASE_PARAMS_FWD['len_w']=len_w;     BASE_PARAMS_REV['len_w']=len_w

    test_path = os.path.join(COMP_DIR, "test.csv")
    if not os.path.exists(test_path):
        cand = [p for p in glob(f"{COMP_DIR}/*.csv") if os.path.basename(p).lower().startswith("test")]
        if cand: test_path = cand[0]
    test_df = pd.read_csv(test_path)

    rename_map={}
    for c in test_df.columns:
        cn=c.strip().lower().replace("_"," ")
        if cn=="row id": rename_map[c]="Row ID"
        elif cn=="source lang": rename_map[c]="Source Lang"
        elif cn=="source sentence": rename_map[c]="Source Sentence"
        elif cn=="target lang": rename_map[c]="Target Lang"
        elif cn=="target sentence": rename_map[c]="Target Sentence"
    test_df = test_df.rename(columns=rename_map)

    required_cols = ["Row ID","Source Lang","Source Sentence","Target Lang","Target Sentence"]
    for col in required_cols:
        if col not in test_df.columns:
            if col=="Target Sentence": test_df[col]=""
            else: raise ValueError(f"Missing required column in test.csv: {col}")

    preds=[]
    for _,row in test_df.iterrows():
        hyp=translate_sentence(str(row["Source Sentence"]), str(row["Source Lang"]), str(row["Target Lang"]))
        preds.append(hyp if hyp.strip() else ".")
    test_df["Target Sentence"]=preds
    test_df["Target Sentence"]=test_df["Target Sentence"].fillna(".").apply(lambda s: s if str(s).strip() else ".")
    submission = test_df[required_cols].copy()
    out_path=f"{WORK_DIR}/submission.csv"
    submission.to_csv(out_path, index=False)
    print("Wrote:", out_path)
    try:
        display(submission.head(10))
    except Exception:
        print(submission.head(10).to_string(index=False))

# -------------------- Main --------------------
if DEV_TUNE:
    c,l = run_dev_tune()
    run_inference(c,l)
else:
    run_inference()
```

```
Using competition directory: /kaggle/input/mm-lo-so-2025
Train files found: ['bhili-train.csv', 'gondi-train.csv', 'mundari-train.csv', 'santali-train.csv']
Loaded 20,000 pairs: Hindi -> Bhilli
Loaded 20,000 pairs: Hindi -> Gondi
Loaded 20,000 pairs: Hindi -> Mundari
Loaded 20,000 pairs: English -> Santali
[LEX-DIAG] Hindi->Bhilli | EM=7 dice_top=40 λ=4.0 pairs=20,000
[LEX-DIAG] Bhilli->Hindi | EM=5 dice_top=30 λ=4.0 pairs=20,000
[LEX-DIAG] Hindi->Gondi | EM=7 dice_top=40 λ=4.0 pairs=20,000
[LEX-DIAG] Gondi->Hindi | EM=5 dice_top=30 λ=4.0 pairs=20,000
[LEX-DIAG] Hindi->Mundari | EM=7 dice_top=40 λ=4.0 pairs=20,000
[LEX-DIAG] Mundari->Hindi | EM=5 dice_top=30 λ=4.0 pairs=20,000
[LEX-DIAG] English->Santali | EM=7 dice_top=40 λ=4.0 pairs=20,000
[LEX-DIAG] Santali->English | EM=5 dice_top=30 λ=4.0 pairs=20,000
Wrote: /kaggle/working/submission.csv
```

|   | Row ID | Source Lang | Source Sentence                                                                           | Target Lang | Target Sentence                                                                            |
| - | ------ | ----------- | ----------------------------------------------------------------------------------------- | ----------- | ------------------------------------------------------------------------------------------ |
| 0 | 54334  | Hindi       | उन्होंने कहा कि 2014 के बाद, इस परियोजना को प्...         | Bhilli      | तिनायी केदू कि 2014 ना बाद, इनी परियोजना ने प्...          |
| 1 | 87641  | Hindi       | वित्तीय कठिनाइयों को हल करने में सहायक होने के...   | Bhilli      | वित्तीय थिया ने हल करवा मा सहायक होवा न हाते -...       |
| 2 | 32543  | Hindi       | मेरा सुझाव है कि हमारे सक्रिय दृष्टिकोण, नीतिय...   | Bhilli      | मारो सुझाव से कि हमारा सक्रिय कोण, नीतियों अन ...      |
| 3 | 26313  | Hindi       | श्री मोदी ने कहा यह अटल जी ही थे जिन्होंने देश...     | Bhilli      | श्री मोदी ये केदू यो अटल जी ही थे जिनायी देह न...       |
| 4 | 83303  | Hindi       | उत्सवादि मनाने के उपलक्ष्य में सुरापान करना सा...  | Bhilli      | उत्सवादि मनावा ना उपलक्ष्य मा सुरापान करना साध...   |
| 5 | 131411 | Hindi       | तुम्‍हारे साथ कभी ऐसा हुआ।                                          | Bhilli      | तारा साथ कदी ऐवु थायो।                                                   |
| 6 | 101809 | Hindi       | यह सत्र ग्लासगो, यूनाइटेड किंगडम में आयोजित हुआ।  | Bhilli      | यो सत्र ग्लासगो, यूनाइटेड किंगडम मा आयोजित थायों। |
| 7 | 59328  | Hindi       | यह 9 मार्च 2012 को रिलीज़ हुई थी, जिसे आम तौर ...            | Bhilli      | यो 9 मार्च 2012 ने रिलीज थाई हती, जिने आम तोर ...             |
| 8 | 57205  | Hindi       | नियुक्ति मामलों की मंत्रिमंडलीय समिति ने भारती... | Bhilli      | नियुक्ति मामला नी मंत्रिमंडलीय समिति ये भारत व...   |
| 9 | 103641 | Hindi       | भारत को और अधिक साफ-सुथरा बनाने और बेहतर स्वच्...    | Bhilli      | भारत ने अने वदारे साफ - सफ बणावा अन असल सफाई स...        |
