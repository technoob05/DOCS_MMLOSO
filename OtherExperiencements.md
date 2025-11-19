**# Tính năng mới:

# • N-best beam -> Minimum Bayes-Risk (MBR) reranking với gain = 0.6*BLEU_sent + 0.4*chrF3_sent

# • BLEU_sent dùng smoothing (Lin & Och style add-1) để tránh 0 ở câu ngắn

# Tài liệu: MBR trong SMT (Kumar & Byrne, 2004; Ehling 2007); chrF (Popović, 2015)                  [citations at end]

Điểm : 188

MMLoSo 2025 – Offline Lexical+IBM1+LM

Điểm : 182

# ========================= MMLoSo 2025 – IBM1 (Diagonal prior) + LM + Beam =========================

# Điểm khác biệt: IBM1 EM có diagonal alignment prior (fast_align-style):

# p(a_j=i | ...) ∝ t(f_j|e_i) * exp(-lambda_diag * | j/l - i/m |)

# Nguồn ý tưởng: fast_align (Dyer+Chahuneau+Smith, 2013); IBM1/EM: Brown 1993; Collins notes.

# -----------------------------------------------------------------------------------------------

Điểm : 187.32414

# ========================= MMLoSo 2025 – IBM1 + KN word LM + char-LM + length bonus =========================

# Focus: tăng điểm theo công thức (0.6 BLEU forward + 0.4 BLEU reverse) + 0.4*(0.6 chrF forward + 0.4 chrF reverse)

# Nâng cấp mới:

# • Char 5-gram LM (nhẹ) + fusion vào beam

# • Length bonus để giảm brevity penalty (BLEU)

# • (Tùy chọn) DEV tuner grid-search cho char_lm_w và len_w

# ============================================================================================================

Điểm : 154.23435

# ================= MMLoSo 2025 – IBM1(+diag) + KN-LM + IndicNormalize + MBR-Union =================

# Thêm mới:

# • Indic normalizer (pre/post) cho Devanagari/Santali punctuation

# • Word trigram LM = Kneser–Ney interpolation (KN-3) thay cho add-k

# • N-best từ 2 cấu hình beam (union) -> MBR rerank (0.6*BLEU_sent + 0.4*chrF3_sent)

# Tài liệu: Indic normalization (Indic NLP Lib); KN smoothing (Chen & Goodman 1998);

# MBR decoding (Kumar & Byrne 2004); fast_align diagonal prior (Dyer et al. 2013)

# ===================================================================================================

Điểm : 189.63113

# ========================= MMLoSo 2025 – IBM1(+Diagonal Prior) + KN word LM + char-LM + Length Bonus =========================

# Mục tiêu: Kết hợp bản điểm cao (IBM1 + diagonal prior) với char-LM fusion + length bonus + KN 3-gram LM

# Tùy chọn: DEV tuner (grid) cho char_lm_w ∈ [0.1,0.4], len_w ∈ [0.05,0.25]

# -----------------------------------------------------------------------------------------------------------------------------

Public Score :  175.82615

# ========================= MMLoSo 2025 – IBM1 (Diagonal prior) + LM + Beam =========================

# === CẢI TIẾN: Thêm Unicode Normalization (NFKC) và Data Augmentation (Back-Translation) ===

# 1. Preprocessing:

# - Thêm `unicodedata.normalize('NFKC', ...)` vào `normalize_space` để xử lý các ngôn ngữ Ấn Độ.

# - Thêm tách số (number splitting) trong `simple_tokenize`.

# 2. Augmentation:

# - Thay đổi quy trình huấn luyện để thực hiện Back-Translation.

# - Vòng 1: Huấn luyện các mô hình ngược (ví dụ: Bhilli -> Hindi).

# - Vòng 2: Dùng mô hình ngược để tạo dữ liệu "tổng hợp" (ví dụ: (Synthetic_Hindi, Real_Bhilli)).

# - Vòng 3: Huấn luyện mô hình xuôi (ví dụ: Hindi -> Bhilli) trên (Dữ_liệu_thật + Dữ_liệu_tổng_hợp).

# -----------------------------------------------------------------------------------------------

Public Score : 189.62325

# ========================= MMLoSo 2025 – IBM1(+Diag) + LM + Beam — OPTUNA TUNER =========================

# Hyper-tune bằng Optuna:

# • lambda_diag (độ mạnh diagonal prior kiểu fast_align)

# • beam, lm_w, rep_pen, len_w (riêng forward/reverse)

# • (nhẹ) em_iter_dev & dice_top_dev khi tune trên dev subset (để nhanh)

# Objective (để MAXIMIZE) = 0.6*(BLEU_fwd) + 0.4*(BLEU_rev) + 0.4*(0.6*chrF_fwd + 0.4*chrF_rev)

# -> Proxy BLEU câu (BLEU-2 + BP) & chrF3 (nhanh, ổn định) cho dev.

# Sau khi tìm best params: retrain lexicon FULL bằng best lambda_diag + em_iter_full, decode test với best decoder params.

# ========================================================================================================

Public Score  189.42793

# ========================= MMLoSo 2025 – (Diagonal IBM1 + Back-Translation) + MBR Reranking =========================

# Chiến lược:

# 1. NỀN (Base): Dùng mô hình mạnh nhất từ trước:

# - Preprocessing: NFKC Unicode Normalization + Tách số.

# - Augmentation: Back-Translation (Huấn luyện 3 vòng).

# - Model: IBM1 (Diagonal prior) + Word 3-gram LM.

# 2. CẢI TIẾN (Reranking):

# - Decoder: Sửa đổi beam search để tạo ra N-best list (N=10).

# - Reranker: Thêm Minimum Bayes-Risk (MBR) reranking.

# - Utility Function: Dùng hàm gain của bạn: 0.6*BLEU_sent + 0.4*chrF3_sent.

# ------------------------------------------------------------------------------------------------------------------

Public : 193.25794

# ========================= MMLoSo 2025 – NLLB LoRA MAX-BLEU Edition (FIXED) =========================

 Public :  302.17705

**
