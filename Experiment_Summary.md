# Experiment Summary & Methodology Evolution for MMLoSo 2025

## 1. Introduction
This document summarizes the experimental journey, data analysis, and methodology evolution for the MMLoSo 2025 Shared Task. The goal was to build a robust translation system for low-resource Indic languages (Bhili, Gondi, Mundari, Santali) to/from Hindi and English.

## 2. Dataset Analysis (EDA)
**Source:** `eda_outputs/Dataset Analysis and Linguistic Implications.md`, `eda_final_stats.csv`

*   **Syntactic Isomorphism**: Hindi-Bhili and Hindi-Gondi show strong length correlation ($r > 0.9$) and length ratio $\approx 1.0$, indicating similar sentence structures. This suggests SMT models (alignment-based) should perform well.
*   **Structural Divergence**: English-Santali has a length ratio of $\approx 1.18$ (Santali is longer), due to Santali's agglutinative nature. This requires length penalty adjustments in decoding.
*   **Morphological Richness & Sparsity**: Mundari has a Type-Token Ratio (TTR) of 0.222 (vs. Hindi's 0.107), indicating extreme morphological richness and data sparsity.
*   **Naturalness**: Zipf's law analysis confirms the corpora are natural and free from significant synthetic noise.

## 3. Methodology Evolution (Experiments)

### Phase 1: Statistical Machine Translation (SMT) Baselines
**Source:** `Experiences/ML/*.md`

We started with SMT to establish a strong baseline, especially given the high isomorphism in some pairs.

| Experiment ID | Method Description | Public Score | Private Score | Key Findings |
| :--- | :--- | :--- | :--- | :--- |
| **ML0** | **Dice Coefficient Baseline**<br>Word-by-word translation based on co-occurrence statistics (Dice). No LM. | 158.84 | 140.32 | Establishes a lower bound. Shows that simple lexical substitution works reasonably well for isomorphic pairs. |
| **ML5** | **Offline Lexical + IBM1 + LM**<br>Basic IBM Model 1 with word-based Language Model. No deep learning. | 182.53 | 148.68 | Strong baseline for isomorphic pairs. |
| **ML1** | **IBM1 (Diagonal Prior) + KN LM + Char LM**<br>Added diagonal alignment prior (fast_align style) to guide EM. Added Kneser-Ney word LM and Char 5-gram LM. | 175.83 | 143.91 | Diagonal prior helps alignment. Char LM adds robustness. |
| **ML2** | **IBM1 + KN LM + Char LM + Length Bonus**<br>Tuned length bonus to fix brevity penalty. Grid search for weights. | 154.23 | 133.46 | Over-tuning on dev set caused regression. |
| **ML3** | **Upgraded SMT Baseline**<br>Memory optimized, pickle-safe implementation. | 96.07 | 87.42 | Implementation fix, but lower score due to simpler config. |
| **Other Exp 1** | **IBM1 + Optuna Tuner**<br>Hyper-parameter tuning for $\lambda_{diag}$, beam width, and LM weights using Optuna. Objective: $0.6 \times BLEU + 0.4 \times chrF$. | 189.43 | - | Automated tuning found better parameters than manual grid search. |
| **Other Exp 2** | **IBM1 + IndicNormalize + MBR-Union**<br>Added Indic Normalization (NFKC) and Union MBR (combining N-best from different beam settings). | 189.63 | - | Normalization is critical for Indic scripts. |
| **Other Exp 3** | **IBM1 (+Diag) + Back-Translation + MBR**<br>Added Back-Translation (3 rounds: Reverse Train $\to$ Synthetic Data $\to$ Forward Train) and MBR reranking. | **193.26** | **153.91** | **Best SMT Result.** Back-translation significantly reduced sparsity. |

### Phase 2: Neural Machine Translation (NLLB + LoRA)
**Source:** `Experiences/MLvsLLM/*.md`

We moved to NMT using NLLB-200 (600M) to handle semantic nuances and non-isomorphic pairs (English-Santali).

| Experiment ID | Method Description | Public Score | Private Score | Key Findings |
| :--- | :--- | :--- | :--- | :--- |
| **LLM0** | **NLLB LoRA + Dice Fallback**<br>Early hybrid: NLLB LoRA for supported pairs, Dice lexicon for others. | 171.64 | 161.10 | Demonstrated that NMT improves over pure Dice, but partial coverage limits performance. |
| **LLM9/10/11** | **NLLB LoRA Continue Training**<br>Continued training from checkpoint-4750. | 168.00 | 155.00 | Initial LoRA attempts showed promise but needed more convergence. |
| **LLM2/6** | **NLLB LoRA (Standard)**<br>Standard fine-tuning. | 302.08 | 166.47 | Huge jump in Public score, but large gap with Private suggests overfitting or domain mismatch. |
| **LLM1/7** | **NLLB LoRA + SMT Ensemble (Simple)**<br>Simple combination. | 306.34 | 174.30 | Combining SMT and NMT improves robustness. |
| **LLM5** | **NLLB LoRA + SMT + MBR**<br>Using MBR to select between NLLB and SMT hypotheses. | **306.56** | **174.53** | **Best Single Model NMT.** MBR effectively filters hallucinations. |

### Phase 3: The Final Hybrid Pipeline (Main Pipeline)
**Source:** `MainPipeLine/*.md`, `post_proccessing.md`

The final submission combines the best of both worlds using a sophisticated post-processing and ensemble strategy.

**Core Components:**
1.  **Retrieval-Augmented Generation (RAG)**:
    *   **Exact Match**: Retrieve gold target if source exists in train.
    *   **Fuzzy Match**: Retrieve target if source matches train with $|\Delta L| \le 1$ (normalized).
    *   *Impact*: Handles administrative redundancy perfectly.

2.  **Hybrid Generator (Ensemble)**:
    *   **SMT**: IBM1 + Diagonal Prior + Back-Translation (provides "literal" candidates).
    *   **NMT**: NLLB-200 (600M) + LoRA (provides "fluent" candidates).

3.  **Minimum Bayes-Risk (MBR) Reranking**:
    *   Pool: Candidates from SMT ($N=5$) and NLLB ($N=10$).
    *   Utility: $0.6 \times BLEU + 0.4 \times chrF$.
    *   *Goal*: Find the "consensus" translation, mitigating NMT hallucinations and SMT grammatical errors.

4.  **Post-Processing**:
    *   **Digit Mapping**: Convert Latin digits (0-9) to Devanagari (०-९) for Indic targets.
    *   **Special Tokens**: Ensure Numbers, URLs, and Emails from source are preserved in target.

**Final Results (Leaderboard):**
*   **Public Score**: 311.61
*   **Private Score**: 186.37 (Rank 2)

## 4. Key Takeaways for Paper
1.  **Hybrid is King**: Pure SMT is robust but rigid; Pure NMT is fluent but hallucinates. The combination via MBR yields the best results.
2.  **Data Augmentation**: Back-translation is crucial for morphologically rich languages like Mundari.
3.  **Domain Adaptation**: RAG is a simple yet highly effective domain adaptation technique for government data.
4.  **Metric-Aware Decoding**: Tuning MBR utility to match the competition metric ($0.6 \times BLEU + 0.4 \times chrF$) directly optimizes the final score.
