# **4\. Proposed Methodology: A Hybrid Retrieval-Augmented Ensemble**

To address the challenges of data sparsity and structural divergence identified in Section 3, we propose a hybrid translation pipeline that integrates Non-Parametric Retrieval, Statistical Machine Translation (SMT), and Neural Machine Translation (NMT) under a Minimum Bayes-Risk (MBR) decision framework.

## **4.1. Retrieval-Augmented Generation (RAG)**

Government and administrative texts often exhibit high redundancy. To exploit this, we implement a hierarchical retrieval module:

1. **Exact Match:** If a test source sentence appears verbatim in the training corpus, we retrieve its gold standard translation.  
2. Fuzzy Match: We employ a conservative fuzzy matching algorithm based on normalized tokens. If a test sentence matches a training sample with a length difference $|\\Delta L| \\le 1$, we retrieve the corresponding target.  
   This non-parametric approach serves as a strong baseline, preventing generation errors on common domain-specific phrases.

## **4.2. The Hybrid Generator: SMT \+ NLLB**

For unseen sentences, we employ an ensemble of two distinct paradigms to maximize coverage and fidelity:

### **Statistical Component (SMT)**

We trained an IBM Model 1-based SMT system with a diagonal alignment prior. SMT acts as a regularization mechanism, providing "literal" translations that are robust against the hallucinations often plaguing NMT in low-resource settings. We generate an $N$-best list ($N=5$) from the SMT decoder.

### **Neural Component (NLLB-LoRA)**

We fine-tuned the **NLLB-200 (600M)** model using **Low-Rank Adaptation (LoRA)**. LoRA allows us to adapt the massive multilingual parameters to the specific tribal languages without catastrophic forgetting. We generate an $N$-best list ($N=10$) using Beam Search with a length penalty $\\alpha=1.2$.

## **4.3. Minimum Bayes-Risk (MBR) Reranking**

Standard Beam Search maximizes the model's likelihood $P(y|x)$, which often favors high-frequency generic phrases. To select the highest quality translation from our candidate pool $\\mathcal{H} \= \\mathcal{H}\_{SMT} \\cup \\mathcal{H}\_{NLLB}$, we apply Minimum Bayes-Risk (MBR) decoding.

MBR selects the hypothesis $\\hat{y}$ that minimizes the expected loss (or maximizes the expected utility) against all other hypotheses in the pool:

$$\\hat{y}\_{MBR} \= \\operatorname\*{argmax}\_{y \\in \\mathcal{H}} \\sum\_{y' \\in \\mathcal{H}} \\text{Utility}(y, y')$$  
We define the utility function as a weighted combination of BLEU and chrF, aligning with the competition metric:

$$\\text{Utility}(y, y') \= 0.6 \\times \\text{BLEU}(y, y') \+ 0.4 \\times \\text{chrF}(y, y')$$  
By using MBR, our system seeks a "consensus" translation. Since SMT and NLLB have different error modes (SMT makes grammatical errors; NMT makes semantic errors), the consensus translation is likely to be the most correct one, effectively filtering out hallucinations.