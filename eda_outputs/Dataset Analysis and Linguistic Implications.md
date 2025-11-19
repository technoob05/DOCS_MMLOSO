## **3\. Dataset Analysis and Linguistic Implications**

We conducted a comprehensive analysis of the MMLoSo 2025 dataset to understand the linguistic barriers inherent in the translation tasks. Our findings, summarized in Table 1 and visualized in Figure 2, reveal three critical challenges that guided our modeling choices.

### **3.1. Syntactic Isomorphism vs. Divergence**

As shown in the Length Correlation Hexbin plot (Figure 2a), the **Hindi-Bhili** and **Hindi-Gondi** pairs exhibit a strong linear correlation ($r \> 0.9$) with a length ratio close to 1.0. This indicates a high degree of **syntactic isomorphism**, suggesting that these tribal languages share significant structural properties with Hindi. This observation explains the competitive performance of statistical baselines (like IBM1) reported in our preliminary experiments, as word-to-word alignment is relatively straightforward.

Conversely, the **English-Santali** pair (Figure 2b) demonstrates significant **structural divergence**, with Santali sentences being on average 18% longer than their English counterparts (Length Ratio $\\approx 1.18$). This expansion is attributed to the agglutinative nature of Santali compared to the analytical structure of English. Consequently, we adjusted the **length penalty** parameter in our beam search decoding to $\\alpha \> 1.0$ specifically for this pair to mitigate under-generation issues.

### **3.2. Morphological Richness and Data Sparsity**

Table 1 highlights a stark contrast in Type-Token Ratio (TTR). Most notably, **Mundari** exhibits a TTR of **0.222**, more than double that of the source Hindi (0.107). This high TTR is a hallmark of **morphologically rich languages**, implying that a single semantic concept can surface in many distinct word forms.

This phenomenon leads to the **data sparsity** problem, where a significant portion of the test vocabulary is unseen during training. To address this, our methodology incorporates:

1. **Subword Tokenization (SentencePiece):** To break down complex agglutinated words into shared subword units.  
2. **Iterative Back-Translation:** As detailed in Section 4, we utilized synthetic data generation to artificially boost the frequency of rare morphological variants, directly countering the high sparsity observed in the Mundari and Gondi subsets.

### **3.3. Naturalness Validation**

Finally, the Zipf’s law analysis (Figure 3\) confirms that all low-resource target languages follow a power-law distribution ($f \\propto 1/r^\\alpha$). The linearity observed in the log-log plot validates the naturalness of the collected corpora, ensuring that the dataset is free from significant crawling artifacts or synthetic noise.