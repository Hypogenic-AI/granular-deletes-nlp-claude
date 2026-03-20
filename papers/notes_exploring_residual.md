# Notes: "Neuron-Level Knowledge Attribution in Large Language Models"

**Authors:** Zeping Yu, Sophia Ananiadou
**Affiliation:** Department of Computer Science, National Centre for Text Mining, University of Manchester
**ArXiv ID:** 2312.12141v4 (submitted 2023, revised September 24, 2024)
**Filename stem used:** `yu2023_exploring_residual`

> **Note on title discrepancy:** The paper filename and folder reference "exploring_residual," but the actual paper title is "Neuron-Level Knowledge Attribution in Large Language Models." The arXiv ID 2312.12141 confirms this is the correct document. The paper is closely related to residual stream analysis — it decomposes the residual stream into neuron-level contributions — but its primary framing is neuron attribution for knowledge storage, not a general residual stream exploration study.

---

## 1. Research Question and Key Contribution

### Research Question
How can we identify and localize the specific neurons (in both attention and FFN layers) that are responsible for storing and producing factual knowledge in large language models? More specifically: which neurons directly contribute to the final prediction probability (called "value neurons"), and which neurons activate those neurons (called "query neurons")?

### Key Contributions
1. **A static neuron-level attribution method** based on *log probability increase* that outperforms seven other methods across three evaluation metrics. The method can scale to millions of neurons because it requires only a single forward pass.
2. **A method to identify "query neurons"** — neurons that do not directly encode the final answer but activate the value neurons that do. This is computed via inner products between neuron subkeys and the residual stream components.
3. **An empirical analysis of six types of factual knowledge** (language, capital, country, color, number, month) across both attention and FFN layers in GPT2-large and Llama-7B, yielding a detailed map of where and how knowledge is stored.

---

## 2. Methodology for Exploring the Residual Stream

### Transformer Residual Stream Formulation
The paper treats the transformer's residual stream explicitly. Each layer's output is the additive sum:

```
h_i^l = h_i^{l-1} + A_i^l + F_i^l
```

where `h_i^{l-1}` is the input from the previous layer, `A_i^l` is the attention output, and `F_i^l` is the FFN output. The final hidden state `h_T^L` (last token, last layer) is projected to vocabulary logits via the unembedding matrix `E_u`.

Because the residual stream is a direct sum, the final vector can be decomposed into contributions from all neuron-level vectors across all layers. The paper uses this decomposability as the foundation for attribution.

### FFN Neuron Decomposition
Following Geva et al. (2020), FFN output is a weighted sum of "subvalue" vectors:

```
F_i^l = sum_k [ m_{i,k}^l * fc2_k^l ]
```

where `fc2_k^l` is the k-th column of the second FFN weight matrix (the "subvalue"), and `m_{i,k}^l` is the scalar coefficient computed by the nonlinear activation applied to the dot product between the residual stream and `fc1_k^l` (the "subkey").

### Attention Neuron Decomposition
Similarly, attention output is decomposed into a sum over heads and positions:

```
A_i^l = sum_j sum_p [ alpha_{i,j,p}^l * W_o^{j,l}(W_v^{j,l} * h_p^{l-1}) ]
```

The k-th column of `W_o^{j,l}` is the k-th "attention subvalue," and the k-th row of `W_v^{j,l}` is the k-th "attention subkey."

In total, the final hidden state is a sum of `L × (T × H × d/H + N) + 1` neuron-level vectors.

### Distribution Change Analysis (Section 3.2)
The key theoretical analysis examines how adding a neuron vector `v` to the running sum `x` changes the output probability distribution. The key insight is:

- The change in each token's **before-softmax (bs) value** is linear: `bs(x + v) = bs(x) + bs(v)`
- However, the probability change `p(w|x+v) - p(w|x)` is **nonlinear**
- Both the **coefficient score** `m` and the **ranking of the target token** in the neuron's vocabulary projection matter; neither alone is sufficient
- A neuron's effect is to magnify the token with the largest bs-value — this explains why many FFN neurons exhibit human-interpretable concepts when projected into vocabulary space

### Importance Score for Value Neurons
The proposed importance score is **log probability increase**:

```
Imp(v^l) = log(p(w | v^l + h^{l-1})) - log(p(w | h^{l-1}))
```

This score: (a) accounts for the context `x` in which the neuron fires, (b) is approximately additive across modules (`Imp(x+v) ≈ Imp(x) + Imp(v)`), and (c) attributes neurons in both medium-deep and very deep layers (unlike probability increase, which collapses to only the deepest layers).

### Importance Score for Query Neurons
For each identified value neuron (with subkey `fc1_k^l`), the query importance score of another neuron/subvector within the residual stream is computed as its **inner product** with the value neuron's subkey. Despite the nonlinear activation `sigma`, the relative ordering of inner products reliably predicts which neurons most strongly activate the value neuron.

---

## 3. Key Findings About Residual Stream Structure

### Knowledge Lives in Deep Layers
- All top-10 most important layers for value neurons (both attention and FFN) are in **deep layers**
- In GPT2-large (36 layers): important layers are in the range 18–35
- In Llama-7B (32 layers): important layers are in the range 14–31
- Shallow layers contribute negligibly to direct knowledge output

### Attention vs. FFN Contributions Depend on Knowledge Type
From Table 3 (sum of importance scores), the split between attention (A) and FFN (F) layers varies by knowledge type:
- **Language, capital, country:** Dominated by attention layers (e.g., Llama: L-A = 6.28 vs. L-F = 1.74 for language)
- **Number:** Dominated by FFN layers (e.g., Llama: L-F = 5.60 vs. L-A = 2.22)
- **Color:** More balanced, with slight FFN dominance
- **Month:** Roughly balanced with slight attention dominance

### Semantically Similar Knowledge Clusters in Similar Layers/Heads
- Language, capital, and country (semantically related) tend to be stored in the **same layers and attention heads**
  - GPT2: `a26, a30, a28, a22` rank top for all three
  - Llama: `a23` ranks first for all three
- Semantically dissimilar knowledge (e.g., language vs. color vs. number) resides in distinct layers

### Head-Level Specialization
- The same heads store similar knowledge types: in GPT2, head `a30^6` (layer 30, head 6) ranks in the top 8 for language, capital, and country simultaneously
- Intervening on the top 1% of heads (7 in GPT2, 10 in Llama) causes large drops for the targeted knowledge type (44.5%/53.3% MRR/prob in GPT2) but minimal cross-knowledge interference (only 7.1%/9.5% for semantically unrelated knowledge)

### Neuron-Level Concentration
- Though many neurons contribute to predictions, the **information is highly concentrated**: a small set captures almost all signal
- In Llama-7B:
  - Top 200 attention neurons capture importance scores comparable to *all* positive attention neurons
  - Top 100 FFN neurons similarly capture the bulk of FFN importance
- Intervening on just 300 neurons total (200 attention + 100 FFN) causes MRR/probability decreases of **96.9%/99.6%** in Llama and **96.3%/99.2%** in GPT2
- Random intervention on the same number of neurons causes only 0.22%/0.14% decrease

---

## 4. How Information Flows and Is Modified Across Layers

### The Information Flow Circuit
The paper identifies a two-stage information flow:

**Stage 1 — Feature Extraction (Shallow/Medium FFN layers → Deep Attention layers):**
- Shallow and medium FFN layers (query neurons) extract features from the input and activate the deep attention "value neurons"
- In GPT2: the very shallowest FFN layers (f0, f1, f2) are the top query layers for attention value neurons
- In Llama: medium FFN layers (f14–f24) are the top query layers for attention value neurons

**Stage 2 — Knowledge Activation (Medium-Deep Attention layers → Deep FFN neurons):**
- Medium-deep attention layers activate the deep FFN "value neurons"
- In GPT2: `a19, a22, a26` serve as both value layers (contributing directly to predictions) and query layers (activating deeper FFN neurons)
- In Llama: `a16, a18, a19, a21` play this dual role

**Summary:** `shallow/medium FFN → deep attention → deep FFN → final prediction`

### Query vs. Value Neuron Distribution
- "Query-only" neurons (those that activate value neurons but don't directly encode the answer) are far more numerous than "query-value" neurons (those that do both)
- The proportion of query-only neurons drops sharply after the 60% depth mark in both models, indicating that shallow/medium layers focus on activation rather than direct prediction
- In GPT2, the very shallowest FFN layers play an unusually large role as query layers compared to Llama

### Shared Neurons Across Instances
- Value neurons are more consistent across different sentences of the same knowledge type than query neurons
- In GPT2: 27.6% of top-300 value neurons are shared across >50% of sentences for a given knowledge type
- In GPT2: only 15.7% of query neurons are shared (more dispersed)
- In Llama: even lower sharing — 14.1% value, 5.2% query

---

## 5. Findings About Information Deletion or Suppression

### Negative Importance Scores
The paper notes that many neurons have **negative** importance scores — they actively *suppress* the target token's probability. The distinction between positive and negative contributions is explicit:

- In Llama attention layers: `all` neurons sum to ~6.7 (for language), while `positive`-only neurons sum to ~30.5, indicating substantial cancellation from negative neurons
- In Llama FFN layers: `all` sum ~2.5 (language), `positive`-only sum ~77.4 — an even larger cancellation effect, suggesting that FFN layers contain many suppressive neurons that partially cancel the strong positive ones

### Coefficient Sign Flipping
The theoretical analysis (Section 3.2, Table 1) shows that if a neuron's coefficient score sign is flipped (e.g., from +6 to -6), the effect on the target token's probability reverses from amplification to suppression. This means the coefficient score `m` is a key control signal for whether a neuron promotes or suppresses a token.

### Intervening by Zeroing Neurons
The evaluation methodology directly tests information deletion: setting neuron parameters to zero and measuring the resulting drop in MRR and probability. This is the gold standard used to validate attribution quality — the better the attribution method identifies the "right" neurons, the larger the drop when those neurons are zeroed.

### No Explicit "Deletion Circuit" Analysis
The paper does not specifically frame its findings in terms of information deletion or unlearning circuits. The negative-contribution neurons are noted but not analyzed in depth. The suppressive role of certain neurons is an implicit finding rather than a dedicated study. This is flagged as a direction for future work (e.g., identifying toxicity neurons or bias neurons that could be suppressed).

---

## 6. Datasets and Models Used

### Models
- **GPT2-large** (Radford et al., 2019):
  - 36 transformer layers
  - 20 attention heads per layer
  - 64 neurons per attention head
  - 5,120 FFN neurons per layer
- **Llama-7B** (Touvron et al., 2023):
  - 32 transformer layers
  - 32 attention heads per layer
  - 128 neurons per attention head
  - 11,008 FFN neurons per layer

### Dataset
- **TriviaQA** (Joshi et al., 2017): A large-scale reading comprehension dataset with distantly supervised question-answer pairs
- Filtered to six knowledge types: **language, capital, country, color, number, month**
- Only sentences where the correct token ranks within the top 10 predictions AND higher than other tokens of the same knowledge type are retained
- Final dataset size:
  - 1,350 sentences for GPT2-large
  - 3,141 sentences for Llama-7B

---

## 7. Key Results and Visualizations

### Attribution Method Comparison (Table 2)
Eight attribution methods are compared by their ability to identify the 10 FFN neurons whose removal causes the largest drop in target-token probability/rank:

| Method | GPT2 MRR | GPT2 prob | Llama MRR | Llama prob |
|--------|-----------|-----------|-----------|------------|
| Original | 0.361 | 7.1% | 0.551 | 21.5% |
| **Log prob increase (proposed)** | **0.201** | **3.4%** | **0.312** | **9.2%** |
| Log probability | 0.214 | 3.6% | 0.339 | 10.8% |
| Prob increase | 0.219 | 3.7% | 0.345 | 10.0% |
| Norm | 0.363 | 7.1% | 0.549 | 21.3% |
| Coeff score | 0.439 | 8.6% | 0.529 | 22.9% |
| Rank | 0.306 | 5.8% | 0.493 | 18.1% |
| m × norm | 0.394 | 8.1% | 0.523 | 22.6% |
| m × rank | 0.232 | 4.0% | 0.389 | 13.0% |

Lower post-intervention scores indicate better attribution. The proposed method consistently wins.

### Neuron Distribution Figures
- **Figure 2 / Figure 9:** Histogram of neuron layer distribution in Llama / GPT2. Log probability increase attributes neurons across layers 17–31 (Llama), while probability increase collapses to only layers 23–31.
- **Figure 3 / Figure 10:** Curves of log probability and raw probability as a function of interpolation segment from input to final hidden state. Log probability is approximately linear across the first 2/3 of the path; raw probability surges only at the final segment.

### Layer-Level Importance Heatmaps
- **Figure 4 (GPT2) / Figure 5 (Llama):** Heatmaps showing the top-10 most important "value layers" for each knowledge type. Color intensity indicates importance. Semantic clustering is visually apparent.
- **Figure 6 (GPT2) / Figure 7 (Llama):** Heatmaps for top-10 "query layers" activating FFN value neurons. Medium-depth attention layers dominate.

### Query Neuron Intervention (Table 7)
Intervening on top-1000 query FFN neurons causes MRR/probability drops of:
- GPT2: 91%/96% average across knowledge types
- Llama: 87%/95% average
- Random intervention of same size: only 0.8%/1.1%

### Value Neuron Intervention (Table 13)
Intervening on top-200 attention + top-100 FFN value neurons causes:
- GPT2: ~96%/99% MRR/probability drop
- Llama: ~97%/99% MRR/probability drop

### Neuron Interpretability (Table 9)
Projecting value neurons into vocabulary space reveals interpretable content:
- GPT2 FFN value neuron f29-3771: top tokens include Chile, Nicaragua, Finland, Ireland, Belarus, Norway (all countries)
- Llama attention value neuron a23^12-70: top tokens include German, Greek, Netherlands, Dutch, Germany, Greece (language/country cluster)
- Query neurons are largely uninterpretable in vocabulary space

---

## 8. Connection to Logit Lens and Similar Tools

### Direct Relationship to Logit Lens
The paper's methodology is fundamentally a **neuron-level extension of the logit lens** (Nostalgebraist, 2020). The logit lens involves projecting intermediate hidden states via the unembedding matrix `E_u` to inspect what the model "predicts" at each layer. This paper does the same but at the level of individual neuron vectors rather than the full hidden state.

### "Before-Softmax Value" (bs-value)
The paper introduces the term "bs-value" (before-softmax value): for a vector `x` and token `w`, `bs_w^x = e_w · x` where `e_w` is the w-th row of `E_u`. This is precisely the quantity that the logit lens examines. The paper's insight is that this quantity changes **linearly** when a neuron vector is added, even though the resulting probability changes nonlinearly.

### Direct Logit Attribution (DLA)
The paper explicitly references Wang et al. (2022)'s Direct Logit Attribution and notes that its own "log probability" baseline (method b) is equivalent to DLA. The proposed log probability *increase* method is shown to be strictly better than DLA because it conditions on the running residual state rather than evaluating the neuron in isolation.

### Dar et al. (2022) and Geva et al. (2022)
The approach of projecting FFN subvalue vectors into vocabulary space to interpret neurons (Table 9 in this paper) directly follows the methodology of "Analyzing Transformers in Embedding Space" (Dar et al., 2022) and "Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space" (Geva et al., 2022). This paper extends those methods with a rigorous importance score.

### Pal et al. (2023) — Future Lens
The paper cites "Future Lens" (Pal et al., 2023), which also anticipates subsequent tokens from intermediate hidden states, as related work in the vocabulary-projection tradition.

---

## 9. Implications for Understanding Transformer Computation

### Knowledge Is Localized but Distributed
The finding that ~300 neurons capture nearly all predictive signal for a specific fact (while random neurons of the same count have negligible effect) confirms that knowledge is **structurally localized** — not uniformly distributed — even though many neurons nominally participate.

### Hierarchical Computation in Deep Networks
The two-stage circuit (shallow FFN → deep attention → deep FFN → output) reveals a clear **functional hierarchy**:
- Early layers handle feature detection and transformation
- Middle layers route information via attention
- Late layers consolidate and output predictions
This hierarchical view is consistent with the broader mechanistic interpretability literature (Elhage et al., 2021) but is now grounded at neuron level.

### Both Modules Store Knowledge
A key empirical result is that **both attention and FFN layers store factual knowledge** — this contradicts earlier views that only FFN layers act as key-value memories (Geva et al., 2020). The dominance of attention vs. FFN depends on the knowledge type.

### Semantic Modularity in Attention Heads
The discovery that semantically similar knowledge types share attention heads suggests that attention heads implement **topic-specialized** computations rather than purely syntactic or positional operations. Heads for language/capital/country are the same; heads for number/color are different.

### Implications for Knowledge Editing
The paper directly motivates neuron-level knowledge editing: if a small set of neurons (300 value neurons or 1000 query neurons) controls a specific fact, precisely editing those neurons should change the model's output with minimal collateral damage. The paper suggests this as a future application, noting that editing query neurons may be preferable since they are shallower and more accessible.

### Implications for Granular Deletion / Suppression
For research on granular deletion (the project context of this repository):
- The negative-importance neurons (those that suppress target tokens) demonstrate that transformer computation involves active **competition and suppression**, not merely additive accumulation
- The large gap between "all" and "positive-only" importance scores (especially in FFN layers) means that any deletion mechanism must account for these suppressive components — removing a negative-importance neuron would *increase* a token's probability, not decrease it
- The query-value distinction suggests that granular deletion might require targeting the query neurons (which are more distributed and shallower) as well as the value neurons (which are more concentrated and deeper)
- The method of measuring importance via log probability increase (conditioned on context) is directly applicable to designing targeted deletion metrics — any score that ignores the running residual context will misidentify which neurons to remove

### Scalability
Because the method requires only a single forward pass and computes importance scores analytically (no backward pass, no repeated interventions for ranking), it scales to models with millions of neurons. This makes it practically usable for large-scale mechanistic analysis, in contrast to causal tracing or integrated gradients which require many forward/backward passes.

### Limitations Acknowledged
1. Only six knowledge types studied — generalization to other facts, syntax, or reasoning is not established
2. Only two models (GPT2-large and Llama-7B) — behavior in much larger models is unknown
3. Static methods compared only against other static methods; comparison with causal mediation analysis and gradient-based methods is deferred to future work
4. Query neuron interpretability in vocabulary space is poor — the mechanism by which query neurons encode features remains opaque
