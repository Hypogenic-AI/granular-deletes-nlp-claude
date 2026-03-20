# Literature Review: Characterizing Granular Deletes via Logit Lens and Residual Stream Geometry

## Research Area Overview

This review covers the intersection of three active research threads in mechanistic interpretability: (1) **logit lens and related tools** for probing what information is encoded at each layer of a transformer, (2) **geometry of the residual stream** — how concepts, features, and belief states are spatially organized in activation space, and (3) **granular information deletion** — mechanistic approaches to selectively removing specific knowledge from language models. The hypothesis under investigation is that logit lens techniques can identify *where* information is deleted from the residual stream and help map the geometry of that stream.

---

## Key Papers

### 1. Interpreting GPT: The Logit Lens (nostalgebraist, 2020)
- **Source**: LessWrong blog post
- **Key Contribution**: Introduced the idea of projecting intermediate hidden states through the unembedding matrix to decode what the model "believes" at each layer. The logit lens shows predictions converging gradually to the final output, with earlier layers producing rough semantic approximations.
- **Relevance**: Foundational method. Establishes that the residual stream carries progressively refined predictions that can be read out at any layer.

### 2. Eliciting Latent Predictions from Transformers with the Tuned Lens (Belrose et al., 2023)
- **Authors**: Belrose, Ostrovsky, McKinney, Furman, Smith, Halawi, Biderman, Steinhardt
- **arXiv**: 2303.08112
- **Key Contribution**: Addresses logit lens failures (bias of 4-5 bits KL, failure on BLOOM/OPT) by training per-layer affine translators. The tuned lens is unbiased, causally faithful (Spearman ρ up to 0.98), and works across 50+ model variants.
- **Methodology**: Affine probe `TunedLens(h_ℓ) = LayerNorm[A_ℓ·h_ℓ + b_ℓ]·W_U` trained with KL distillation loss.
- **Key Findings**: (a) Predictions converge smoothly across layers — iterative inference view. (b) Representation drift is approximately affine between layers. (c) "Rogue dimensions" dominate covariance at certain layers. (d) Causal basis extraction identifies the most influential residual stream directions. (e) Layer deletion is nearly harmless except for layer 1.
- **Code**: https://github.com/AlignmentResearch/tuned-lens

### 3. Neuron-Level Knowledge Attribution in Large Language Models (Yu & Ananiadou, 2023)
- **arXiv**: 2312.12141
- **Key Contribution**: Decomposes the residual stream into individual neuron-level contributions and attributes factual knowledge to ~300 specific neurons (200 attention + 100 FFN) that capture 97-99% of the predictive signal.
- **Key Findings**: (a) Two-stage information flow: shallow/medium FFN → deep attention → deep FFN → output. (b) ~300 neurons capture nearly all knowledge signal; random same-count intervention causes <0.2% effect. (c) Semantically similar knowledge types share attention heads. (d) Large cancellation effects from suppressive neurons, especially in FFN layers.
- **Datasets**: TriviaQA, filtered to 6 knowledge types
- **Models**: GPT2-large, Llama-7B
- **Relevance**: Directly applicable to granular deletes — identifies which neurons to target for surgical knowledge removal.

### 4. The Linear Representation Hypothesis and the Geometry of Large Language Models (Park et al., 2023)
- **Authors**: Park, Choe, Veitch (University of Chicago)
- **arXiv**: 2311.03658
- **Key Contribution**: Formalizes the linear representation hypothesis into three equivalent notions (subspace, measurement, intervention) and introduces the **causal inner product** `⟨γ̄, γ̄'⟩_C = γ̄ᵀ Cov(γ)⁻¹ γ̄'` — the correct metric for measuring concept orthogonality in the residual stream.
- **Key Findings**: (a) 26/27 tested concepts have linear representations. (b) Causally separable concepts are orthogonal under the causal (not Euclidean) inner product. (c) Steering vectors derived from the causal inner product change target concepts without affecting orthogonal concepts. (d) The Euclidean inner product fails for some architectures (Gemma-2B).
- **Models**: LLaMA-2-7B, Gemma-2B
- **Code**: https://github.com/KihoPark/linear_rep_geometry
- **Relevance**: Defines the correct geometry for analyzing concept directions in the residual stream, essential for understanding what "deleting a concept" means geometrically.

### 5. Transformers Represent Belief State Geometry in their Residual Stream (Shai et al., 2024)
- **arXiv**: 2405.15943 (NeurIPS 2024)
- **Key Contribution**: Shows that transformers trained on HMM-generated data linearly represent **belief state geometry** — including fractal structures — in their residual stream. The model encodes full future distributions, not just next-token predictions.
- **Key Findings**: (a) Fractal belief state geometry is linearly embedded in the final-layer residual stream (Mess3 process). (b) When belief states are degenerate in next-token space (RRXOR), the geometry is distributed across layers and absent at the final layer. (c) Pairwise distance correlation with belief states: R²=0.95 vs. R²=0.31 for next-token distances. (d) Linearity is empirically discovered, not mathematically required.
- **Relevance**: Provides a theoretical framework for understanding *what* the residual stream encodes — and why information may be distributed across layers rather than concentrated at any single layer.

### 6. Characterizing Stable Regions in the Residual Stream of LLMs (Janiak et al., 2024)
- **arXiv**: 2409.17113 (NeurIPS 2024 Workshop)
- **Key Contribution**: Identifies **stable regions** — contiguous zones in activation space where small perturbations don't change model output, separated by sharp boundaries where small perturbations cause discrete output changes. Stable regions are much coarser than polytopes (hundreds to thousands of polytope boundaries within one stable region).
- **Key Findings**: (a) Stable regions emerge during training. (b) Larger models have sharper boundaries. (c) Semantic content aligns with region membership. (d) Perturbations within a stable region are "absorbed" by the network.
- **Models**: OLMo (1B, 7B), Qwen2 (0.5B, 1.5B, 7B)
- **Relevance**: Directly constrains granular deletes — effective deletion requires pushing activations across region boundaries, not just perturbing within a region.

### 7. Mechanistic Unlearning: Robust Knowledge Unlearning and Editing via Mechanistic Localization (Ghosh et al., 2024)
- **arXiv**: 2410.12949
- **Key Contribution**: Demonstrates that targeting the **Fact Lookup (FLU) mechanism** (early-to-middle MLPs, layers 2-8) for knowledge editing is dramatically more robust than targeting output-tracing (OT) components (later-layer extraction mechanisms).
- **Key Findings**: (a) Knowledge has a two-stage structure: storage (FLU, early-mid MLPs) and extraction (later attention/MLPs). (b) OT methods achieve 0% in-distribution forget accuracy but allow 47-63% recovery via adversarial relearning; FLU limits this to ~20%. (c) After OT editing, linear probes still detect ground-truth knowledge in early layers; after FLU editing, probes converge to the edited answer. (d) FLU editing generalizes to MCQ format (>40% advantage).
- **Datasets**: Sports Facts (1,567 athletes), CounterFact
- **Models**: Gemma-7B, Gemma-2-9B, Llama-3-8B
- **Relevance**: Provides the clearest evidence that granular deletion must target the knowledge *storage* stage (where the residual stream is enriched), not the *extraction* stage.

### 8. Layer-Targeted Multilingual Knowledge Erasure in LLMs (Li et al., 2026)
- **arXiv**: 2602.22562
- **Key Contribution**: Uses **Logit Lens probing** to verify genuine knowledge erasure vs. surface masking ("Erasure Illusion"). Introduces the MUTE framework targeting language-agnostic intermediate layers identified via CKA and LRDS metrics.
- **Key Findings**: (a) Logit Lens reveals three layer regimes: shallow (erasure + utility collapse), target (genuine erasure), deep (no effect). (b) Language-agnostic layers (Llama L8-23, BLOOM L5-22, Qwen L19-24) are where semantic knowledge resides. (c) Single-layer targeting achieves cross-lingual erasure from only 3 source languages. (d) 2.75× utility improvement over global gradient ascent.
- **Relevance**: Most direct demonstration of using logit lens to characterize where information deletion occurs in the residual stream.

---

## Common Methodologies

### Residual Stream Probing
- **Logit Lens / Tuned Lens**: Project intermediate states through unembedding to decode layer-by-layer predictions (Papers 1, 2, 8)
- **Linear Probing**: Train logistic regression on residual stream activations to detect specific knowledge (Papers 3, 5, 7, 8)
- **Causal Basis Extraction**: Find most influential directions via sequential ablation (Paper 2)

### Knowledge Localization
- **Activation Patching / Causal Tracing**: Replace activations to measure causal contribution (Papers 7)
- **Path Patching**: Trace causal paths through specific components (Paper 7)
- **CKA / LRDS Profiling**: Identify language-agnostic layers via alignment metrics (Paper 8)
- **Neuron-level Attribution**: Log-probability increase scoring per neuron (Paper 3)

### Geometric Analysis
- **Causal Inner Product**: `Cov(γ)⁻¹`-based metric for concept orthogonality (Paper 4)
- **Interpolation Analysis**: Linear interpolation between activations to find stable region boundaries (Paper 6)
- **Belief State Regression**: Affine map from activations to probability simplex (Paper 5)

---

## Standard Baselines

- **Logit Lens** (nostalgebraist, 2020): Zero-parameter baseline for layer-wise prediction decoding
- **Causal Tracing / Attribution Patching**: Output-tracing localization methods (ROME, Meng et al., 2022)
- **Gradient Ascent**: Standard unlearning baseline — maximize loss on forget set
- **All-MLPs / Nonlocalized**: Edit all MLP parameters or all parameters
- **Random component selection**: Baseline for localization methods

## Evaluation Metrics

- **Forget Accuracy** (↓): Remaining recall of deleted knowledge
- **Retain Accuracy** (↑): Preservation of unrelated knowledge
- **MCQ Generalization**: Whether deletion generalizes to alternative prompt formats
- **Adversarial Relearning**: Recovery of deleted knowledge via fine-tuning
- **Logit Lens Recall**: Internal activation-level measurement of knowledge presence
- **Linear Probe Accuracy**: Whether knowledge is detectable in intermediate residual stream
- **KL Divergence**: Lens output vs. final layer distribution
- **Pairwise Distance Correlation**: Geometric fidelity of residual stream projections

---

## Datasets in the Literature

| Dataset | Used In | Task |
|---------|---------|------|
| CounterFact | Papers 7, 8 | Factual knowledge editing/deletion |
| Sports Facts | Paper 7 | Knowledge unlearning (athlete-sport) |
| TOFU | Unlearning literature | Fictitious author unlearning |
| TriviaQA | Paper 3 | Factual knowledge probing |
| MMLU / MMMLU | Paper 8 | Multilingual knowledge evaluation |
| The Pile | Paper 2 | General language modeling |
| LAMA | Probing literature | Cloze-style knowledge probing |

---

## Gaps and Opportunities

1. **No unified framework connecting logit lens to deletion geometry**: Papers study logit lens (2, 8), geometry (4, 5, 6), and deletion (7, 8) separately. No work systematically maps how the logit lens signal changes across layers during and after targeted deletion, in the context of the causal inner product or stable region geometry.

2. **Scale of geometric analysis**: Belief state geometry (Paper 5) is demonstrated only on small models (4-layer, d=64). Whether fractal belief state structures appear in production-scale models is unknown.

3. **Stable regions and deletion**: Paper 6 shows stable regions constrain minimum edit size, but no one has used stable region analysis to design deletion interventions or verify their completeness.

4. **Causal inner product at intermediate layers**: Paper 4's causal inner product is defined only at the final layer. Extending it to intermediate layers would provide a principled metric for measuring concept deletion at each layer.

5. **Suppressive neurons in deletion**: Paper 3 identifies neurons that actively suppress knowledge. The role of these suppressive neurons during unlearning is unstudied — does deletion work by enhancing suppression or by removing the positive signal?

---

## Recommendations for Our Experiment

### Recommended Datasets
1. **CounterFact** (primary): Tight (subject, relation, object) format, counterfactual pairs, used by both mechanistic unlearning and ROME. Enables tracking of where factual predictions appear/disappear in the residual stream.
2. **TOFU** (secondary): Controlled synthetic knowledge for clean unlearning evaluation.
3. **Sports Facts**: Simple, well-studied knowledge type with established FLU mechanism characterization.

### Recommended Baselines
1. **Logit Lens** and **Tuned Lens**: Layer-by-layer prediction tracking before/after deletion
2. **Causal Tracing** (ROME): Output-tracing localization for comparison
3. **FLU Localization** (Mechanistic Unlearning): Mechanistic localization baseline
4. **Gradient Ascent**: Standard unlearning baseline

### Recommended Metrics
1. **Logit Lens recall at each layer**: Primary metric for tracking information deletion through the residual stream
2. **Linear probe accuracy per layer**: Detect latent knowledge remaining after deletion
3. **Stable region interpolation curves**: Characterize geometric impact of deletion
4. **Causal inner product between concept directions**: Measure geometric isolation of deleted concepts
5. **MCQ and adversarial relearning**: Robustness evaluation

### Methodological Considerations
- Use **TransformerLens** for activation access and hook-based interventions
- Start with **GPT-2** or **Pythia** models for tractability, validate on **Llama-3-8B** or **Gemma-7B**
- Layer-by-layer logit lens analysis before and after deletion interventions is the core experimental paradigm
- The **causal inner product** (not Euclidean) should be used for all geometric analyses
- Compare FLU-targeted vs. output-tracing-targeted deletions via logit lens to directly test the hypothesis
