# Notes: Layer-Targeted Multilingual Knowledge Erasure in Large Language Models

**Paper:** Wang et al. (2026), arXiv:2602.22562
**Authors:** Taoran Li (Texas A&M), Varun Chandrasekaran (UIUC), Zhiyuan Yu (Texas A&M)
**Date read:** 2026-03-20

---

## 1. Key Contribution: Using Logit Lens to Verify Knowledge Erasure

The paper's central methodological innovation is using **Logit Lens probing** as a mechanistic verification tool to distinguish genuine knowledge removal from surface-level output suppression — what prior work (Jia et al. 2025) calls the "Erasure Illusion."

The Logit Lens works by projecting intermediate hidden states directly into vocabulary space at each layer using the model's pre-trained embedding matrix:

```
P_lens(y | x, l) = softmax(E · LayerNorm(h_l))_y
```

This allows inspection of what the model "believes" the answer to be at each intermediate layer, before the final output is computed. The key diagnostic is the **probability assigned to the correct answer at intermediate layers** (referred to as "recall" in the paper).

The Logit Lens serves three distinct diagnostic functions in this paper:

1. **Confirming genuine erasure at target layers:** After MUTE intervention at Layers 9/16 in Llama-3.1, forget set recall drops to near-zero internally (English: 29.39% → 1.50%; Hindi: 25.42% → 1.18%). The knowledge is gone from the model's computation path, not just blocked at output.

2. **Exposing the shallow-layer failure mode:** At Layer 2, the Logit Lens shows that retain set recall for held-out languages collapses structurally (Hindi: 30.05% → 3.47%; German: 30.71% → 12.77%), proving the intervention destroyed representational capacity for those languages rather than selectively removing target knowledge.

3. **Exposing the deep-layer failure mode:** At Layer 30, forget set recall is essentially unchanged after optimization (English: 29.39% → 29.42%; German: 27.44% → 27.41%). The optimization had no effect on the stored knowledge. This explains the behavioral failure: knowledge was never modified, because deep layers handle token generation rather than semantic storage.

The Logit Lens results for target layers (L9, L16) simultaneously show near-zero forget recall and near-finetuned retain recall for all languages including held-out ones, providing a white-box confirmation that the intervention is both selective and complete.

---

## 2. How They Identify Which Layers to Target

The paper identifies "language-agnostic layers" — intermediate layers where cross-lingual representations converge — using two complementary metrics computed on the MMLU/MMMLU High School Chemistry subset.

### CKA (Centered Kernel Alignment) — measures multilingual alignment

For each layer l, hidden states are collected for aligned concepts across all languages. The pairwise CKA score between languages ℓ_j and ℓ_k is:

```
CKA_l(ℓ_j, ℓ_k) = HSIC(K, L) / sqrt(HSIC(K,K) · HSIC(L,L))
```

where K and L are Gram matrices of the hidden state matrices for each language. CKA is invariant to orthogonal transformations and isotropic scaling, making it robust for high-dimensional representations.

The aggregate alignment score for layer l is the average pairwise CKA across all supported languages. **Higher CKA = more language-invariant representations.**

### LRDS (Linguistic Regions Development Score) — measures language-specificity

LRDS quantifies the divergence between intra-language similarity and inter-language similarity for representations of semantically different inputs:

```
LRDS_l = E[S_l(s_p, s_q) | same_language, different_semantics]
        - E[S_l(s_p, s_q) | different_language, different_semantics]
```

where similarity is cosine similarity of normalized token-averaged hidden states. **LRDS near zero = semantics-based clustering (language-agnostic). High LRDS = language-based clustering (language-specific).**

### Combining into the Language-Agnostic Region Λ

Thresholds are computed as:
- τ_align = mean CKA across all layers (selects above-average alignment)
- τ_spec = α × min(LRDS) across all layers (α is architecture-dependent)

The language-agnostic region is: **Λ = {l | CKA_l ≥ τ_align AND LRDS_l ≤ τ_spec}**

### Results per architecture

| Model | E[CKA] | min(LRDS) | α | τ_spec | Region Λ |
|---|---|---|---|---|---|
| Llama-3.1-8B | 0.647 | 0.0039 | 2.5 | 0.0096 | Layers 8–23 |
| BLOOM-7b1 | 0.642 | 0.0017 | 4.4 | 0.0073 | Layers 5–22 |
| Qwen-2.5-7B | 0.512 | 0.0287 | 1.5 | 0.0441 | Layers 19–24 |

The LRDS plateau region (low language-specificity) consistently corresponds to intermediate layers. LRDS surges sharply in deep layers as representations re-specialize for language-specific generation.

### Selecting the specific target layer within Λ

The optimal layer within Λ depends on the unlearning algorithm's mechanism:

- **Parameter-based methods (RMU, SimNPO):** Select the *earliest* layer in Λ where CKA peaks. Early intervention ensures disruption propagates through all downstream layers, preventing reconstruction of the concept. → L9 for Llama-3.1.
- **Activation-based methods (SLUG):** Select the *deepest* layer in Λ. Deeper layers carry more fully-formed semantic representations, improving probe accuracy while remaining language-agnostic. → L20 for Llama-3.1.

Sensitivity analysis shows that varying α by ±0.5 produces modest changes in region boundaries but the selected target layers remain within Λ in all tested configurations.

---

## 3. The MUTE Framework Methodology

MUTE (Multilingual Unlearning via Targeted Erasure) is a two-stage framework:

### Stage 1: Target Layer Localization

1. For each layer l, compute CKA alignment (Align_l) and LRDS language-specificity (LRDS_l).
2. Compute architecture-specific thresholds τ_align and τ_spec.
3. Identify language-agnostic region Λ.
4. Select target layer l* within Λ based on the unlearning algorithm type.

### Stage 2: Layer-Targeted Unlearning

All model parameters are **frozen except for the target layer θ_{l*}**. Unlearning optimization uses only source language data (3 languages: EN, ES, PT). Three unlearning algorithms are adapted:

**Adapted RMU (Representation Misalignment Unlearning):**
- Restricts updates to single target layer l*_RMU (e.g., Layer 9 in Llama-3.1).
- Uses an L2 loss that pushes h_{l*}(x_forget) toward a fixed random vector u, scaled to match typical hidden state magnitude.
- Retains a constraint keeping representations of retain data close to frozen base model.
- By targeting the shared multilingual representation before it diverges into language-specific variations, the effect propagates to all languages.

**Adapted SLUG (Single Layer Unlearning Gradient):**
- Targets the deepest layer within Λ (l*_SLUG, e.g., Layer 20).
- Incorporates an explicit retain constraint (unlike the original SLUG which uses only forget gradient):
  `Δθ = ∇_{θ_{l*}} L_forget - α∇_{θ_{l*}} L_retain`
- Updates applied exclusively to self-attention matrices (Wq, Wk, Wv, Wo) at l*_SLUG; MLP parameters remain fixed.
- Single-step update: θ_{l*} ← θ_{l*} + λΔθ.

**Adapted SimNPO (Simple Negative Preference Optimization):**
- Freezes all parameters except target layer l*_RMU (same layer as RMU, since both are parameter-based).
- Gradients computed at final output layer but backpropagated only to update θ_{l*_RMU}.
- Uses reference-free SimNPO loss with length normalization:
  `L_SimNPO = -(2/β) log σ(-(β/|y|) log π_{θ_{l*}}(y|x) - γ)`
- Length normalization provides robustness to varying response lengths across languages and gives implicit regularization against utility degradation.

---

## 4. How Logit Lens Probing Confirms Genuine Erasure vs. Surface Masking

The mechanistic verification distinguishes three distinct internal signatures across intervention depths (Table 9 in the paper, Llama-3.1):

### Shallow layer (L2) — erasure + representational destruction
- **Forget set:** Internal recall drops to near-zero (English: 1.38%, Hindi: 0.53%). Erasure is real.
- **Retain set:** Internal recall for held-out languages collapses catastrophically:
  - Hindi: 30.05% → 3.47%
  - German: 30.71% → 12.77%
- **Interpretation:** Intervention does not selectively remove target knowledge. It destroys fundamental multilingual representational capacity. The model can no longer encode any concept in non-source languages.

### Deep layer (L30) — no effect on knowledge
- **Forget set:** Internal recall is virtually identical before and after:
  - English: 29.39% → 29.42%
  - German: 27.44% → 27.41%
- **Retain set:** Fully preserved.
- **Interpretation:** Optimization at deep layers completely fails to modify stored knowledge. Deep layers handle language-specific token generation; the semantic knowledge lives upstream. This is a complete failure to intervene, not an "Erasure Illusion" — it is erasure that never happened at all.

### Target layers (L9, L16) — selective knowledge removal
- **Forget set:** Internal recall drops to near-zero across ALL languages including held-out:
  - English: 29.39% → 1.50% (L9), 2.41% (L16)
  - Hindi (held-out): 25.42% → 1.18% (L9), 1.11% (L16)
  - German (held-out): 27.44% → 1.08% (L9), 1.74% (L16)
- **Retain set:** Internal recall remains close to finetuned baseline for all languages:
  - Hindi: 30.05% → 29.64% (L9)
  - Italian: 31.44% → 31.32% (L9)
  - German: 30.71% → 30.59% (L9)
- **Interpretation:** The target knowledge is removed from the model's computation path. The multilingual representation space is preserved. This is genuine knowledge removal, not output suppression.

The Logit Lens thus provides a white-box guarantee: because internal activations drop to near-zero at the intervention layer and remain near-zero in all subsequent layers, there is no upstream pathway through which the knowledge could be recovered via multilingual queries or other adversarial probing.

---

## 5. Models and Datasets Used

### Models

Three multilingual LLMs were selected to test across diverse architectures and pretraining distributions:

1. **Llama-3.1-8B** — primary model; Rotary Positional Embeddings (RoPE); strong general multilingual capabilities.
2. **Qwen-2.5-7B** — explicitly optimized for multilingual heavy-tail distributions; denser non-English pretraining; tests whether language-agnostic layers appear in models with broader multilingual training.
3. **BLOOM-7b1** — ALiBi (Attention with Linear Biases) positional embeddings; verifies architecture-agnosticism of the language-agnostic layer phenomenon.

Models were prepared via supervised fine-tuning with LoRA (rank 8, all linear layers, lr=5e-5, cosine scheduler, 2 epochs, bf16) using the LLaMA-Factory framework, injecting domain knowledge across all supported languages before unlearning experiments began.

### Datasets

- **MMLU** (Hendrycks et al. 2021) — base English benchmark.
- **MMMLU** (OpenAI 2024) — professionally translated multilingual versions of MMLU subjects.

**Forget set (Dforget):** High School Chemistry subset — used as a proxy for hazardous dual-use knowledge (203 samples/language).

**Retain set (Dretain):** History and Law subjects — High School World History (237), Professional Law (1534), International Law (121), Jurisprudence (108); used to measure preservation of general capabilities.

### Language splits

- **Source languages (training):** English (EN), Spanish (ES), Portuguese (PT) — 3 languages used for unlearning gradient computation.
- **Held-out languages (zero-shot evaluation):**
  - Llama-3.1: German (DE), French (FR), Hindi (HI), Italian (IT) — 4 languages
  - Qwen-2.5: Arabic (AR), German (DE), French (FR), Hindi (HI), Indonesian (ID), Italian (IT), Japanese (JA), Korean (KO), Chinese (ZH) — 9 languages
  - BLOOM: Arabic (AR), French (FR), Hindi (HI), Indonesian (ID), Chinese (ZH) — 5 languages

The alternative source language ablation (DE/FR/IT as source, EN/ES/PT as held-out) confirms results are not tied to specific language selection.

---

## 6. Key Results: Where Knowledge Lives and How to Delete It

### The Two Failure Modes (Pilot Study, Table 1)

The most fundamental finding is that intervention depth is not a free parameter — it determines whether multilingual unlearning succeeds or fails entirely. Two failure modes were characterized using RMU on Llama-3.1 (Layer 2 vs. Layer 30):

**Shallow layer (L2) — successful erasure, catastrophic utility loss:**
| Language | Forget (FT→Unl) | Retain (FT→Unl) |
|---|---|---|
| English* | 86% → 2% | 84% → 63% |
| German (held-out) | 68% → 0% | 58% → 1% |
| Italian (held-out) | 80% → 2% | 62% → 0% |
| Hindi (held-out) | 64% → 1% | 23% → 0% |

**Deep layer (L30) — utility preserved, zero erasure:**
| Language | Forget (FT→Unl) | Retain (FT→Unl) |
|---|---|---|
| English* | 86% → 86% | 84% → 84% |
| German (held-out) | 68% → 69% | 58% → 58% |

### Target Layer Results — Llama-3.1 (Table 3)

Layer 9 (early within Λ, parameter-based):
- Forget set: ≤ 2% across ALL 7 languages (source and held-out)
- Retain set: German 49%, Italian 61%, Hindi 20%, vs. finetuned 58%, 62%, 23%

Layer 16 (deeper within Λ):
- Forget set: 0% across all languages
- Retain set: English 79% vs. finetuned 84%; German 56% vs. 58%

### Cross-Model Summary (Table 2)

| Model | Shallow (Forget/Retain) | Target (Forget/Retain) | Deep (Forget/Retain) |
|---|---|---|---|
| Llama-3.1-8B (L9) | 1.0% / 21.9% | **1.3% / 54.7%** | 74.0% / 59.1% |
| Qwen-2.5-7B (L19) | 0.3% / 4.9% | **15.8% / 13.3%** | 33.7% / 14.0% |
| BLOOM-7b1 (L5) | 0.4% / 6.0% | **0.0% / 6.4%** | 12.9% / 7.9% |

The target layer consistently achieves the best erasure-utility trade-off across all architectures. The pattern is universal: shallow fails utility, deep fails erasure, target achieves both.

### Cross-Algorithm Summary — Llama-3.1 (Table 4)

| Algorithm | Shallow (Forget/Retain) | Target (Forget/Retain) | Deep (Forget/Retain) |
|---|---|---|---|
| RMU (L9) | 1.0% / 21.9% | **1.3% / 54.7%** | 74.0% / 59.1% |
| SLUG (L20) | 0.0% / 0.0% | **35.4% / 31.4%** | 75.0% / 59.7% |
| SimNPO (L9) | 0.7% / 55.3% | **0.0% / 58.6%** | 68.9% / 58.1% |

SimNPO achieves the best overall trade-off at the target layer (0% forget, 58.6% retain). SLUG at target achieves partial erasure only; its single-step gradient update requires additional regularization.

### Key structural finding: Where knowledge lives

The paper establishes that factual/semantic knowledge in multilingual LLMs is primarily encoded in **intermediate layers** (roughly 25-65% depth depending on architecture):
- Layers 8–23 in Llama-3.1-8B (32 total)
- Layers 5–22 in BLOOM-7b1 (30 total)
- Layers 19–24 in Qwen-2.5-7B (28 total)

These are the layers where the same concept, expressed in any language, maps to similar hidden state representations. Shallow layers encode language-specific surface features; deep layers re-specialize for language-specific output generation. Semantic knowledge — the actual factual content that unlearning targets — lives in the middle.

### Comparison to baseline (Table 7)

All-layer gradient ascent cannot balance erasure and utility. MUTE achieves a **2.75× improvement in retain accuracy** over aggressive GA at equivalent erasure levels (~1% forget accuracy):
- Aggressive GA (EN/ES/PT, lr=1e-4): 1% forget, 20% retain
- MUTE (EN/ES/PT, L9): 1% forget, 55% retain

---

## 7. Implications for Granular Deletes

### Intervention location matters as much as intervention content

This paper reframes the unlearning problem: the question is not only *what to optimize* but *where in the model to intervene*. For granular deletes, this means:

- Deleting a specific concept requires identifying the layers where that concept's semantic representation resides, not just running gradient-based optimization globally.
- Different types of knowledge may reside at different depths. The paper specifically targets factual domain knowledge (chemistry), but the CKA-LRDS methodology is general and could be applied to locate other knowledge types.

### The language-agnostic region as a hub for cross-concept generalization

The reason MUTE achieves zero-shot multilingual transfer by training on only 3 source languages is that the target layers encode *semantics independent of surface form*. This has a broader implication: the same intermediate layers that generalize across languages may also generalize across phrasings, reformulations, and contexts in a single language. Granular deletes targeting these layers may be more robust to paraphrase-based recovery attempts than output-level suppression methods.

### Genuine erasure vs. the Erasure Illusion

The Logit Lens verification methodology provides a practical tool for auditing whether a delete is genuine. Any unlearning method that operates only on final-layer outputs risks creating the Erasure Illusion (Jia et al. 2025) where the model still encodes the knowledge internally and could surface it under adversarial probing. MUTE's approach — and the Logit Lens check — provides an internal activation-level test for genuine deletion.

For granular delete systems, this suggests a two-tier evaluation:
1. Behavioral (QA accuracy on forget/retain sets) — necessary but insufficient.
2. Mechanistic (Logit Lens recall at multiple layers) — confirms the knowledge is removed from the computation path.

### Single-layer targeting is feasible

MUTE demonstrates that updating *a single layer's parameters* is sufficient to erase a concept across up to 12 languages when that layer is correctly identified. This has significant implications for granular deletes in deployment settings:
- Minimal parameter modification (high surgical precision).
- Reduced risk of collateral damage to unrelated knowledge.
- Efficient: only 609 forget samples (203 questions × 3 languages) needed.

### Architecture-specific calibration is required

The language-agnostic region falls at different relative depths across architectures (Llama: ~28-72% depth, BLOOM: ~17-73% depth, Qwen: ~68-86% depth). Any granular delete pipeline must include a CKA-LRDS profiling step for the specific model being modified — there is no universal target layer.

The α hyperparameter for LRDS threshold is also architecture-dependent (1.5–4.5 in these experiments). However, the paper provides a practical heuristic: visualize the CKA-LRDS curves and select α to capture the stable low-LRDS plateau before the characteristic surge toward deep layers.

### Algorithm choice affects the erasure-utility trade-off

For granular deletes where utility preservation is critical:
- **SimNPO** offers the best balance (0% forget, ~58% retain on Llama-3.1 at Layer 9).
- **RMU** is effective with moderate utility preservation (~1% forget, ~55% retain).
- **SLUG** (single-step, gradient-based) requires additional regularization to avoid over-erasure.

### Limits: partial erasure in some architectures

Qwen-2.5 shows less complete erasure at the target layer (11–26% forget accuracy across languages, vs. near-0% for Llama-3.1 and BLOOM). The paper attributes this to the flatter LRDS profile in Qwen-2.5 and a narrower language-agnostic region (Layers 19–24). This suggests granular deletes may be harder in models with denser multilingual pretraining, where language-agnostic representations are less concentrated in a single region.

### The paradigm shift stated explicitly

The paper's conclusion frames the core contribution as: shifting from "what data to optimize on" to "where in the model to intervene." For granular deletes research, this is a direct challenge to methods that treat layers as interchangeable and apply updates globally. The mechanistic structure of the model — the hierarchical progression from surface features to language-agnostic semantics to language-specific generation — must be respected for targeted deletions to work.

---

## Summary Table: Layer Behavior at a Glance

| Layer Region | CKA | LRDS | Effect on Forget | Effect on Retain | Mechanism |
|---|---|---|---|---|---|
| Shallow (early ~10-25%) | Low | Low | Erased | Collapsed (held-out) | Destroys multilingual primitives |
| Language-agnostic (intermediate ~25-75%) | High | Low | Erased | Preserved | Disrupts shared semantic representation |
| Deep (late ~75-100%) | High→Low | High | Not erased | Preserved | Only token generation; knowledge upstream |

---

## Citation

Li, T., Chandrasekaran, V., & Yu, Z. (2026). Layer-Targeted Multilingual Knowledge Erasure in Large Language Models. arXiv:2602.22562.
