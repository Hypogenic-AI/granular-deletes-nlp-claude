# Notes: Mechanistic Unlearning: Robust Knowledge Unlearning and Editing via Mechanistic Localization

**Citation:** Ghosh, Guo, Syed, Sheshadri, Ewart, Dziugaite (2024). arXiv:2410.12949v2.
**Authors:** Phillip Guo, Aaquib Syed, Abhay Sheshadri, Aidan Ewart, Gintare Karolina Dziugaite
**Affiliations:** University of Maryland, Georgia Tech, University of Bristol, Google DeepMind

---

## 1. Research Question and Key Contribution

### Research Question
Can mechanistic interpretability — specifically the identification of which internal model components (circuits) implement specific high-level computational mechanisms — improve the precision, robustness, and effectiveness of knowledge editing and unlearning in LLMs?

### Core Insight
The paper introduces a critical distinction between two types of localization:

- **Output-Tracing (OT) localization**: Measures the causal effect of components on the *output* (e.g., Causal Tracing, Attribution Patching). These methods tend to identify "attribute extraction" components — the later-layer mechanisms that read out and format factual information into the output token — rather than the earlier-layer components where the knowledge is actually encoded.
- **Mechanistic localization (FLU)**: Identifies components based on their role in a well-understood *intermediate* computational mechanism — specifically, the "Fact Lookup" (FLU) mechanism, which enriches the residual stream with subject-attribute information before any output is produced.

### Key Contribution
Editing or unlearning weights localized to the FLU mechanism (early-to-middle MLP layers responsible for enriching subject representations with factual attributes) is substantially more robust than editing components found by OT methods or baseline approaches. The paper coins the term **"mechanistic unlearning"** for this approach.

### Central Claim
OT methods find the *extraction* stage (where latent knowledge is read out to the output) rather than the *storage* stage (where knowledge is first encoded into the residual stream). Targeting extraction leaves the knowledge in the latent stream where alternative extraction mechanisms can recover it. Targeting the storage stage via FLU disrupts the latent knowledge itself.

---

## 2. How They Localize Knowledge in the Model

### The Two-Stage Factual Recall Model
Building on Nanda et al. (2023) and Geva et al. (2023), the paper adopts a two-stage model of factual recall:

1. **Fact Lookup (FLU) stage** — Early-to-middle MLP layers enrich the latent representation of the subject token with information about the subject's attributes (e.g., "Jordan" → enriched with "basketball player" information). This happens in intermediate residual stream positions and does not directly write to the output logits.

2. **Attribute Extraction stage** — Later attention heads and MLPs read the enriched subject representation from the residual stream and move/format that information to the final token position, increasing the logit for the correct answer. These components have high direct logit importance.

### Which Layers/Components Implement FLU

**For Sports Facts (across Gemma-7B, Gemma-2-9B, Llama-3-8B):**
- FLU is localized to the MLP layers where **linear probe accuracy for the correct sport increases rapidly**.
- Specifically: layers **2 through 7** in Gemma-7B, and layers **2 through 8** in Gemma-2-9B and Llama-3-8B.
- These are early-to-middle layers. After these layers, probe accuracy plateaus near 100% for the athlete token's representation, indicating the sport attribute is already encoded by these layers.
- Attention heads past layer 2 also affect attribute representations but were excluded from FLU localization because (following Geva et al. and Nanda et al.) MLPs are the primary site of factual representation enrichment; attention heads may serve other roles (e.g., token concatenation).

**For CounterFact:**
- Path patching is used first to identify "fact extraction mechanism" components (attention heads and MLPs with high direct path importance to the final logit difference).
- Then path patching again identifies MLPs whose outputs to those extraction components most increase logit difference (threshold: > 0.02 change). These upstream MLPs form the FLU localization for CounterFact.
- The resulting FLU components are again concentrated in early-to-middle layers.

### What OT Methods Find (By Contrast)
- Attribution Patching (AP) assigns highest scores to **later-layer MLPs** (extraction stage), not early FLU layers, across all models and both datasets.
- Causal Tracing (CT) highlights some early-layer MLPs in Gemma-2-9B but primarily targets later extraction layers in other models.
- In parameter-count terms (Table 2, CounterFact): AP covers 60% of extraction MLP parameters but only 9.1% of fact lookup parameters; CT covers 30% of extraction MLPs and 36.4% of fact lookup parameters; FLU covers 100% of fact lookup and 0% of extraction.

---

## 3. Mechanistic Localization Method

### For Sports Facts: Probe-Based Localization
1. Train logistic regression probes on the internal activations of each layer to predict the correct sport, on the maintained (non-edited) athlete set.
2. Identify the layers where probe accuracy **rapidly increases** — these are the layers where the FLU mechanism is active.
3. The FLU localization is the set of MLP modules in those layers.

This is inspired directly by Nanda et al. (2023)'s fact-finding analysis on Pythia-2.8B, replicated on the three larger models used here.

### For CounterFact: Path Patching-Based Localization
1. Use **path patching** (Goldowsky-Dill et al., 2023) to find which attention heads and MLPs have large direct causal effects on the logit difference between the correct and incorrect answers → identifies "attribute extraction" components.
2. Apply path patching again, this time measuring the contribution of each MLP's output *as mediated through the extraction components* (patching MLP → extraction mechanism paths). MLPs with contribution > 0.02 are included in the FLU localization.
3. This identifies MLPs that enrich representations later used by extraction heads — the fact storage/enrichment site.

### Key Conceptual Distinction
FLU localization targets **intermediate representations** used by the factual recall mechanism, not direct effects on output logits. OT methods measure direct effects on output and therefore favor the final extraction stage.

---

## 4. How Information Is Deleted/Edited

### Localized Fine-Tuning
Once the FLU-localized components (specific early-to-middle MLP layers) are identified, weight updates are restricted **exclusively** to those components. The update method is gradient descent on a composite loss:

**For editing (changing a fact to a false target):**
```
L = λ₁·L_injection + λ₂·L_retain + λ₃·L_SFT
```
- `L_injection`: Cross-entropy loss on forget facts, maximizing probability of the false target answer.
- `L_retain`: Cross-entropy loss on non-forget facts (retain set), preserving other knowledge.
- `L_SFT`: Cross-entropy loss on the Pile dataset, preserving general language modeling capability.

**For unlearning (removing a fact without injecting a replacement):**
```
L = λ₁·L_forget + λ₂·L_retain + λ₃·L_SFT
```
- `L_forget`: Uses the `log(1 − p)` measure (probability of correct sport), from Mazeika et al. (2024), chosen for empirical stability over vanilla gradient ascent.

### Weight Masking (Parameter Efficiency Analysis)
A separate analysis trains a **binary differentiable mask** over individual weights within the localized components, with L1 regularization to control sparsity. This allows controlled comparison of edit sizes across localizations. Even when controlling for the exact number of modified parameters, FLU localization consistently outperforms others.

### Optimizer and Training Details
- AdamW optimizer, batch size 4, 16 gradient accumulation steps, 50 iterations.
- Cosine annealing scheduler.
- Gemma-2-9B requires 8-bit optimizer (memory constraints).
- Learning rates swept over {2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4}; sensitive to localization type.
- Sequential CounterFact editing: 100 total iterations, 25 per subset of 16 facts.

---

## 5. Use of Logit Lens / Residual Stream Analysis

### Probe-Based Residual Stream Analysis (Yes — Central Method)
The paper's core localization technique for Sports Facts is **logistic regression probing of the residual stream** at each layer:
- Probes are trained to predict the correct sport from the internal activation (residual stream state) at each layer.
- The sharp increase in probe accuracy across layers 2–7/8 identifies the FLU mechanism.
- Post-editing probing is used as an evaluation: probes trained on the maintained set are applied to the forget set, measuring whether the edited model's residual stream still represents the ground truth or has been updated to reflect the edited answer.

### Logit Lens (Implicit, via Path Patching)
The paper does not use the logit lens (projecting intermediate residual stream states through the unembedding matrix) by name. However, the CounterFact localization method uses **path patching** to measure logit difference effects, which is conceptually related — it quantifies how much each component contributes to correct-answer logit probability.

### Key Probing Result (Latent Knowledge Analysis, Section 3.3)
After editing with different localizations:
- **FLU localization**: Probes on the forget set consistently predict the **edit answer** (not the ground truth) starting from early layers, with monotonically increasing edit accuracy and decreasing forget accuracy. This confirms that FLU editing disrupts the latent knowledge at its source.
- **OT localizations (CT, AP)**: Probes on the forget set show near-100% classification of the **ground truth** answer in early layers even after editing. The knowledge remains intact in the residual stream; only the extraction stage has been changed. This explains why OT-edited models are vulnerable to alternative extraction methods.
- Attribution Patching in particular maintains 100% probing forget accuracy across many layers after editing — meaning the ground truth is essentially unchanged in internal representations.

---

## 6. Datasets and Models Used

### Datasets
1. **Sports Facts** (Nanda et al., 2023): Subject-sport relations for 1,567 athletes across three sports (basketball, baseball, football), with golf as a foil.
   - *Sports-Athlete-Editing*: Edit 16 or 64 randomly selected athletes to a different sport.
   - *Full-Sports-Editing*: Edit all athletes in one sport to golf.
   - *Sports-Unlearning*: Remove sport associations for all athletes in one sport.

2. **CounterFact** (Meng et al., 2023): Diverse factual associations. Filtered to facts where the model assigns >50% probability to the correct answer.
   - *CounterFact-Editing*: Edit 16 or 64 facts to false targets.
   - *Sequential-CounterFact-Editing*: Edit 64 facts in four sequential batches of 16 (exploiting per-fact localization).

### Models
- **Gemma-7B** (~28 layers)
- **Gemma-2-9B** (~42 layers)
- **Llama-3-8B** (~32 layers)

Pythia-2.8B and GPT-2 (used in prior interpretability work) were excluded because the larger models have stronger general capabilities enabling richer side-effect measurement and more diverse prompting formats. All main results are averaged across the three models.

---

## 7. Evaluation Metrics and Baselines

### Primary Metrics
- **Forget Accuracy** (↓): How accurately the model still recalls the ground truth answer on edited facts.
- **Forget Error** (↑): 1 − Forget Accuracy.
- **Edit Accuracy** (↑): How accurately the model produces the new (false) target answer.
- **Maintain Accuracy** (↑): Accuracy on non-edited facts.
- **MMLU Accuracy** (↑): Proxy for general language understanding.

### Robustness Metrics
- **MCQ Forget Accuracy** (↓) / **MCQ Edit Accuracy** (↑): Same facts in multiple-choice format (not seen during training), testing generalization of edits across prompt formats.
- **Paraphrase Edit Accuracy** (↑): CounterFact's paraphrase prompts — edited answer should generalize to paraphrased questions.
- **Neighborhood Edit Error** (↓): CounterFact's neighborhood prompts — edited answer should NOT overgeneralize to semantically similar but distinct facts.

### Adversarial Evaluations
- **Adversarial Relearning**: Retrain on half the forget set (rank-512 LoRA across all linear modules, 20 iterations), evaluate on the other half. Tests whether editing truly removes the factual association or merely obfuscates it.
- **Soft Prompt Attack**: Optimize continuous embeddings appended to prompts (half forget set) to recover correct answers; evaluate on other half. A narrow form of few-shot finetuning.

### Localization Methods Compared
| Method | Type | Description |
|---|---|---|
| **FLU (Fact Lookup)** | Mechanistic | Early-to-middle MLPs from probe/path-patching analysis |
| **Causal Tracing (CT)** | OT | Repeated activation patching, full components |
| **Causal Tracing MLPs** | OT | CT restricted to MLP components only |
| **Attribution Patching (AP)** | OT | Linear approximation of activation patching |
| **Attribution Patching MLPs** | OT | AP restricted to MLP components only |
| **Nonlocalized** | Baseline | All model parameters updated |
| **All-MLPs** | Baseline | All MLP parameters updated |
| **Random** | Baseline | Random selection of components |
| **Random-MLPs** | Baseline | Random MLP components |

Parameter counts are standardized across OT, FLU, and Random localizations where possible.

---

## 8. Key Results

### 8.1 Prompting Robustness (MCQ Format)
- FLU editing is the only method that meaningfully generalizes to the MCQ format for Sports-Athlete-Editing: **>40% advantage in MCQ Edit Accuracy** over all other localizations.
- OT localizations (CT, AP) achieve near-zero MCQ Edit Accuracy despite succeeding on the in-distribution prompt format — they have overfit to the extraction mechanism for the training prompt, not changed the underlying knowledge.
- For Full-Sports-Editing (where the association is simpler and more redundant), all localizations perform similarly, with FLU slightly superior on MCQ.
- For CounterFact, FLU is most robust in MCQ and Paraphrase evaluations and does not over-generalize to Neighborhood prompts.

### 8.2 Adversarial Relearning
- OT methods (CT, CT-MLPs) allow relearning to recover **63% and 47%** of forgotten accuracy respectively, with just 20 relearning steps on half the forget set.
- **FLU localization limits recovery to ~20%**, the lowest of all methods.
- This gap suggests OT edits merely mask the retrieval pathway; the knowledge is still present in the model weights and easily recoverable. FLU edits genuinely disrupt the encoded knowledge.
- For CounterFact and Full-Sports-Editing, all methods show minimal relearning (CounterFact is too hard to relearn; Full-Sports-Editing facts are not independent).

### 8.3 Latent Knowledge (Probe Analysis)
- After FLU editing: early-layer probes on the forget set monotonically converge toward predicting the *edited* answer (forget accuracy → 0, edit accuracy → 1 across layers).
- After OT editing: early-layer probes still classify the ground truth answer at near-100% accuracy — the ground truth is fully preserved in intermediate representations.
- Attribution Patching is the worst in this respect: ground truth is retained in probes across virtually all layers.
- This provides direct mechanistic evidence that OT editing targets extraction (output formatting), while FLU editing targets the latent knowledge source.

### 8.4 Parameter Efficiency (Weight Masking)
- FLU achieves superior MCQ generalization with the same number of masked weights, and is better at preserving MMLU while reducing forget accuracy.
- When the size of the edit (number of masked weights) is swept, only FLU consistently maintains MCQ generalization as the edit size grows; other methods may eliminate in-distribution accuracy but fail MCQ throughout the sweep.

### 8.5 Mechanism Weight Analysis
Table 3 (discretized weight masks, ~6M weights): proportions of each mechanism modified:
| Method | Extraction Heads | Extraction MLPs | Fact Lookup |
|---|---|---|---|
| AP | 0.60% | 0.14% | 0.12% |
| CT | 0.11% | 0.15% | 0.13% |
| **FLU** | **0.0%** | **0.0%** | **0.55%** |
| Nonlocalized | 1.3% | 0.12% | 0.10% |

Even at matched parameter count, AP and CT touch more extraction mechanism parameters than FLU mechanism parameters, confirming they prioritize extraction over storage.

### 8.6 OT Localization Behavior
- For Gemma-7B: both CT and AP target **later-layer MLPs** (extraction stage), not early FLU layers.
- For Gemma-2-9B: CT partially overlaps with FLU layers; AP still targets later layers.
- For Llama-3-8B: CT targets extraction layers; AP targets later layers.
- This mismatch between OT localization and the FLU mechanism is the root cause of OT editing's fragility.

### 8.7 Sports-Unlearning (Table 1)
Direct comparison for unlearning all basketball athletes:
| Method | Forget ↓ | Retain ↑ | MCQ ↓ | MMLU ↑ |
|---|---|---|---|---|
| Attrib. Patching | 0.000 | 1.000 | 0.767 | 0.602 |
| Causal Tracing | 0.201 | 0.998 | 0.849 | 0.611 |
| **FLU** | **0.002** | 0.995 | **0.110** | **0.613** |
| Random | 0.952 | 0.980 | 0.822 | 0.612 |
| All-MLPs | 0.000 | 0.994 | 0.279 | 0.606 |
| Nonlocalized | 0.000 | 0.985 | 0.196 | 0.595 |

FLU achieves near-zero forget accuracy and dramatically lower MCQ forget accuracy (0.110 vs. 0.767–0.849 for OT methods), with the best MMLU retention.

### 8.8 Sequential Editing
Sequential editing (64 facts edited in four batches of 16) shows marginally better MCQ robustness than editing all 64 at once. This is attributed to sequential edits being able to exploit per-fact localization (different facts reside in slightly different model components).

### 8.9 Soft Prompt Attacks
- Across most tasks, soft prompts do not significantly recover forgotten accuracy for any localization.
- Exception: For Gemma-2-9B on Sports-Athlete-Editing, soft prompts recover >60% forget accuracy for OT localizations, while FLU, Nonlocalized, and All-MLPs remain under 40%.

---

## 9. Implications for Understanding "Granular Deletes" in Transformers

This paper is directly relevant to the question of how to perform granular, targeted deletion of specific factual knowledge from transformer models. The key implications are:

### 9.1 Knowledge Has a Two-Stage Structure in Transformers
Factual knowledge is not stored in one place. There is a distinct **storage stage** (early-to-middle MLPs that enrich subject token representations with attribute information in the residual stream) and an **extraction stage** (later attention heads and MLPs that read that information and format it as output). Effective granular deletion must target the *storage* stage, not the *extraction* stage.

### 9.2 Probing the Residual Stream Is Essential for Locating What to Delete
The residual stream analysis (layer-by-layer probe accuracy) is the correct diagnostic for understanding *where* specific factual associations are encoded. The sharp probe accuracy increase between specific layers directly pinpoints the MLP layers performing fact lookup. Logit-based methods (causal tracing, attribution patching) are systematically biased toward finding the extraction stage.

### 9.3 Targeting Output Does Not Mean Targeting Knowledge
A key negative result: methods that successfully suppress the model's output of a fact (achieving zero forget accuracy on in-distribution prompts) do not necessarily remove the knowledge from internal representations. The knowledge can survive in the residual stream and be recovered by:
- Alternative prompt formats (MCQ)
- Paraphrased questions
- Short adversarial relearning (63% accuracy recovery in 20 steps for OT methods)
- Soft prompt optimization

### 9.4 Effective Granular Deletion Requires Disrupting the Latent Stream
The paper provides strong evidence that robust "granular deletion" requires modifying the weights that control what information enters the residual stream at the subject token, not just the weights that read from it. After FLU editing, probe classifiers on intermediate representations show the ground truth answer has been replaced by the edited answer at the source, whereas OT-edited models retain ground truth information in early layers.

### 9.5 Mechanistic Understanding Enables More Parameter-Efficient Deletion
FLU-localized edits are more parameter-efficient: fewer weight changes are needed to achieve robust generalization compared to OT or nonlocalized methods. This has practical implications for systems requiring targeted knowledge removal — mechanistic understanding of which specific components implement fact storage allows surgical, minimal edits.

### 9.6 Not All MLPs Are Equal: Location Within the Network Matters
A key finding is that the heuristic "edit all MLPs" does not achieve the same robustness as mechanistically localized MLP editing. The specific early-to-middle MLP layers (the FLU mechanism) are uniquely important for robust deletion. Random MLP selection performs similarly to nonlocalized editing on robustness metrics. Mechanistic knowledge of *which* MLPs store which facts is necessary.

### 9.7 The Standard Evaluation Protocol for Unlearning Is Insufficient
Standard unlearning evaluation (in-distribution prompt accuracy) is insufficient for assessing granular deletion quality. Robust evaluation must include: alternative prompt formats (MCQ), paraphrasing, adversarial relearning, and residual stream probing. Methods that pass standard evaluation can still fail completely on these harder tests. This has direct implications for evaluating granular delete systems.

### 9.8 Sequential Localization May Enable More Precise Per-Fact Deletion
The success of Sequential-CounterFact-Editing (editing 64 facts in sequential batches of 16, with potentially different localizations per batch) suggests that granular deletion may benefit from per-fact localization — different facts may reside in different specific components, and exploiting this enables more precise, less side-effect-prone deletions.

### 9.9 Residual Knowledge Is the Critical Failure Mode
The paper connects to Hong et al. (2024), which showed that current unlearning approaches fail to remove residual knowledge traces from internal activations, making them exploitable. The FLU approach is the first demonstrated non-oracle method that meaningfully disrupts internal residual knowledge (as shown by probing), though complete elimination remains challenging. This framing — residual knowledge in the residual stream as the key failure mode — is central to understanding what a truly "granular" delete must accomplish.

---

## Summary Table: Localization Methods for Granular Deletes

| Criterion | OT (Causal Tracing, Attr. Patching) | FLU (Mechanistic) | Nonlocalized |
|---|---|---|---|
| In-distribution forget accuracy | Good | Good | Good |
| MCQ generalization | Poor | **Best** | Moderate |
| Adversarial relearning resistance | Poor (63% recovery) | **Best (~20%)** | Moderate |
| Latent knowledge disruption | Poor (knowledge intact) | **Best** | Moderate |
| Parameter efficiency | Moderate | **Best** | Worst |
| Side effects (MMLU) | Moderate | Low | Moderate |
| Targets | Extraction stage (late layers) | Storage stage (early-mid layers) | All components |
