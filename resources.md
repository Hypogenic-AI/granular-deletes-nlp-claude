# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project "Characterizing Granular Deletes" — investigating whether logit lens techniques can identify where information is deleted from the residual stream in language models and map the geometry of the residual stream.

---

## Papers
Total papers downloaded: 16

| # | Title | Authors | Year | File | Key Info |
|---|-------|---------|------|------|----------|
| 1 | Tuned Lens | Belrose et al. | 2023 | papers/2303.08112_belrose2023_tuned_lens.pdf | Core method: per-layer affine probes for residual stream |
| 2 | Attention Lens | Todd et al. | 2023 | papers/2310.16270_todd2023_attention_lens.pdf | Logit lens extended to attention heads |
| 3 | Entropy-Lens | Li et al. | 2025 | papers/2502.16570_li2025_entropy_lens.pdf | Entropy-based residual stream analysis |
| 4 | LogitLens4LLMs | Kim et al. | 2025 | papers/2503.11667_kim2025_logitlens4llms.pdf | Logit lens for modern LLMs |
| 5 | Belief State Geometry | Shai et al. | 2024 | papers/2405.15943_li2024_belief_state_geometry.pdf | Fractal belief geometry in residual stream |
| 6 | Constrained Belief Updates | Guo et al. | 2025 | papers/2502.01954_guo2025_constrained_belief.pdf | Geometric structures from belief updating |
| 7 | Linear Representation Hypothesis | Park et al. | 2023 | papers/2311.03658_park2023_linear_representation.pdf | Causal inner product for concept geometry |
| 8 | Neuron-Level Knowledge Attribution | Yu & Ananiadou | 2023 | papers/2312.12141_yu2023_exploring_residual.pdf | ~300 neurons capture 97-99% of knowledge signal |
| 9 | Not All Features Linear | Engels et al. | 2024 | papers/2405.14860_engels2024_not_all_linear.pdf | Multi-dimensional features |
| 10 | Stable Regions | Janiak et al. | 2024 | papers/2409.17113_ma2024_stable_regions.pdf | Stable basins in residual stream geometry |
| 11 | Secretly Linear | Gromov et al. | 2024 | papers/2405.12250_gromov2024_secretly_linear.pdf | Near-linear layer-to-layer mappings |
| 12 | Mechanistic Unlearning | Ghosh et al. | 2024 | papers/2410.12949_ghosh2024_mechanistic_unlearning.pdf | FLU-based deletion: storage vs extraction |
| 13 | Layer-Targeted Erasure | Li et al. | 2026 | papers/2602.22562_wang2026_layer_targeted_erasure.pdf | Logit Lens verifies genuine erasure |
| 14 | Erasing Without Remembering | Chen et al. | 2025 | papers/2502.19982_chen2025_erasing_without_remembering.pdf | Residual knowledge in intermediate layers |
| 15 | Practical Review MI | Rai et al. | 2024 | papers/2407.02646_rai2024_practical_review_mi.pdf | Comprehensive MI survey |
| 16 | How Many Features | He et al. | 2026 | papers/2602.11246_he2026_how_many_features.pdf | Capacity bounds under LRH |

Deep reading notes available for 7 papers in papers/notes_*.md.

See papers/README.md for detailed descriptions.

---

## Datasets
Total datasets downloaded: 2

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| CounterFact | HuggingFace (azhx/counterfact) | 21,919 examples | Factual knowledge editing | datasets/counterfact/ | Standard benchmark; (subject, relation, object) triples |
| TOFU | HuggingFace (locuslab/TOFU) | 4,000 QA pairs | Machine unlearning | datasets/tofu/ | Synthetic author profiles; controlled deletion |

See datasets/README.md for download instructions and detailed descriptions.

---

## Code Repositories
Total repositories cloned: 3

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| TransformerLens | github.com/TransformerLensOrg/TransformerLens | MI library: activation access, hooks | code/TransformerLens/ | Primary tool for logit lens experiments |
| tuned-lens | github.com/AlignmentResearch/tuned-lens | Tuned lens implementation | code/tuned-lens/ | Per-layer affine probes |
| ROME | github.com/kmeng01/rome | Knowledge editing + causal tracing | code/rome/ | Baseline localization method |

See code/README.md for detailed descriptions.

---

## Resource Gathering Notes

### Search Strategy
1. Used paper-finder service (fallback to web search due to service unavailability)
2. Searched arXiv, Semantic Scholar, Papers with Code via web search
3. Keyword clusters: "logit lens", "tuned lens", "residual stream geometry", "mechanistic interpretability", "knowledge erasure/unlearning", "linear representation hypothesis", "stable regions"
4. Downloaded papers via arXiv PDF API
5. Deep-read 7 key papers using PDF chunker (all chunks)

### Selection Criteria
- Papers directly studying the residual stream with logit lens or probing methods
- Papers characterizing the geometry of transformer representations
- Papers on granular/targeted knowledge deletion with mechanistic analysis
- Prioritized papers with code availability and established datasets

### Challenges Encountered
- Paper-finder service was unavailable (httpx dependency); resolved by manual web search
- arXiv paper 2312.12141 had a different title than expected ("Neuron-Level Knowledge Attribution" vs. "Exploring the Residual Stream") — noted in reading notes

### Gaps and Workarounds
- No paper directly combines all three threads (logit lens + geometry + deletion) — this is the research gap
- Belief state geometry work is limited to small models — production-scale validation needed
- Stable region analysis has not been connected to deletion interventions

---

## Recommendations for Experiment Design

### Primary dataset(s)
1. **CounterFact** — tight factual triples, counterfactual pairs, compatible with both ROME causal tracing and logit lens analysis
2. **TOFU** — controlled unlearning benchmark with pre-trained checkpoints available

### Baseline methods
1. **Logit Lens** (raw unembedding projection) — zero-parameter baseline
2. **Tuned Lens** (Belrose et al.) — trained affine probes per layer
3. **Causal Tracing** (ROME) — output-tracing localization
4. **FLU Localization** (Mechanistic Unlearning) — mechanistic localization

### Evaluation metrics
1. **Logit Lens recall at each layer** — track information presence/absence through the residual stream
2. **Linear probe accuracy per layer** — detect latent knowledge after deletion
3. **Forget/Retain accuracy** — standard unlearning evaluation
4. **MCQ generalization** — robustness to prompt format changes
5. **Adversarial relearning** — test whether deletion is genuine

### Code to adapt/reuse
1. **TransformerLens** — activation caching and logit lens computation
2. **tuned-lens** — trained per-layer probes for more reliable analysis
3. **ROME** — causal tracing implementation for comparison
