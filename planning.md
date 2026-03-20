# Research Plan: Characterizing Granular Deletes via Logit Lens and Residual Stream Geometry

## Motivation & Novelty Assessment

### Why This Research Matters
Understanding where and how factual information is stored and can be removed from language models is critical for AI safety (machine unlearning), privacy compliance, and mechanistic interpretability. The logit lens provides a window into the residual stream's evolving predictions, but no work has systematically mapped the geometry of information deletion across layers.

### Gap in Existing Work
Papers study logit lens (Belrose et al., Li et al. 2026), residual stream geometry (Park et al., Shai et al.), and deletion (Ghosh et al.) **separately**. No unified analysis tracks: (a) how logit lens signals change layer-by-layer when specific facts are present vs. ablated, (b) the geometric structure of fact-encoding directions in the residual stream, and (c) whether deletion interventions move activations across stable region boundaries.

### Our Novel Contribution
We provide the first unified analysis combining logit lens tracking with geometric characterization of the residual stream before and after targeted knowledge ablation. We map where facts "live" across layers, how they are geometrically organized, and what happens to the residual stream geometry when facts are surgically removed.

### Experiment Justification
- **Experiment 1 (Logit Lens Profiling)**: Establish baseline layer-by-layer information presence for factual knowledge — needed to know *where* to look for deletion effects.
- **Experiment 2 (Ablation + Logit Lens)**: Zero-ablate MLPs at different layers and measure how logit lens signals change — directly tests where information enters/leaves the residual stream.
- **Experiment 3 (Geometric Analysis)**: PCA and cosine similarity analysis of residual stream activations encoding different facts — maps the geometry of fact representations.
- **Experiment 4 (Deletion Geometry)**: Compare residual stream geometry before and after rank-one editing (ROME-style) — shows how deletion reshapes the geometric landscape.

## Research Question
Can the logit lens reveal where factual information is added to and deleted from the residual stream, and what is the geometric structure of fact representations across layers?

## Hypothesis Decomposition
1. **H1**: Factual information (correct object token probability) appears in logit lens outputs at specific layers (not uniformly) — we expect enrichment in early-to-mid MLP layers (FLU mechanism).
2. **H2**: Ablating MLPs at these enrichment layers causes the largest drop in correct-object probability at all subsequent layers.
3. **H3**: Fact-encoding directions in the residual stream are approximately linear and can be visualized via PCA.
4. **H4**: After targeted editing, the logit lens profile shifts — correct-object probability drops at storage layers, and the geometric structure of the residual stream changes measurably.

## Proposed Methodology

### Approach
Use TransformerLens to run GPT-2-small on CounterFact examples, applying logit lens at every layer. Perform MLP ablation studies and rank-one model editing to characterize information flow and deletion geometry.

### Experimental Steps
1. Load GPT-2-small via TransformerLens, load CounterFact dataset
2. For N=200 factual prompts, cache all residual stream activations
3. Apply logit lens (unembedding projection) at each layer, record probability of correct object token
4. Perform layer-wise MLP zero-ablation, measure effect on logit lens profiles
5. Compute PCA of residual stream activations at key layers
6. Apply rank-one editing (ROME-style) to delete specific facts
7. Re-run logit lens analysis post-editing
8. Compare pre/post geometry via cosine similarity and PCA

### Baselines
- Random token probability at each layer (chance baseline)
- Logit lens on unrelated prompts (control)
- Full model output as upper bound

### Evaluation Metrics
- **Layer-wise correct-object probability** (logit lens output)
- **Information enrichment score**: Δ probability from layer l to l+1
- **Ablation impact**: Drop in correct-object probability when layer l MLP is zeroed
- **Cosine similarity** between fact directions across layers
- **PCA variance explained** for fact representations
- **Pre/post editing probability shift** at each layer

### Statistical Analysis Plan
- Bootstrap confidence intervals (N=200 facts, 1000 resamples)
- Paired t-tests for pre/post editing comparisons
- Effect sizes (Cohen's d) for ablation impacts
- α = 0.05 with Bonferroni correction for multiple layer comparisons

## Expected Outcomes
- H1: Logit lens shows a sigmoid-like increase in correct-object probability across layers, with steepest increase at layers 4-8 (FLU region).
- H2: Ablating MLPs at layers 4-8 has the largest downstream effect.
- H3: Fact representations form interpretable clusters in PCA space.
- H4: Post-editing, correct-object probability drops primarily at storage layers.

## Timeline and Milestones
- Setup & data loading: 10 min
- Experiment 1 (logit lens profiling): 20 min
- Experiment 2 (ablation study): 30 min
- Experiment 3 (geometric analysis): 20 min
- Experiment 4 (deletion geometry): 30 min
- Analysis & visualization: 20 min
- Documentation: 20 min

## Potential Challenges
- GPU memory for caching all activations (mitigate: batch processing)
- ROME editing implementation complexity (mitigate: use simplified rank-one update)
- CounterFact tokenization issues (mitigate: filter to single-token objects)

## Success Criteria
- Clear layer-by-layer logit lens profiles showing where facts are enriched
- Statistically significant ablation effects identifying critical layers
- Interpretable geometric visualizations of the residual stream
- Measurable geometric changes after fact deletion
