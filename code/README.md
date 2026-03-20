# Cloned Repositories

## 1. TransformerLens
- **URL**: https://github.com/TransformerLensOrg/TransformerLens
- **Location**: code/TransformerLens/
- **Purpose**: Primary library for mechanistic interpretability of GPT-style LMs. Provides hook-based access to all internal activations (residual stream, attention patterns, MLP outputs). Supports 50+ open-source models.
- **Key files**: `transformer_lens/HookedTransformer.py` (main model class), `transformer_lens/utils.py`
- **Install**: `pip install transformer-lens`
- **How to use**: Load model via `HookedTransformer.from_pretrained("gpt2")`, access residual stream via `model.run_with_cache()`, apply logit lens via unembedding intermediate states.
- **Notes**: This is the standard tool for logit lens experiments and residual stream analysis. Created by Neel Nanda.

## 2. tuned-lens
- **URL**: https://github.com/AlignmentResearch/tuned-lens
- **Location**: code/tuned-lens/
- **Purpose**: Official implementation of the tuned lens (Belrose et al., 2023). Trains per-layer affine translators for more reliable residual stream probing than the raw logit lens.
- **Key files**: `tuned_lens/nn/lenses.py`, `tuned_lens/scripts/`
- **Install**: `pip install tuned-lens`
- **How to use**: Train tuned lens on a model, then use it to decode intermediate hidden states into vocabulary distributions at each layer.
- **Notes**: Pre-trained lens checkpoints available for common models. Recommended over raw logit lens for quantitative analysis.

## 3. ROME (Rank-One Model Editing)
- **URL**: https://github.com/kmeng01/rome
- **Location**: code/rome/
- **Purpose**: Reference implementation for locating and editing factual associations in GPT. Includes causal tracing (activation patching) code and the CounterFact dataset.
- **Key files**: `rome/rome_main.py`, `experiments/causal_trace.py`, `notebooks/`
- **Install**: `pip install -r requirements.txt`
- **How to use**: Run causal tracing to identify where facts are stored, then apply ROME/MEMIT to edit them. Compare with logit lens analysis.
- **Notes**: Causal tracing is the standard baseline for knowledge localization. CounterFact dataset is auto-downloaded by the scripts.

## Additional Recommended Repositories (not cloned)

- **LogitLens4LLMs**: https://github.com/zhenyu-02/LogitLens4LLMs — extends logit lens to Llama-3.1, Qwen-2.5
- **nnsight**: https://github.com/ndif-team/nnsight — alternative to TransformerLens, supports remote inference
- **EasyEdit**: https://github.com/zjunlp/EasyEdit — unified knowledge editing framework (ROME, MEMIT, etc.)
- **OpenUnlearning**: https://github.com/locuslab/open-unlearning — 13 unlearning algorithms, 16 evaluations, pre-released checkpoints
- **linear_rep_geometry**: https://github.com/KihoPark/linear_rep_geometry — causal inner product implementation
