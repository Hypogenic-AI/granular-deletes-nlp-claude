# Characterizing Granular Deletes via Logit Lens and Residual Stream Geometry

Using the logit lens to map where factual information is stored, enriched, and deleted across layers of a language model, and analyzing the geometric structure of the residual stream before and after targeted knowledge removal.

## Key Findings

- **Factual information follows a non-monotonic trajectory**: It enters the residual stream at layers 15-21 and is partially *suppressed* in the final 2-3 layers. The logit lens directly reveals this natural "information deletion."
- **Early MLPs are the critical storage site**: MLP layer 0 alone accounts for ~100% of factual recall. This matches the Fact Lookup (FLU) mechanism from mechanistic unlearning literature.
- **Directional deletion is highly effective**: Projecting out a single "fact direction" from MLP outputs at layers 0-3 removes 99.7% of factual signal.
- **Deletion reshapes residual stream geometry**: Multi-layer deletion pushes final-layer activations into completely uncorrelated regions (cosine similarity ~ 0, L2 displacement > activation norm).
- **The residual stream is geometrically anisotropic**: A single principal component captures 63-91% of variance, peaking in middle layers.

## Reproduction

```bash
# Setup
uv venv && source .venv/bin/activate
uv add torch transformer-lens numpy pandas matplotlib seaborn scipy scikit-learn tqdm datasets

# Run experiments
python src/experiment.py          # GPT-2-small baseline
python src/experiment_enhanced.py # GPT-2-medium (primary results)
```

Requires a CUDA GPU (tested on NVIDIA RTX A6000). Total runtime: ~2 minutes.

## File Structure

```
├── REPORT.md                    # Full research report with results
├── planning.md                  # Research plan and methodology
├── src/
│   ├── experiment.py            # GPT-2-small experiments
│   └── experiment_enhanced.py   # GPT-2-medium experiments (primary)
├── results/data/                # Raw results (JSON, NPY)
├── figures/                     # All visualizations (PNG)
│   ├── fig1_logit_lens_comprehensive.png  # 4-panel logit lens analysis
│   ├── fig2_ablation_study.png            # MLP/attention ablation
│   ├── fig3_deletion_effects.png          # Pre/post deletion profiles
│   ├── fig4_enrichment_heatmap.png        # Per-fact enrichment
│   ├── fig5_geometry_pca.png              # PCA across layers
│   ├── fig6_cross_layer_similarity.png    # Cosine similarity matrix
│   └── fig7_norm_evolution.png            # Norm growth across layers
├── literature_review.md         # Background and related work
├── resources.md                 # Dataset and code catalog
├── datasets/                    # CounterFact and TOFU
├── code/                        # TransformerLens, tuned-lens, ROME
└── papers/                      # 16 downloaded research papers + notes
```

See [REPORT.md](REPORT.md) for full details.
