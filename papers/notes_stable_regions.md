# Notes: Characterizing Stable Regions in the Residual Stream of LLMs

**Paper:** Ma et al. (2024), arXiv:2409.17113v4
**Venue:** Workshop on Scientific Methods for Understanding Deep Learning, NeurIPS 2024
**Authors:** Jett Janiak (LASR Labs), Jacek Karwowski (University of Oxford), Chatrik Singh Mangat (LASR Labs), Giorgi Giglemiani (LASR Labs), Nora Petrova (LASR Labs), Stefan Heimersheim (Apollo Research)

---

## 1. Key Contribution: What Are Stable Regions?

The paper identifies **stable regions** in the residual stream of Transformer language models. A stable region is a contiguous zone of the activation space (specifically the residual stream after the first layer) where:

- Small changes in the activation vector produce **minimal change** in the model's final output (next-token prediction distribution).
- At the **boundaries** between regions, even small perturbations cause sharp, significant changes in output.

This is formalized by studying how model output distance changes as activations are linearly interpolated between two prompt-derived points. When two prompts occupy the same region, the output distance changes smoothly and near-linearly across the interpolation. When they occupy different regions, the distance curve is flat near both endpoints (inside each respective region) and then jumps sharply at the boundary crossing.

The stable regions are **much larger** than the "polytopes" studied in prior piecewise-linear network analysis (e.g., Black et al. 2022, Hanin & Rolnick 2019). While a single interpolation between semantically different prompts typically crosses only one stable region boundary, hundreds or thousands of gate activation signs flip during the same interpolation — meaning many polytope boundaries are crossed inside what appears to be a single stable region boundary from the output's perspective.

---

## 2. Methodology for Identifying Stable Regions

### Core Setup

For a given prompt `p`, the paper defines a forward pass with activation patching as a function `F_p: R^D -> R^D` that returns the residual stream activations after the final layer at the last sequence position. `F_p(X)` replaces the residual stream activation after the **first** layer (at the last sequence position) with the input vector `X`, then runs the remaining layers normally.

### Interpolation Procedure

To probe stable regions, the authors interpolate between two prompt-derived activation vectors `A` and `B` (the residual stream after layer 1 for prompts `p_A` and `p_B`):

```
X(alpha) = A + alpha * (B - A),   alpha in [0, 1]
```

The output distance is then measured as:

```
d(alpha) = || F_{p_A}(A) - F_{p_A}(A + alpha*(B-A)) ||_2
```

This measures how much the model's output shifts relative to its clean run on `p_A` as the patched activation slides from `A` toward `B`. They compute `d` at 50 uniformly spaced values of alpha.

The **relative output distance** `d(alpha) / d(1)` is plotted to normalize across prompt pairs. The shape of this curve is the primary diagnostic:
- **Linear shape** → both prompts in the same stable region.
- **Flat-jump-flat (sigmoidal) shape** → prompts in different regions, with a sharp boundary crossing in the middle.

The **maximum slope** of the relative distance curve is used as a scalar summary of region boundary sharpness, enabling quantitative comparison across model sizes and training checkpoints.

### 2D Slice Visualization

To visualize the geometry directly, the authors project the residual stream activation space into 2D slices spanned by three model-generated activations A, B, C:

```
X = A + alpha*(B - A) + beta*(C - P)
```

where `P` is the orthogonal projection of C onto the line AB, and alpha, beta range from -0.25 to 1.25. For each point in this 2D grid, they compute how similar the model output is to the outputs produced by A, B, and C respectively, and encode this as an RGB color (red = similarity to A's output, green = B's, blue = C's). Solid-colored regions with sharp color transitions indicate distinct stable regions with clear boundaries.

### Polytope Comparison (Appendix F)

To directly compare stable region size to polytopes, the authors count how many gate activations (in the last layer of Qwen2-0.5B) **switch sign** between alpha=0 and alpha=1. Each sign flip represents a crossing of a polytope boundary. Even when only one stable region boundary is crossed in terms of output behavior, the histogram shows **hundreds to thousands** of gate sign changes — a direct empirical lower bound on how many polytopes are contained within a single stable region.

### Semantic Similarity Analysis (Appendix G)

The authors analyzed 1,000 prompt pairs filtered by their maximum derivative of the relative distance function. They found a clean quantitative split:
- Pairs with **low max derivative (< 1.2)**: consistently share the same last token (semantically similar).
- Pairs with **high max derivative (> 10)**: have different last tokens (semantically dissimilar).

---

## 3. Key Findings: How Stability Evolves Across Layers and Training

### Across Training (OLMo-1B and OLMo-7B)

- **Randomly initialized models** show near-linear `d(alpha)` curves — no stable regions exist at initialization.
- As training progresses (measured in tokens processed), the curves sharpen: flat plateaus emerge near each endpoint and the jump between them becomes more abrupt.
- For OLMo-1B, the sharpening effect **plateaus earlier** in training than for OLMo-7B, suggesting larger models continue refining their stable regions for longer.
- Checkpoints at 0B, 4B, 10B, 31B, 3050B tokens (OLMo-1B) and 0B, 10B, 100B, 501B, 2750B tokens (OLMo-7B) are examined.
- The 2D slice visualizations show that at early training, the 2D slice is a smooth gradient of colors. As training continues, distinct solid-colored blobs emerge and boundaries sharpen. Some slices hint at existing stable regions **splitting** during training (notably Figure 11), suggesting both refinement of existing regions and possible proliferation of new ones.

### Across Model Size

- Across both the OLMo (1B, 7B) and Qwen2 (0.5B, 1.5B, 7B) families, larger models produce sharper `d(alpha)` curves.
- The median maximum slope increases monotonically with parameter count across both families.
- The sharpening with scale mirrors the sharpening with training — both represent increased definition of stable region boundaries.
- A notable exception: **Gemma** does not show the same behavior.

### Across Layers (Appendix E)

The primary analysis patches after the **first** layer, but Appendix E presents a comparison with patching after the **seventh** layer in Qwen2-1.5B. Results patching at layer 7 are **smoother** but still show a significant jump around the midpoint of interpolation. This suggests that stable region structure is present (though less sharply defined) at intermediate layers too, and is not solely an artifact of the very first layer.

---

## 4. Connection to Information Storage and Deletion

The paper does not make an explicit, dedicated argument about information deletion, but the stable region framework is directly relevant:

- **Within a stable region**, the model's output is insensitive to activation perturbations. This means the model effectively "ignores" or "absorbs" many directions of variation within a region — activations in that zone all produce the same (or very similar) next-token predictions. From an information standpoint, fine-grained differences between activations within a region are **not read out** by the downstream layers.
- **At region boundaries**, the model is maximally sensitive. Small shifts in activation direction that cross a boundary cause a discrete change in the predicted output distribution. This is where the model "reads" semantic distinctions.
- This geometry implies a form of **lossy compression**: the Transformer maps the high-dimensional activation space into a small number of semantically-distinct stable output classes. Information that does not push an activation across a boundary is effectively discarded by the computation.
- The fact that stable regions **emerge during training** (not present at initialization) suggests that learning specifically sculpts this insensitive-interior / sensitive-boundary structure as part of the process of encoding knowledge. Before training, all directions are roughly equally informative; after training, only boundary-crossing directions carry output-relevant information.
- The connection to SAE (Sparse Autoencoder) work (cited from Gurnee 2024, Lindsey 2024, Lee & Heimersheim 2024) is noted: SAE reconstruction errors and the sensitive directions identified in earlier work are related phenomena — the directions that SAEs fail to reconstruct well may correspond to the directions that lie near stable region boundaries.

---

## 5. Models and Datasets Used

### Models

| Model        | Parameters | Layers | Hidden Size |
|--------------|-----------|--------|-------------|
| OLMo-1B      | 1B        | 16     | 2048        |
| OLMo-7B      | 7B        | 32     | 4096        |
| Qwen2-0.5B   | 0.5B      | 24     | 896         |
| Qwen2-1.5B   | 1.5B      | 28     | 1536        |
| Qwen2-7B     | 7B        | 28     | 3584        |

OLMo models use non-parametric layer norm and have vocabulary size 50,304. Qwen2 models use RMSNorm and have vocabulary size 151,936. The authors note qualitatively similar results for GPT-2, Pythia, Phi, and Llama families; Gemma is a noted exception where the effect does not appear.

OLMo checkpoints at multiple training stages (0B through 2750B tokens) are used to study training dynamics.

### Dataset

- **sedthh/gutenberg_english** (Project Gutenberg English books dataset on HuggingFace): Used to sample 1,000 pairs of 10-token-long, unrelated prompts for the large-scale model-size and training-progress experiments. This dataset was chosen because it is likely present in most Transformer LM training corpora, has limited diversity (books), and produces easily interpretable next-token predictions.
- **Manually constructed prompt pairs** (D1-D3 dissimilar, S1-S3 similar): Used for illustrative qualitative examples in the main experiments.

---

## 6. Implications for Residual Stream Geometry

### The Residual Stream as a Partitioned Space

The findings suggest that the residual stream is not a featureless high-dimensional vector space but is instead organized into a **partition of semantically-coherent basins** separated by sharp boundaries. Within each basin, the rest of the network (layers 2 through final) implements essentially the same function (up to small perturbations), yielding the same output class.

### Relation to the Hypersphere

The linear representation hypothesis and the presence of normalization layers in Transformers (LayerNorm for OLMo, RMSNorm for Qwen2) place model-generated activations near a hypersphere in the residual stream. Because of the high dimensionality, linear interpolations between two points on (or near) the hypersphere stay close to the hypersphere's surface rather than cutting through its interior. The stable regions therefore partition this hypersphere surface, not the full volume of the space.

### Stable Regions vs. Polytopes

Prior work on polytopes in piecewise-linear networks defines regions by the sign pattern of all gating activations. The number of such polytopes grows exponentially with depth and width, potentially reaching astronomical numbers. The paper's key geometric finding is that **stable regions are coarser than polytopes by a very large factor**: hundreds or thousands of polytope boundaries can be crossed while remaining inside a single stable region. This means the network's output function is far smoother than the sign-pattern decomposition would predict — many gate sign changes do not translate into output changes.

### Region Splitting During Training

Several 2D slice visualizations (particularly Figure 11) hint that as training progresses, a single large stable region can **split into two or more** sub-regions with different output classes. This dynamic suggests that the network progressively learns finer semantic distinctions over the course of training, carving up what was previously a single output basin into more refined categories.

### Semantic Alignment

The stable regions align with semantic content: prompts that share the same top predicted token (semantically similar) tend to cluster within the same stable region, while prompts that predict different tokens (semantically different) occupy different regions. The maximum derivative of the interpolation curve cleanly separates pairs by whether they share a last token, providing quantitative validation of the semantic interpretation.

---

## 7. Connection to Logit Lens and Mechanistic Interpretability

### Logit Lens Connection

The paper does not explicitly cite the logit lens, but the methodology is deeply related. The logit lens (Nostalgebraist 2020) applies the unembedding matrix directly to intermediate residual stream activations to inspect what token is being "predicted" at each layer. The present paper's analysis of stable regions is effectively asking: if we hold the residual stream at layer 1 and vary the activation, at what point does the predicted token (as computed by all subsequent layers) change? This is a probing of the residual stream's output-class structure that complements what the logit lens does layer-by-layer. Stable regions can be thought of as the "input-space" counterpart to the logit lens's "layer-space" view: instead of asking how predictions evolve across layers for a fixed input, the authors ask how predictions change across the input space for a fixed layer handoff point.

### Mechanistic Interpretability

The paper explicitly positions itself within the mechanistic interpretability research program and connects to several key threads:

1. **Linear representation hypothesis**: The authors build on the idea (from Elhage et al. 2022 / Toy Models of Superposition, Olah 2024) that features correspond to directions in activation space. Stable regions can be understood as the output-level consequence of this directional structure — directions that do not cross a region boundary are not "decoded" into different predictions by the model.

2. **SAE (Sparse Autoencoder) connection**: The paper cites work by Gurnee (2024) on pathological SAE reconstruction errors and Lindsey (2024) on how strongly dictionary learning features influence behavior, and follow-up work by Giglemiani et al. (2024) and Lee & Heimersheim (2024) connecting interpolation sensitivity to SAE features. The suggestion is that the directions near stable region boundaries correspond to the features that SAEs identify as most influential, and that SAE reconstruction errors occur in directions orthogonal to boundaries (i.e., within stable regions, where errors don't change outputs).

3. **Activation patching**: The interpolation methodology is a form of activation patching — a core mechanistic interpretability technique. The paper generalizes single-point patching to a continuous interpolation, enabling characterization of the geometry around patched activations.

4. **Training dynamics interpretability**: By studying how stable regions emerge with training checkpoints, the paper opens a window into what the training process is actually learning at a geometric level. The emergence of sharp boundaries during training suggests that the network progressively learns to be robust to within-region variation while becoming increasingly sensitive to between-region variation — a form of representation learning that organizes semantic categories in the residual stream.

5. **Complexity characterization**: The work connects mechanistic interpretability to the broader project of characterizing the complexity of neural networks (VC dimension, polytope counting), arguing that stable regions are a more practically relevant unit of complexity than polytopes for understanding how trained networks actually behave.

---

## Summary Assessment

This paper makes a clean, empirically grounded contribution: stable regions are a real and characterizable feature of trained Transformer residual streams, they align with semantic content, and they emerge from training. The key insight — that stable regions are much coarser than polytopes, meaning the network's effective decision space is far simpler than worst-case complexity bounds would suggest — has direct implications for interpretability. For granular deletion / machine unlearning research, the stable region framework suggests that **erasing information requires moving activations across region boundaries**, not merely perturbing them within a region. Perturbations that stay within a stable region will be "absorbed" by the network and leave predictions unchanged, which is both a challenge for deletion (you need boundary-crossing edits) and a potential insight (knowing where boundaries are tells you the minimum edit needed to change a prediction).
