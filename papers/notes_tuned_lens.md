# Notes: Eliciting Latent Predictions from Transformers with the Tuned Lens

**Citation:** Belrose, N., Ostrovsky, I., McKinney, L., Furman, Z., Smith, L., Halawi, D., Biderman, S., & Steinhardt, J. (2023). Eliciting Latent Predictions from Transformers with the Tuned Lens. arXiv:2303.08112v6.

**Affiliations:** EleutherAI, FAR AI, University of Toronto, Boston University, UC Berkeley.

**Code:** https://github.com/AlignmentResearch/tuned-lens

---

## 1. Research Question and Key Contribution

**Research question:** How do transformer language model predictions evolve layer by layer, and can we reliably decode the "latent prediction" at each intermediate layer?

**Framing:** The authors view each transformer layer as performing an incremental update to a latent prediction of the next token — the *iterative inference* perspective. Each residual update moves the hidden state closer to the final answer, much like gradient descent.

**Key contribution:** The *tuned lens* — a set of learned affine probes (one per transformer block) that decode intermediate hidden states into vocabulary distributions. This is a drop-in improvement over the earlier *logit lens* (nostalgebraist, 2020), which was unreliable and biased. The tuned lens is:
- More predictive (lower perplexity) than the logit lens
- Less biased relative to the model's final output distribution
- Applicable to models where the logit lens completely fails (BLOOM, OPT, GPT-Neo)
- Causally faithful: its important features are also important to the actual model

Secondary contributions:
- **Causal basis extraction (CBE):** a novel algorithm for finding the most causally influential directions in the residual stream
- Applications: prompt injection detection, example difficulty estimation, extending the "Overthinking the Truth" results to new models

---

## 2. How the Logit Lens Works (Original Method)

The logit lens (nostalgebraist, 2020) decodes intermediate hidden states directly using the model's pretrained unembedding matrix, skipping all remaining layers.

**Formal definition:** For a pre-LayerNorm transformer with hidden state h_ℓ at layer ℓ, the logit lens sets all remaining residual updates to zero:

```
LogitLens(h_ℓ) = LayerNorm[h_ℓ] · W_U
```

where W_U is the unembedding matrix. This is equivalent to assuming the model has already finished computing and the current hidden state can be directly projected to vocabulary space.

**The residual stream view:** The full transformer output can be written as:
```
M_{>ℓ}(h_ℓ) = LayerNorm[h_ℓ + Σ_{ℓ'=ℓ}^{L} F_{ℓ'}(h_{ℓ'})] · W_U
```
The logit lens approximates this by dropping the summed residual updates entirely.

**Problems with the logit lens:**

1. **Unreliability:** The logit lens fails entirely on BLOOM, OPT 125M, and GPT-Neo. For these models, the top-1 prediction is often the *input* token itself (rather than a plausible continuation) in more than half of layers. The "extended" logit lens variant (which retains the last transformer layer) is only partially successful.

2. **Bias:** The logit lens is a *biased* estimator of the final output distribution. It systematically puts more probability mass on certain vocabulary items than the final layer does. For GPT-Neo-2.7B, this bias is 4–5 bits (KL divergence) for most layers — far larger than the 0.0068-bit gap between Pythia 160M and Pythia 12B. This means the logit lens prediction trajectory cannot be interpreted as rational belief updating: a "Dutch book" could be constructed against it.

3. **Representation drift:** Transformer hidden states contain "rogue dimensions" (Timkey & van Schijndel, 2021) — a small number of very high-variance dimensions distributed unevenly across layers. Even after controlling for rogue dimensions, covariance matrices of hidden states drift apart with layer depth (Figure 6). The final layer's covariance often shifts sharply from previous layers, so the logit lens may "misinterpret" earlier representations.

---

## 3. How the Tuned Lens Improves on the Logit Lens

The tuned lens addresses both the bias and representation drift problems by learning a separate affine "translator" for each layer.

**Step 1 — Debiasing:** Replace the zero residual assumption with a learnable constant bias b_ℓ:
```
LogitLens_debiased(h_ℓ) = LayerNorm[h_ℓ + b_ℓ] · W_U
```

**Step 2 — Correcting for representation drift:** Add a learnable change-of-basis matrix A_ℓ that maps from the representation space at layer ℓ to the space expected at the final layer:
```
TunedLens_ℓ(h_ℓ) = LayerNorm[A_ℓ · h_ℓ + b_ℓ] · W_U
```

The pair (A_ℓ, b_ℓ) is called the *translator* for layer ℓ. There are L translators total (one per layer).

**Training objective (distillation loss):** Translators are trained to minimize KL divergence between the tuned lens output and the final layer output:
```
argmin_θ E_x [ D_KL( f_{>ℓ}(h_ℓ) || TunedLens_ℓ(h_ℓ) ) ]
```
The final layer distribution acts as a soft label. This prevents the probe from learning to extract information *beyond* what the model has learned — a known problem when training probes with ground truth labels (Hewitt & Liang, 2019).

**Key design choice:** Unlike classical early-exit probes (Alain & Bengio, 2016), the tuned lens does *not* learn a new unembedding matrix per layer. It only learns A_ℓ (d × d) and b_ℓ (d), reducing the parameter count from |V| × d to d × d per layer (|V| ranges from 50K to 250K depending on model).

**Relation to model stitching:** The tuned lens is equivalent to "stitching" an intermediate layer directly to the unembedding with an affine alignment in between — the same concept as model stitching (Lenc & Vedaldi, 2015; Bansal et al., 2021), here applied within a single model across its own layers.

**Training details:**
- Trained on validation slices from pretraining data (Pile validation set for BLOOM and GPT-2)
- Documents chunked into 2048-token sequences
- Evaluated on 16.4M token random samples from respective validation sets
- Initial optimizer: SGD with Nesterov momentum, linear LR decay over 250 steps, base LR 1.0 (or 0.25 with final layer), gradient clipping at norm 1, batch size 2^18 tokens
- Translators initialized to identity
- Weight decay: 1e-3
- **Muon optimizer (post-publication update):** Using Muon (Jordan et al., 2024) instead of SGD dramatically accelerates training and achieves much lower KL divergence. Muon-trained lenses have much larger Frobenius norms, indicating they are much further from the logit lens. The authors note that all lenses in the original paper were "severely undertrained" and recommend Muon for future work.

---

## 4. Key Findings About How Predictions Evolve Across Layers

**Smooth convergence:** The prediction trajectory (the sequence of tuned lens distributions across layers) exhibits a strong tendency to converge smoothly to the final output distribution, with each successive layer achieving lower perplexity. This supports the iterative inference view.

**Qualitative examples (Appendix B):**
- *"A Tale of Two Cities" (Pythia 12B, Figure 15):* The model becomes very confident at early layers after processing the prefix "It was the best of times," suggesting memorization. The predictions lock onto the correct continuation early.
- *GPT-3 abstract (Pythia 12B, Figure 16):* The model gradually builds up the correct tokens, with early layers showing vague semantic approximations that sharpen by later layers.
- *Vaswani et al. abstract (Figure 1):* The logit lens shows unintelligible predictions until layer 21; the tuned lens shows meaningful predictions from early layers.

**Transferability across layers:** Translators trained for one layer can often zero-shot transfer to nearby layers with only a modest increase in perplexity ("transfer penalty"). Transfer penalties are strongly negatively correlated with covariance similarity between layers (Spearman ρ = −0.78), confirming that representation drift is the key barrier.

**Transfer is asymmetric:** Transfer penalties are higher when training on a layer with outlier dimensions (layer 5 and beyond in Pythia 12B) and testing on a layer without them, than the reverse.

**Transfer to fine-tuned models:** Lenses trained on a base model transfer well to fine-tuned versions. Specifically, a lens trained on LLaMA 13B transferred to Vicuna 13B with at most 0.3 bits/byte increase in KL divergence on RedPajama, and no significant difference on Anthropic HH-RLHF data. Fine-tuning minimally affects the representations used by the tuned lens.

**Causal fidelity — features used by tuned lens match those used by model:**
- Using causal basis extraction (CBE), the authors extract the most influential directions for the tuned lens at each layer.
- When these same directions are ablated in the actual model's hidden states, there is a strong correlation between the influence on the tuned lens and the influence on the model output (Spearman ρ = 0.89 at layer 18 of Pythia 410M; ρ ranges from 0.25 at the embedding layer to 0.98 at the final layer — Figure 20).
- No features are found that are influential for the tuned lens but not for the model, confirming the tuned lens is not relying on spurious features.

**Stimulus-response alignment:** When the same directions are ablated, the tuned lens's output change (stimulus) is aligned with the model's output change (response) in Aitchison geometry. Alignment increases with depth and is higher for the tuned lens than the logit lens at all layers (Figure 9).

**Iterative inference — theoretical and empirical support (Appendix C):**
- By Taylor expanding the loss around intermediate hidden states, the model is theoretically encouraged to align each residual with the negative gradient (gradient-residual alignment).
- Empirically on Pythia 6.9B: at every layer, the cosine similarity between the residual F(h_i) and the gradient ∂L/∂h_i is negative at least 95% of the time, and of much larger magnitude than expected by chance in this high-dimensional space (hidden_size × seq_length ≈ 8.4M dimensions).
- **Layer deletion robustness:** Deleting any single layer from Pythia 6.9B (replacing it with identity) causes only a nearly imperceptible increase in perplexity — *except* for the very first layer, which is crucial. This mirrors findings in ResNets (Veit et al., 2016) and further supports the iterative inference view.

---

## 5. Datasets and Models Used

**Models evaluated:**
- **Pythia suite** (EleutherAI): 70M, 160M, 410M, 1.4B, 2.8B, 6.9B, 12B (deduped and non-deduped)
- **GPT-NeoX-20B** (EleutherAI)
- **GPT-2** (OpenAI): small, medium, large, XL
- **GPT-Neo** (EleutherAI): 125M, 1.3B, 2.7B
- **BLOOM 560M** (BigScience)
- **OPT** (Meta): 125M, 1.3B, 6.7B (OPT 350M omitted — uses post-LN architecture)
- **LLaMA 13B** and **Vicuna 13B** (for transfer experiments)

**Datasets:**
- **The Pile** (Gao et al., 2020): 800GB diverse text; used for training and evaluation (validation set for Pythia, GPT-NeoX)
- **RedPajama** (Together, 2023): Open-source replication of LLaMA training data; used for LLaMA/Vicuna experiments
- **Anthropic HH-RLHF** (Bai et al., 2022): Used to evaluate lens transfer to Vicuna
- BLOOM and GPT-2 do not have public validation sets; Pile validation set used for both

**Downstream tasks (for applications):**
- ARC-Easy, ARC-Challenge (commonsense QA)
- BoolQ (boolean QA)
- MC TACO (temporal commonsense)
- MNLI, QNLI, QQP, RTE (NLI/entailment)
- SciQ (science QA)
- SST-2 (sentiment)
- WinoGrande (commonsense)
- LogiQA (logical reasoning)
- PiQA (physical commonsense)
- SICK (Sentences Involving Compositional Knowledge) — for Halawi et al. replication

---

## 6. Evaluation Metrics

**Primary metric — perplexity (bits per byte):** Measures how well the tuned/logit lens distribution matches the true next-token distribution. Lower is better.

**Bias metric — KL divergence of marginals:** Measures how much the lens's marginal token distribution (averaged over all positions in a dataset) diverges from the model's true marginal distribution. Formally: D_KL(p || q_ℓ) where p and q_ℓ are the marginal distributions of the final layer and lens at layer ℓ.

**Transfer penalty:** Expected increase in cross-entropy loss when evaluating a translator trained for layer ℓ on a different layer ℓ'.

**Causal influence (bits):** Expected KL divergence between model outputs before and after ablating a direction via mean ablation: E[D_KL(f(h) || f(r(h, v)))].

**Stimulus-response alignment (Aitchison similarity):** Cosine similarity between the lens's response to an intervention (stimulus) and the model's response to the same intervention, measured in Aitchison geometry (turns probability simplex into inner-product space).

**AUROC:** For prompt injection detection, area under the ROC curve across 10 random train-test splits.

**Spearman rank correlation:** Used to correlate prediction depth (tuned lens) with iteration learned (training difficulty); also used to assess causal fidelity.

**Automated interpretability score:** Pairwise cosine similarity of BPEmb word embeddings of top-k predicted tokens — measures "monosemanticity" of token lists produced by static analysis methods.

**Median-calibrated accuracy:** Used in the Halawi et al. replication for few-shot task performance.

---

## 7. Key Results and Figures

**Figure 1 (main comparison):** On GPT-Neo-2.7B processing the Vaswani et al. abstract, the logit lens produces unintelligible tokens before layer 21; the tuned lens produces meaningful predictions from early layers.

**Figure 3 (bias):** For GPT-Neo-2.7B, the logit lens bias is 4–5 bits KL for most layers; the tuned lens bias is near-zero throughout, confirming it as an unbiased estimator.

**Figure 4 (BLOOM perplexity):** The tuned lens achieves substantially lower perplexity than the logit lens for BLOOM 560M, whether or not the final transformer layer is included. The tuned lens and logit lens are complementary rather than the tuned lens simply inheriting a good final layer.

**Figure 5 (Pythia/GPT-NeoX-20B perplexity):** Tuned lens predictions are uniformly lower perplexity across all model sizes (70M to 20B) and all layers, and exhibit lower variance across independently trained models of the same size class.

**Figure 6 (covariance similarity):** Pairwise Frobenius cosine similarity of hidden state covariance matrices across layers of Pythia 12B. Layer 4 introduces two outlier ("rogue") dimensions that dominate the covariance. Removing them reveals smooth representational drift with depth.

**Figure 7 (transfer penalties):** Low off-diagonal values in the transfer penalty matrix for Pythia 12B confirm that nearby layers share similar representations. Transfer penalties are strongly negatively correlated with covariance similarity (Spearman ρ = −0.78).

**Figure 8 (causal fidelity at layer 18):** Spearman ρ = 0.89 between CBE feature influence on tuned lens and feature influence on model for Pythia 410M. No features appear in lower-right quadrant (influential for lens but not model).

**Figure 9 (stimulus-response alignment):** At all layers of Pythia 160M, tuned lens alignment is higher than logit lens alignment; both increase with depth.

**Figure 11 (Overthinking the Truth replication):** The tuned lens extends Halawi et al.'s results to BLOOM and GPT-Neo (which the logit lens cannot handle). Under incorrect few-shot demonstrations, the best layer's calibrated performance peaks at ~0.4–0.45 before falling to chance at the final layer.

**Table 1 (prompt injection detection):** On Pythia 12B, tuned lens + LOF achieves perfect or near-perfect AUROC on BoolQ (1.00), MNLI (1.00), QNLI (1.00), QQP (1.00), SST-2 (1.00). The logit lens is substantially worse on most tasks. The SRM baseline is competitive overall; combining both approaches is suggested for future work.

**Table 2 (example difficulty):** Positive Spearman correlation between prediction depth (tuned lens) and iteration learned across all 11 tasks. Tuned lens outperforms logit lens on 8 of 11 tasks, sometimes dramatically (QQP: tuned ρ=0.585 vs logit ρ=−0.340; QNLI: tuned ρ=0.409 vs logit ρ=−0.099).

**Figure 12 (token-level prediction depth):** Visualization on GPT-4's technical report abstract via Pythia 12B. Function words and predictable tokens have low prediction depth (cool colors); content words and surprising tokens have high depth (warm colors).

**Figure 13 (LLaMA→Vicuna transfer):** Transferred lens from LLaMA 13B closely tracks a lens specifically trained on Vicuna 13B; both dramatically outperform the logit lens.

**Figure 14 (GPT-2, GPT-Neo, OPT perplexities):** Consistent results across all model families: tuned lens always achieves lower perplexity. OPT's logit lens has very high perplexity (sometimes exceeding 10 bits), making it essentially unusable.

**Figure 15 (A Tale of Two Cities):** After processing "It was the best of times," the model's predictions at early layers rapidly converge to the correct continuation ("worst of times"), suggesting memorization.

**Figure 17 (BLOOM logit lens pathology):** Logit lens for BLOOM 560M assigns high probability to the *input* token at many layers — a clear failure mode.

**Figure 19 (iterative inference evidence):** Left: cosine similarity between residuals and loss gradients in Pythia 6.9B — consistently negative at all layers, far below chance. Right: layer deletion experiment — only removing layer 1 causes meaningful perplexity increase.

**Figure 20 (causal fidelity across all layers):** Spearman ρ increases from 0.25 (embeddings) to 0.98 (final layer) for Pythia 410M, showing the lens becomes increasingly faithful to the model as depth increases.

**Table 3 (static interpretability):** For Pythia 125M, tuned lens consistently improves interpretability scores over logit lens for all weight matrix types (OV, QK, Win, Wout SVDs and column vectors). Improvement is less clear for larger models.

**Table 4 (toxicity reduction):** For OPT-125m, tuned lens outperforms logit lens on toxicity reduction via value vector editing (overall toxicity reduced from 0.50 to 0.39 with Tuned Lens Top-20 vs 0.43 for Logit Lens Top-20) with no significant perplexity increase. Results did not generalize to other models tested.

---

## 8. Limitations

1. **Training required:** Unlike the logit lens (which works on any pretrained model out-of-the-box), the tuned lens requires training a set of translator probes. However, training is fast: under one hour on a single 8×A40 node. Pre-trained tuned lens checkpoints are released for commonly used models.

2. **Undertrained lenses:** The authors acknowledge that all lenses in the original paper were "severely undertrained" (under SGD). Using the Muon optimizer achieves much better results; the Muon-trained lenses are much further from the identity than the original SGD-trained ones. All original quantitative results should be taken as lower bounds on what is achievable.

3. **Causal basis extraction is computationally intensive:** CBE sequentially optimizes d_model causal basis vectors per layer, making it expensive. Future work could optimize entire k-dimensional subspaces simultaneously.

4. **Language models only (in this work):** The paper focuses exclusively on autoregressive LMs. The authors note the approach is likely applicable to other modalities (vision, etc.) but this is left to future work.

5. **Scope of prompt injection detection:** The SRM baseline is competitive and sometimes superior, particularly for MC TACO and SciQ. The tuned lens approach and SRM appear to be complementary (one uses layer trajectory, the other uses high-dimensional final-layer representations).

6. **Static interpretability gains are inconsistent:** Improvements from the tuned lens over the logit lens in static weight matrix interpretability are much less clear for larger models, where both methods perform poorly.

7. **Limited to pre-LayerNorm architectures:** The method is designed primarily for pre-LN transformers (the dominant architecture in state-of-the-art models). Post-LN models (e.g., OPT 350M) are excluded.

8. **Toxicity editing results don't generalize:** The improved toxicity reduction using the tuned lens worked for OPT-125m but not for pythia-125m, pythia-350m, gpt-2-medium, or gpt-neo-125m.

---

## 9. Residual Stream Geometry and Information Flow

This paper contains substantial material relevant to residual stream geometry and information flow through transformer layers.

### Representation Drift (Figure 6)
- The covariance matrices of hidden states at different layers drift apart smoothly with depth, as measured by Frobenius cosine similarity between pairwise covariance matrices.
- Layer 4 of Pythia 12B introduces two "rogue" (outlier) dimensions that dominate the covariance structure and are distributed unevenly across layers; removing them reveals the underlying smooth drift.
- The covariance at the final layer often shifts sharply relative to previous layers, which is why the logit lens misinterprets earlier representations.
- Transfer penalties between tuned lens translators are strongly negatively correlated with covariance similarity (Spearman ρ = −0.78), quantitatively linking representational geometry to the cost of cross-layer translation.

### Rogue Dimensions
- A small number of very high-variance dimensions are present in transformer hidden states (following Timkey & van Schijndel, 2021).
- These outlier dimensions are distributed unevenly across layers and can dominate the covariance.
- Ablating an outlier direction can drastically harm model performance (Kovaleva et al., 2021).
- The presence/absence of rogue dimensions creates an asymmetry in transfer: it is harder to transfer from a layer *with* rogue dimensions to one *without* them, than the reverse.

### Gradient-Residual Alignment (Appendix C, Figure 19)
- The theoretical analysis (following Jastrzebski et al., 2017) shows that residual connections encourage each layer's output to be anti-aligned with the loss gradient — i.e., each layer performs a step toward lower loss.
- Empirically in Pythia 6.9B: the cosine similarity between the residual update F(h_i) and the gradient ∂L/∂h_i is consistently negative (≤ −0.01) at all layers, and far below what would be expected for random vectors (random 5th percentile ≈ −6×10^{-4}).
- This cosine similarity is never large in absolute terms (max ≈ −0.05), consistent with many small gradient steps.

### Layer Deletion Robustness
- Replacing any single layer with the identity function in Pythia 6.9B causes only a negligible increase in perplexity, except for layer 1.
- This robustness suggests that adjacent layers encode fundamentally similar representations (Greff et al., 2016), implying a smooth, incremental structure to the residual stream.
- The special role of the first layer is consistent with ResNet findings (Veit et al., 2016), suggesting this is a general property of residual networks.

### Information Flow — Prediction Trajectory Properties
- The prediction trajectory (sequence of tuned lens distributions) converges smoothly and monotonically to the final output, with each successive layer achieving lower perplexity.
- The "prediction depth" of a token (the layer at which the top-1 prediction stabilizes) correlates positively with training difficulty ("iteration learned") across tasks — harder examples require more layers to classify.
- Tokens with high prediction depth tend to be semantically surprising or content-bearing; tokens with low depth tend to be function words or high-probability continuations.
- In the few-shot setting, earlier layers are *more robust* to incorrect demonstrations than later layers — suggesting that contextual information from demonstrations is integrated progressively, and the model's "prior knowledge" is more accessible at earlier layers.

### Affine Sufficiency of Layer-to-Layer Translation
- The success of the tuned lens (an affine map) in translating representations across layers demonstrates that the geometry of the residual stream changes in an approximately affine manner between layers.
- This is consistent with model stitching results (Bansal et al., 2021; Csiszárik et al., 2021) showing that affine transforms suffice to stitch independently trained models.
- The translators have Frobenius norms several times larger than the identity when trained with Muon, indicating that the required transformation is not trivially close to a change of scale/shift.

### Causal Features and the Residual Stream
- Causal basis extraction identifies the principal directions in the residual stream at each layer that are most influential on the tuned lens output.
- These directions are highly correlated with the directions most influential on the actual model output (Spearman ρ = 0.89 at intermediate layers, increasing to 0.98 near the output).
- The model is somewhat "more causally sensitive" than the lens: even the least influential features for the tuned lens have some effect on the model (≥ 2×10^{-3} bits), producing the characteristic hockey-stick shape in causal influence plots.

### Static Analysis of Weight Matrices
- Transformer weight matrices (OV circuits, QK circuits, MLP in/out matrices) can be projected into token space via the unembedding to yield interpretable semantic clusters.
- Applying the tuned lens translator before the unembedding improves interpretability for Pythia 125M (Table 3), as measured by BPEmb embedding cosine similarity.
- Most singular vectors are *not* more interpretable than random; interpretability is concentrated in a small fraction of high-scoring directions (long right tail), complicating the use of static analysis methods at scale.
