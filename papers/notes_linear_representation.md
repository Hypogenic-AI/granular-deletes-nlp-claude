# Notes: The Linear Representation Hypothesis and the Geometry of Large Language Models

**Citation:** Park, K., Choe, Y. J., & Veitch, V. (2023). The Linear Representation Hypothesis and the Geometry of Large Language Models. *Proceedings of the 41st International Conference on Machine Learning*, Vienna, Austria. PMLR 235, 2024. (arXiv:2311.03658v2)

**Authors:** Kiho Park, Yo Joong Choe, Victor Veitch (University of Chicago)

**Code:** https://github.com/KihoPark/linear_rep_geometry

---

## 1. Research Question and Key Contribution

### Research Questions
The paper addresses two tightly coupled questions:
1. What does "linear representation" of a concept actually mean — precisely and formally?
2. How do we make sense of geometric notions (cosine similarity, projection, orthogonality) in the representation space of an LLM?

### Key Contributions
1. **Formal unification of three notions of linear representation** using counterfactual pairs, in both the embedding (input/context) space and the unembedding (output/word) space. The paper proves that:
   - The *unembedding* notion connects to *measurement* (linear probing).
   - The *embedding* notion connects to *intervention* (model steering).

2. **Introduction of the causal inner product**: a non-Euclidean inner product on the representation space defined by the principle that causally separable concepts should be represented as orthogonal directions. This inner product unifies the embedding and unembedding representations via a Riesz isomorphism.

3. **An explicit, tractable formula for the causal inner product**: shown to be the inverse covariance of the unembedding vectors, `M = Cov(γ)^{-1}`, estimated directly from the LLM's unembedding matrix.

4. **Empirical validation on LLaMA-2-7B**: demonstrating linear representations for 27 concepts, the orthogonality of causally separable concept directions under the causal inner product, and successful construction of both probes and steering vectors.

---

## 2. What is the Linear Representation Hypothesis

The **Linear Representation Hypothesis (LRH)** is the informal claim that high-level semantic concepts are represented linearly as directions in the representation space of a model. Prior work (Mikolov et al., 2013; Arora et al., 2016; Elhage et al., 2022) had empirically observed this but without a precise definition.

The paper identifies **three distinct prior interpretations** and formalizes all three:

### 2a. Subspace Notion
Each concept corresponds to a 1-dimensional subspace (direction). For example, the differences `γ("queen") − γ("king")`, `γ("woman") − γ("man")`, and all similar gendered pairs are parallel — they point to a common direction representing the male⇒female concept.

**Formal definition (Unembedding):** A vector `γ̄_W` is an *unembedding representation* of concept `W` if for all counterfactual output pairs `(Y(0), Y(1))` that differ only in the value of `W`:
```
γ(Y(1)) − γ(Y(0)) ∈ Cone(γ̄_W)  almost surely
```
(The "cone" accounts for the directional sign being meaningful.)

**Formal definition (Embedding):** A vector `λ̄_W` is an *embedding representation* of concept `W` if differences between context embeddings that vary only on the target concept (and not on causally separable off-target concepts) all point in the direction of `λ̄_W`.

### 2b. Measurement Notion
The probability of a concept value can be measured using a linear probe on the representation. Formally, the probability that the output word reflects concept `W = 1` is logit-linear in the embedding `λ`:
```
logit P(Y = Y(1) | Y ∈ {Y(0), Y(1)}, λ) = α λ^T γ̄_W
```
The key distinction from a trained linear probe: the unembedding representation `γ̄_W` does not encode spurious correlations between concepts (e.g., a probe trained on French text might conflate "French" with "about men" if those co-occur in training data, but the unembedding representation would not).

### 2c. Intervention Notion
The value a concept takes can be changed without affecting other concepts by adding a "steering vector" to the context embedding. Adding `λ̄_W` (the embedding representation) increases the probability of `W=1` while leaving all causally separable concepts unchanged.

### Formal Definition of Concepts
A **concept variable** `W` is a latent variable caused by context `X` and causing output `Y`. Concepts are defined by their counterfactual outputs: `{Y(W=w)}` for `w ∈ {0, 1}`. Two concepts `W` and `Z` are **causally separable** if `Y(W=w, Z=z)` is well-defined for all `w, z` — i.e., they can be varied freely and independently (e.g., English⇒French and male⇒female are causally separable; English⇒French and English⇒Russian are not).

---

## 3. How They Test and Validate the Hypothesis

### Validation of the Subspace Notion (Unembedding Space)
For each of 27 concepts, they estimate the concept direction `γ̄_W` as the normalized mean of counterfactual word pair differences in unembedding space. Using a leave-one-out (LOO) estimate to avoid circularity, they compute projections of each counterfactual pair onto `γ̄_W` using the causal inner product.

**Finding:** The distribution of projections for counterfactual pairs is strongly right-skewed compared to random word pairs — showing that counterfactual pairs are substantially more aligned with the concept direction than chance. This holds for 26 of 27 concepts; the sole exception is `thing⇒part`, which appears to lack a linear representation.

### Validation of the Causal Inner Product
They plot a 27×27 heatmap of `|⟨γ̄_W, γ̄_Z⟩_C|` for all concept pairs. If the estimated inner product is truly causal, causally separable concepts should have near-zero inner product.

**Finding:** Most concept pairs are nearly orthogonal under the causal inner product. A clear block-diagonal structure emerges (concepts cluster by semantic type: verb inflections, adjective forms, language pairs, etc.). Non-zero off-diagonal entries are semantically interpretable (e.g., `lower⇒upper` correlates with language pair concepts that differ in capitalization conventions).

### Validation of the Measurement Connection
For language concepts (e.g., French⇒Spanish), they sample contexts from Wikipedia in each language and compute `γ̄_W^T λ(x)` for contexts from both languages.

**Finding:** The unembedding representation `γ̄_W` for French⇒Spanish separates French and Spanish contexts. The representation for an off-target concept (male⇒female) has no predictive power, confirming concept specificity.

### Validation of the Intervention Connection
Using the isomorphism `λ̄_W = Cov(γ)^{-1} γ̄_W`, they construct steering vectors and apply interventions `λ_{C,α}(x) = λ(x) + α λ̄_C` for increasing `α`.

**Finding:** Intervening in the target concept direction (e.g., male⇒female) increases the logit for the target concept ("queen" over "king") while leaving causally separable concept logits unchanged ("King" over "king" stays constant). Intervening in a direction causally separable from both leaves both logits unchanged.

### Additional Sanity Check
They verify the uncorrelatedness assumption underlying the causal inner product: `λ̄_W^T γ` and `λ̄_Z^T γ` are uncorrelated across vocabulary tokens for causally separable concepts (W=male⇒female, Z=English⇒French), but correlated for non-causally-separable concepts (W=verb⇒3pSg, Z=verb⇒Ving).

---

## 4. Geometry of Representations in the Residual Stream

The paper focuses specifically on the final-layer representations (embedding and unembedding vectors), not intermediate residual stream states, though it notes extending to intermediate layers is future work.

### Two Distinct Representation Spaces
- **Embedding space Λ ≅ R^d**: Context representations `λ(x)` — where the model encodes its "understanding" of the input context.
- **Unembedding space Γ ≅ R^d**: Word/token representations `γ(y)` — where output tokens are represented.

The probability of generating token `y` given context `x` is: `P(y|x) ∝ exp(λ(x)^T γ(y))`.

### The Inner Product Problem
The key geometric insight is that the standard Euclidean inner product is **not canonically defined** on these spaces. The reason: the softmax output distribution is invariant under the transformation `γ(y) ← Aγ(y) + β`, `λ(x) ← A^{-T}λ(x)` for any invertible matrix `A`. Since model training only depends on the softmax distribution, the representations are identified only up to such invertible transformations. Any fixed inner product will therefore not be semantically meaningful after such a transformation.

### The Causal Inner Product
**Definition:** A causal inner product `⟨·,·⟩_C` on `Γ̄` (the space of differences of unembedding vectors) is one where:
```
⟨γ̄_W, γ̄_Z⟩_C = 0  for any causally separable concepts W and Z
```

**Explicit Form (Theorem 3.4):** Under the assumption that knowing a randomly sampled word's value on concept `W` gives no information about causally separable concept `Z`, the causal inner product has the form:
```
⟨γ̄, γ̄'⟩_C = γ̄^T Cov(γ)^{-1} γ̄'
```
where `Cov(γ)` is the covariance of the unembedding vectors sampled uniformly over the vocabulary. This is estimated directly from the LLM's unembedding matrix.

### Unification via Riesz Isomorphism (Theorem 3.2)
The causal inner product defines a Riesz isomorphism `γ̄ ↦ ⟨γ̄, ·⟩_C` that maps the unembedding representation `γ̄_W` of each concept to its embedding representation `λ̄_W`. In practice, this means:
```
λ̄_W = Cov(γ)^{-1} γ̄_W
```
This allows steering vectors to be constructed from counterfactual word pairs alone — without needing to find paired prompts.

After applying the transformation `A = M^{1/2} = Cov(γ)^{-1/2}`, the Euclidean inner product in the transformed space equals the causal inner product in the original space. In this "unified" space, embedding and unembedding representations of each concept coincide, and causally separable concepts are orthogonal.

---

## 5. Relationship Between Concepts and Directions in Activation Space

### Core Structural Results

**Lemma (Unembedding-Embedding Relationship):** If `λ̄_W` is the embedding representation and `γ̄_W`, `γ̄_Z` are the unembedding representations of concept `W` and any causally separable concept `Z`, then:
```
λ̄_W^T γ̄_W > 0  and  λ̄_W^T γ̄_Z = 0
```
The embedding representation is positively correlated with the same concept's unembedding direction, and orthogonal to all causally separable concepts' unembedding directions.

**Theorem 2.2 (Measurement):** The unembedding direction `γ̄_W` acts as a linear probe: the log-odds of the model predicting the "W=1" token over the "W=0" token equals `α λ^T γ̄_W` for any context embedding `λ`, where `α > 0` depends only on the specific token pair.

**Theorem 2.5 (Intervention):** Adding `c λ̄_W` to a context embedding:
- Does NOT change the probability of any causally separable concept `Z` (the logit for Z remains constant in `c`).
- DOES increase the probability of concept `W=1` (the logit for W is increasing in `c`).

### Block Structure of Concept Directions
The heatmap of inner products between all 27 concept directions reveals a block-diagonal structure: concepts cluster by semantic similarity. The first 10 concepts (verb inflections: 3pSg, Ving, Ved, V+able, V+er, V+tion, V+ment) form one block; adjective forms form another; language pairs (English⇔French, French⇔German, etc.) form another. Causally non-separable concepts (e.g., different inflections of the same verb) share representation space, while causally separable concepts (e.g., grammatical tense vs. language) are orthogonal.

### The Role of the Inner Product
The paper shows through comparison with the Euclidean inner product that choice of inner product is non-trivial:
- **LLaMA-2-7B**: The Euclidean inner product accidentally works reasonably well (most causally separable concepts are near-orthogonal), possibly because LLaMA-2's training implicitly encourages approximately isotropic unembedding covariance. But the causal inner product still improves on it — e.g., the Euclidean inner product wrongly shows `frequent⇒infrequent` as non-orthogonal to many causally separable concepts.
- **Gemma-2B**: The Euclidean inner product fails to capture semantic structure at all, while the causal inner product still works. This is because Gemma ties its unembedding matrix to its token embedding matrix used before the transformer, making the Euclidean origin meaningful in a different (interfering) sense.

---

## 6. Datasets and Models Used

### Primary Model
- **LLaMA-2-7B** (`llama-2-7b`, Touvron et al., 2023): decoder-only Transformer, 7 billion parameters, pretrained on 2 trillion tokens (90% English), 32,000-token SentencePiece vocabulary, 4,096-dimensional token embeddings.

### Secondary Model (for comparison)
- **Gemma-2B** (Mesnard et al., 2024): used for comparison of Euclidean vs. causal inner product in Appendix D.2.

### Concept Datasets (27 concepts total)

**From BATS 3.0 (Bigger Analogy Test Set, Gladkova et al., 2016) — 22 concepts:**
- Verb inflections: verb⇒3pSg, verb⇒Ving, verb⇒Ved, Ving⇒3pSg, Ving⇒Ved, 3pSg⇒Ved (6 concepts)
- Derivational morphology: verb⇒V+able, verb⇒V+er, verb⇒V+tion, verb⇒V+ment, adj⇒un+adj, adj⇒adj+ly (6 concepts)
- Semantic: small⇒big, thing⇒color, thing⇒part, country⇒capital, pronoun⇒possessive, male⇒female, lower⇒upper (7 concepts)
- Inflectional: noun⇒plural, adj⇒comparative, adj⇒superlative (3 concepts)

**Additional concepts (5):**
- Language pairs: English⇒French, French⇒German, French⇒Spanish, German⇒Spanish (using word translation pairs from the word2word bilingual lexicon, Choe et al., 2020)
- Frequency: frequent⇒infrequent (pairs of common/uncommon synonyms, e.g., "bad"/"terrible", generated with ChatGPT-4)

**Pair counts per concept:** Range from 4 (pronoun⇒possessive) to 63 (noun⇒plural). Total pairs vary; only single-token words in LLaMA-2's vocabulary are used.

**Context data for measurement experiments:** Random-length text samples from French, Spanish, English, and German Wikipedia pages.

**Context data for intervention experiments:** 15 sentence fragments designed to lead to masculine royal vocabulary (e.g., "Long live the ", "Arthur was a legendary "), generated and filtered with ChatGPT-4.

---

## 7. Key Results

### Result 1: Linear Representations Exist (26/27 concepts)
For 26 out of 27 tested concepts, counterfactual word pair differences are substantially more aligned with the concept direction than random word pair differences. The sole exception is `thing⇒part`, which lacks a linear representation in the unembedding space. This strongly supports the subspace notion of the linear representation hypothesis.

### Result 2: Causal Inner Product Respects Causal Separability
The 27×27 heatmap of `|⟨γ̄_W, γ̄_Z⟩_C|` shows:
- Near-zero values (near-orthogonality) for most causally separable concept pairs.
- A meaningful block-diagonal structure grouping semantically related (causally non-separable) concepts.
- Semantically interpretable non-zero off-diagonal entries (e.g., `lower⇒upper` correlates with language pairs that differ in capitalization conventions).

### Result 3: Unembedding Directions Act as Linear Probes
The unembedding representation `γ̄_W` separates language categories in the embedding space. For French⇒Spanish contexts sampled from Wikipedia (not counterfactual pairs), `γ̄_W^T λ(x)` correctly classifies French vs. Spanish contexts. An off-target concept direction (male⇒female) has no predictive power.

### Result 4: Unembedding Directions Translate to Effective Steering Vectors
Using `λ̄_W = Cov(γ)^{-1} γ̄_W` as steering vectors:
- Intervening in the male⇒female direction raises the logit for "queen" over "king" while leaving the logit for "King" over "king" unchanged.
- At `α = 0.4` with context "Long live the ", "queen" becomes the top-1 next word while "king" falls below top-5.
- Results are consistent across 15 tested contexts and multiple concept pairs.
- Intervening in a direction causally separable from both target concepts leaves both logits unchanged (correct null effect).

### Result 5: Euclidean Inner Product is Model-Dependent
- In LLaMA-2-7B, the Euclidean inner product approximates the causal inner product (concepts are nearly orthogonal under both), likely due to implicit regularization.
- In Gemma-2B, the Euclidean inner product fails entirely to capture causal separability, while the causal inner product still works correctly. This highlights that the Euclidean inner product cannot be assumed to be semantically meaningful in general.

### Result 6: Causal Uncorrelatedness Holds Empirically
The scatter plot of `(λ̄_W^T γ, λ̄_Z^T γ)` over all 32K vocabulary tokens shows:
- Circular/uncorrelated scatter for causally separable W=male⇒female, Z=English⇒French.
- Elongated/correlated scatter for non-causally-separable W=verb⇒3pSg, Z=verb⇒Ving.

---

## 8. Limitations and Implications

### Limitations

1. **Focus on final-layer representations only.** The paper studies the embedding (`λ(x)`) and unembedding (`γ(y)`) vectors of the final layer. It does not directly address the internal residual stream activations at intermediate layers — which is the focus of much of mechanistic interpretability. The authors explicitly note extending the framework to intermediate layers as future work.

2. **Binary concepts only.** The formal framework is restricted to binary concept variables (W ∈ {0, 1}). Multi-valued concepts (e.g., tense with more than two values) are not handled in the formal theory, though extensions may be possible.

3. **Non-uniqueness of the causal inner product.** The causal orthogonality conditions impose `d(d-1)/2` constraints, but a positive definite matrix `M` (defining an inner product) has `d(d-1)/2 + d` degrees of freedom — leaving `d` degrees of freedom undetermined. The paper uses `D = I_d` (giving `M = Cov(γ)^{-1}`) as a principled but not uniquely justified choice.

4. **Tokenization noise.** Words that tokenize to multiple tokens cannot be used as counterfactual pairs (since `γ("princess")` does not exist as a single vector). Additionally, subword tokens that appear as prefixes of other words (e.g., French "bas" appearing in "basalt") introduce noise.

5. **Limited concept coverage.** The `thing⇒part` concept fails to have a linear representation, suggesting not all semantic relationships are linearly encoded. The framework does not predict which concepts will or will not be linearly represented.

6. **Requires counterfactual pairs.** Both the subspace definition and the empirical estimation of concept directions require specifying counterfactual word pairs, which must be collected manually or with assistance (e.g., ChatGPT-4 for some concepts).

### Implications

1. **Theoretical grounding for probing and steering.** The paper provides rigorous theoretical backing for why linear probes and steering vectors work. It shows that both are manifestations of the same underlying linear structure, connected via the causal inner product.

2. **The inner product matters.** Practitioners using cosine similarity or projection for representation analysis should use the causal inner product (or the whitened representation space) rather than the Euclidean inner product, especially for models that tie embedding and unembedding matrices (like Gemma).

3. **Probes from word lists, not labeled examples.** Because the unembedding representation is derived from counterfactual word pairs (not labeled contexts), concept directions can be estimated without large labeled datasets of contexts.

4. **Concept orthogonality as a diagnostic.** The causal inner product provides a principled way to check whether two concepts are causally separable — by testing whether their representations are orthogonal. This could be used to audit models for entanglement of intended and unintended concept directions.

5. **Spurious correlations and probing.** The paper highlights that a trained linear probe may learn spurious correlations between concepts (e.g., French text being disproportionately about men), whereas the unembedding representation isolates the target concept.

---

## 9. Connection to Mechanistic Interpretability

### Direct Connections

**The Linear Representation Hypothesis as Foundation.** Mechanistic interpretability (MI) research (e.g., Elhage et al., 2021, 2022; Meng et al., 2022; Nanda et al., 2023; Turner et al., 2023; Zou et al., 2023; Todd et al., 2023) relies heavily on the intuition that features are linearly encoded in residual stream activations. This paper provides the first rigorous formalization of that intuition and proves the key relationships between different operationalizations of it.

**Subspace Structure and Features.** The "superposition hypothesis" (Elhage et al., 2022) proposes that transformer models store more features than they have dimensions by representing them as near-orthogonal directions. This paper's finding that causally separable concepts are represented as orthogonal directions (under the causal inner product) provides formal and empirical support for this geometric picture.

**Steering Vectors / Activation Addition.** The intervention experiments directly connect to the "activation addition" / "representation engineering" line of work (Turner et al., 2023; Zou et al., 2023). The paper provides a principled derivation of why steering vectors work: the embedding representation `λ̄_W = Cov(γ)^{-1} γ̄_W` is the unique direction that increases the target concept without affecting causally separable concepts.

**Linear Probing.** The connection between unembedding representations and measurement (Theorem 2.2) provides theoretical grounding for the widespread practice of linear probing in MI (e.g., Nanda et al., 2023; Gurnee & Tegmark, 2023). Importantly, the paper shows the unembedding representation is in some sense an "ideal" probe — it isolates the target concept without confounding.

**Function Vectors / Task Vectors.** The broader MI literature on function vectors (Todd et al., 2023) and in-context learning task vectors (Hendel et al., 2023) studies how tasks or operations are encoded as additive vectors. This paper's formalization of concept directions as embedding representations provides a theoretical framework for understanding such vectors.

**Residual Stream and the Role of Geometry.** The paper explicitly raises the open question of how the causal inner product extends to intermediate residual stream states — the key substrate of MI analysis. Elhage et al. (2021)'s mathematical framework for transformer circuits treats residual stream activations as lying in a common linear space; this paper's causal inner product could in principle define the "correct" geometry for that space. However, the paper leaves this as future work.

### Scope and Distinction from MI
The authors explicitly distinguish their contribution from mainstream MI work:
- This paper does **not** analyze model parameters or attention heads.
- This paper does **not** study intermediate-layer activations.
- The focus is on the input/output representation structure of the final layer.

The authors frame the causal inner product as "an exciting direction for future work to understand how ideas here — particularly, the causal inner product — translate to [intermediate activation] settings."

### Causal Representation Learning Connection
The paper's use of counterfactual pairs to define concepts is inspired by causal representation learning (Wang et al., 2023; Scholkopf et al., 2021). The causal inner product results connect to identifiability theory: an inner product respecting semantic closeness is not identified by standard LLM training, but can be recovered with additional causal assumptions. This mirrors debates in MI about what structure is genuinely learned vs. an artifact of training or architecture.

---

## Summary Table

| Aspect | Detail |
|--------|--------|
| Model tested | LLaMA-2-7B (primary), Gemma-2B (comparison) |
| Concepts studied | 27 binary concepts (morphological, semantic, language, frequency) |
| Concept source | BATS 3.0, ChatGPT-4 generation, word2word lexicon |
| Representation space | Final-layer embedding (Λ) and unembedding (Γ) spaces, 4096-dim |
| Causal inner product | `⟨γ̄, γ̄'⟩_C = γ̄^T Cov(γ)^{-1} γ̄'` |
| Steering vector formula | `λ̄_W = Cov(γ)^{-1} γ̄_W` |
| Linear representations found | 26/27 concepts |
| Key negative result | `thing⇒part` lacks linear representation; Euclidean IP fails for Gemma |
| Main theorem | Causal IP unifies embedding and unembedding representations (Theorem 3.2) |
