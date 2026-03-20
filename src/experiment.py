"""
Characterizing Granular Deletes: Logit Lens + Residual Stream Geometry
=====================================================================

This script runs four experiments:
1. Logit lens profiling: Layer-by-layer correct-object probability
2. MLP ablation study: Which layers contribute most to factual recall
3. Geometric analysis: PCA of residual stream fact representations
4. Deletion geometry: How rank-one editing changes the residual stream

Uses GPT-2-small via TransformerLens on CounterFact dataset.
"""

import json
import os
import random
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import stats
from sklearn.decomposition import PCA
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = Path("/workspaces/granular-deletes-nlp-claude")
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"


def load_counterfact(max_examples=200):
    """Load CounterFact dataset, filtering to single-token target objects."""
    from datasets import load_dataset
    ds = load_dataset("azhx/counterfact", split="train")
    print(f"Loaded CounterFact with {len(ds)} examples")
    print(f"Columns: {ds.column_names}")
    return ds


def filter_single_token_facts(model, dataset, max_examples=200):
    """Filter to facts where the target object is a single token."""
    filtered = []
    for i in range(len(dataset)):
        if len(filtered) >= max_examples:
            break
        ex = dataset[i]
        rw = ex["requested_rewrite"]
        prompt = rw["prompt"].replace("{}", rw["subject"])
        target_str = rw["target_true"]["str"]

        if not prompt or not target_str:
            continue

        # Check single token
        target_tokens = model.to_tokens(" " + target_str, prepend_bos=False)[0]
        if len(target_tokens) == 1:
            filtered.append({
                "prompt": prompt,
                "target": target_str,
                "target_token": target_tokens[0].item(),
                "index": i,
            })

    print(f"Filtered to {len(filtered)} single-token-target facts")
    return filtered


def logit_lens_at_layer(model, cache, layer, target_tokens=None):
    """Apply logit lens: project residual stream through unembedding.

    Returns log-probabilities at the last token position for each example.
    """
    # Get residual stream at layer
    resid = cache[f"blocks.{layer}.hook_resid_post"]  # (batch, seq, d_model)
    # Take the last token position
    last_resid = resid[:, -1, :]  # (batch, d_model)
    # Apply layer norm and unembedding
    normed = model.ln_final(last_resid)
    logits = model.unembed(normed)  # (batch, vocab)
    log_probs = torch.log_softmax(logits, dim=-1)

    if target_tokens is not None:
        target_tokens = torch.tensor(target_tokens, device=logits.device)
        target_log_probs = log_probs[torch.arange(len(target_tokens)), target_tokens]
        return target_log_probs.detach().cpu().numpy(), log_probs.detach().cpu().numpy()
    return log_probs.detach().cpu().numpy()


def experiment1_logit_lens_profiling(model, facts, batch_size=32):
    """Experiment 1: Layer-by-layer logit lens analysis.

    For each fact, compute the probability of the correct target token
    at every layer using the logit lens.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1: Logit Lens Profiling")
    print("="*60)

    n_layers = model.cfg.n_layers
    all_layer_probs = np.zeros((len(facts), n_layers))
    all_layer_ranks = np.zeros((len(facts), n_layers))

    for batch_start in tqdm(range(0, len(facts), batch_size), desc="Logit lens"):
        batch = facts[batch_start:batch_start+batch_size]
        prompts = [f["prompt"] for f in batch]
        target_tokens = [f["target_token"] for f in batch]

        tokens = model.to_tokens(prompts, prepend_bos=True)
        _, cache = model.run_with_cache(tokens)

        for layer in range(n_layers):
            target_log_probs, full_log_probs = logit_lens_at_layer(
                model, cache, layer, target_tokens
            )
            probs = np.exp(target_log_probs)
            all_layer_probs[batch_start:batch_start+len(batch), layer] = probs

            # Compute rank of target token
            for j, (tlp, flp) in enumerate(zip(target_log_probs, full_log_probs)):
                rank = (flp > tlp).sum()
                all_layer_ranks[batch_start:batch_start+len(batch[j:j+1]), layer] = rank

        del cache
        torch.cuda.empty_cache()

    # Save results
    results = {
        "mean_prob_per_layer": all_layer_probs.mean(axis=0).tolist(),
        "std_prob_per_layer": all_layer_probs.std(axis=0).tolist(),
        "median_prob_per_layer": np.median(all_layer_probs, axis=0).tolist(),
        "mean_rank_per_layer": all_layer_ranks.mean(axis=0).tolist(),
        "n_facts": len(facts),
        "n_layers": n_layers,
    }

    with open(RESULTS_DIR / "data" / "exp1_logit_lens.json", "w") as f:
        json.dump(results, f, indent=2)
    np.save(RESULTS_DIR / "data" / "exp1_layer_probs.npy", all_layer_probs)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    layers = np.arange(n_layers)
    mean_probs = all_layer_probs.mean(axis=0)
    std_probs = all_layer_probs.std(axis=0) / np.sqrt(len(facts))  # SEM

    axes[0].plot(layers, mean_probs, "b-o", markersize=4)
    axes[0].fill_between(layers, mean_probs - 1.96*std_probs, mean_probs + 1.96*std_probs, alpha=0.3)
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("P(correct target)")
    axes[0].set_title("Logit Lens: Correct Target Probability by Layer")
    axes[0].grid(True, alpha=0.3)

    # Enrichment: layer-to-layer change
    enrichment = np.diff(mean_probs)
    colors = ["green" if e > 0 else "red" for e in enrichment]
    axes[1].bar(layers[1:], enrichment, color=colors, alpha=0.7)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("ΔP(correct target)")
    axes[1].set_title("Information Enrichment by Layer (Δ probability)")
    axes[1].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "exp1_logit_lens_profile.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Mean prob at layer 0: {mean_probs[0]:.4f}")
    print(f"Mean prob at final layer: {mean_probs[-1]:.4f}")
    print(f"Layer with max enrichment: {np.argmax(enrichment)+1} (Δ={enrichment.max():.4f})")

    return all_layer_probs, results


def experiment2_ablation_study(model, facts, batch_size=32):
    """Experiment 2: MLP zero-ablation at each layer.

    For each layer, zero out the MLP output and measure the impact
    on final-layer logit lens probability of the correct target.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: MLP Ablation Study")
    print("="*60)

    n_layers = model.cfg.n_layers

    # First get baseline (no ablation) final-layer probabilities
    baseline_probs = np.zeros(len(facts))
    for batch_start in range(0, len(facts), batch_size):
        batch = facts[batch_start:batch_start+batch_size]
        prompts = [f["prompt"] for f in batch]
        target_tokens = [f["target_token"] for f in batch]

        tokens = model.to_tokens(prompts, prepend_bos=True)
        logits = model(tokens)
        log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
        for j, tt in enumerate(target_tokens):
            baseline_probs[batch_start+j] = np.exp(log_probs[j, tt].item())
        del logits
        torch.cuda.empty_cache()

    # Ablation: zero out MLP at each layer
    ablation_probs = np.zeros((len(facts), n_layers))

    for ablate_layer in tqdm(range(n_layers), desc="Ablating MLPs"):
        # Hook to zero out MLP output at this layer
        def make_hook(layer_idx):
            def hook_fn(value, hook):
                value[:, :, :] = 0.0
                return value
            return hook_fn

        for batch_start in range(0, len(facts), batch_size):
            batch = facts[batch_start:batch_start+batch_size]
            prompts = [f["prompt"] for f in batch]
            target_tokens = [f["target_token"] for f in batch]

            tokens = model.to_tokens(prompts, prepend_bos=True)
            hook_name = f"blocks.{ablate_layer}.hook_mlp_out"
            logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_name, make_hook(ablate_layer))]
            )
            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
            for j, tt in enumerate(target_tokens):
                ablation_probs[batch_start+j, ablate_layer] = np.exp(log_probs[j, tt].item())
            del logits
            torch.cuda.empty_cache()

    # Compute impact: baseline - ablated
    impact = baseline_probs[:, None] - ablation_probs  # positive = layer was helpful

    results = {
        "baseline_mean_prob": baseline_probs.mean().item(),
        "mean_impact_per_layer": impact.mean(axis=0).tolist(),
        "std_impact_per_layer": impact.std(axis=0).tolist(),
        "mean_ablated_prob_per_layer": ablation_probs.mean(axis=0).tolist(),
    }

    with open(RESULTS_DIR / "data" / "exp2_ablation.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    layers = np.arange(n_layers)
    mean_impact = impact.mean(axis=0)
    sem_impact = impact.std(axis=0) / np.sqrt(len(facts))

    colors = ["#d62728" if m > 0 else "#1f77b4" for m in mean_impact]
    axes[0].bar(layers, mean_impact, color=colors, alpha=0.8)
    axes[0].errorbar(layers, mean_impact, yerr=1.96*sem_impact, fmt="none", color="black", capsize=2)
    axes[0].set_xlabel("Ablated Layer")
    axes[0].set_ylabel("ΔP (baseline - ablated)")
    axes[0].set_title("Impact of MLP Ablation on Correct Target Probability")
    axes[0].grid(True, alpha=0.3)

    # Also show the logit lens profile with ablation at the most impactful layer
    top_layer = np.argmax(mean_impact)
    axes[1].bar(layers, ablation_probs.mean(axis=0), alpha=0.7, label="Ablated prob")
    axes[1].axhline(y=baseline_probs.mean(), color="red", linestyle="--", label="Baseline")
    axes[1].set_xlabel("Ablated Layer")
    axes[1].set_ylabel("Mean P(correct target)")
    axes[1].set_title("Remaining Target Probability After Ablation")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "exp2_ablation_impact.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Baseline mean prob: {baseline_probs.mean():.4f}")
    print(f"Most impactful MLP layer: {top_layer} (ΔP={mean_impact[top_layer]:.4f})")

    return impact, results


def experiment3_geometry(model, facts, batch_size=32):
    """Experiment 3: Geometric analysis of residual stream.

    Extract residual stream activations at key layers and analyze:
    - PCA of fact representations
    - Cosine similarity between fact directions
    - How fact geometry evolves across layers
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3: Residual Stream Geometry")
    print("="*60)

    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    # Collect residual stream at several layers
    key_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    layer_activations = {l: [] for l in key_layers}

    for batch_start in tqdm(range(0, len(facts), batch_size), desc="Collecting activations"):
        batch = facts[batch_start:batch_start+batch_size]
        prompts = [f["prompt"] for f in batch]

        tokens = model.to_tokens(prompts, prepend_bos=True)
        _, cache = model.run_with_cache(tokens)

        for layer in key_layers:
            resid = cache[f"blocks.{layer}.hook_resid_post"][:, -1, :]  # last token
            layer_activations[layer].append(resid.cpu().numpy())

        del cache
        torch.cuda.empty_cache()

    # Concatenate
    for layer in key_layers:
        layer_activations[layer] = np.concatenate(layer_activations[layer], axis=0)

    # PCA analysis
    pca_results = {}
    fig, axes = plt.subplots(1, len(key_layers), figsize=(4*len(key_layers), 4))

    for idx, layer in enumerate(key_layers):
        activations = layer_activations[layer]
        pca = PCA(n_components=min(50, activations.shape[0], activations.shape[1]))
        transformed = pca.fit_transform(activations)

        pca_results[f"layer_{layer}"] = {
            "explained_variance_ratio": pca.explained_variance_ratio_[:10].tolist(),
            "cumulative_variance_50": float(pca.explained_variance_ratio_.sum()),
        }

        # 2D scatter
        axes[idx].scatter(transformed[:, 0], transformed[:, 1], s=8, alpha=0.5)
        axes[idx].set_title(f"Layer {layer}\n({pca.explained_variance_ratio_[0]:.1%}, {pca.explained_variance_ratio_[1]:.1%})")
        axes[idx].set_xlabel("PC1")
        axes[idx].set_ylabel("PC2")

    plt.suptitle("PCA of Residual Stream (Last Token Position)", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "exp3_pca_layers.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Cosine similarity between fact representations at different layers
    cosine_sim_matrix = np.zeros((len(key_layers), len(key_layers)))
    for i, l1 in enumerate(key_layers):
        for j, l2 in enumerate(key_layers):
            # Mean cosine similarity between corresponding facts
            a1 = layer_activations[l1]
            a2 = layer_activations[l2]
            # Normalize
            a1_norm = a1 / (np.linalg.norm(a1, axis=1, keepdims=True) + 1e-10)
            a2_norm = a2 / (np.linalg.norm(a2, axis=1, keepdims=True) + 1e-10)
            cos_sims = (a1_norm * a2_norm).sum(axis=1)
            cosine_sim_matrix[i, j] = cos_sims.mean()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cosine_sim_matrix, annot=True, fmt=".3f",
                xticklabels=[f"L{l}" for l in key_layers],
                yticklabels=[f"L{l}" for l in key_layers],
                cmap="RdYlBu_r", ax=ax, vmin=-0.1, vmax=1.0)
    ax.set_title("Cross-Layer Cosine Similarity of Fact Representations")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "exp3_cosine_similarity.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Norm analysis: how activation norms change across layers
    all_norms = {}
    for batch_start in range(0, len(facts), batch_size):
        batch = facts[batch_start:batch_start+batch_size]
        prompts = [f["prompt"] for f in batch]
        tokens = model.to_tokens(prompts, prepend_bos=True)
        _, cache = model.run_with_cache(tokens)
        for layer in range(n_layers):
            resid = cache[f"blocks.{layer}.hook_resid_post"][:, -1, :]
            norms = torch.norm(resid, dim=-1).cpu().numpy()
            if layer not in all_norms:
                all_norms[layer] = []
            all_norms[layer].append(norms)
        del cache
        torch.cuda.empty_cache()

    norm_means = [float(np.concatenate(all_norms[l]).mean()) for l in range(n_layers)]
    norm_stds = [float(np.concatenate(all_norms[l]).std()) for l in range(n_layers)]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(n_layers), norm_means, "g-o", markersize=4)
    ax.fill_between(range(n_layers),
                     [m-s for m, s in zip(norm_means, norm_stds)],
                     [m+s for m, s in zip(norm_means, norm_stds)], alpha=0.2, color="green")
    ax.set_xlabel("Layer")
    ax.set_ylabel("L2 Norm")
    ax.set_title("Residual Stream Norm Across Layers")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "exp3_norm_profile.png", dpi=150, bbox_inches="tight")
    plt.close()

    results = {
        "pca": pca_results,
        "cosine_similarity_matrix": cosine_sim_matrix.tolist(),
        "key_layers": key_layers,
        "norm_means": norm_means,
        "norm_stds": norm_stds,
    }

    with open(RESULTS_DIR / "data" / "exp3_geometry.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Key layers analyzed: {key_layers}")
    for layer in key_layers:
        evr = pca_results[f"layer_{layer}"]["explained_variance_ratio"]
        print(f"  Layer {layer}: PC1={evr[0]:.3f}, PC2={evr[1]:.3f}, Top-10 cumul={sum(evr):.3f}")

    return layer_activations, results


def experiment4_deletion_geometry(model, facts, batch_size=32):
    """Experiment 4: Rank-one editing and its effect on logit lens + geometry.

    Apply a simplified rank-one update to the MLP weights at the most
    critical layer, then re-analyze logit lens and geometry.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 4: Deletion Geometry via Rank-One Editing")
    print("="*60)

    n_layers = model.cfg.n_layers
    n_edit = min(50, len(facts))  # Edit a subset of facts
    edit_facts = facts[:n_edit]

    # Step 1: Identify the critical layer (highest enrichment from Exp 1)
    exp1_file = RESULTS_DIR / "data" / "exp1_logit_lens.json"
    with open(exp1_file) as f:
        exp1 = json.load(f)
    mean_probs = exp1["mean_prob_per_layer"]
    enrichment = np.diff(mean_probs)
    critical_layer = int(np.argmax(enrichment) + 1)
    print(f"Critical layer for editing: {critical_layer}")

    # Step 2: Collect pre-edit activations and logit lens profiles
    pre_edit_probs = np.zeros((n_edit, n_layers))
    pre_edit_resid = []

    for batch_start in range(0, n_edit, batch_size):
        batch = edit_facts[batch_start:batch_start+batch_size]
        prompts = [f["prompt"] for f in batch]
        target_tokens = [f["target_token"] for f in batch]

        tokens = model.to_tokens(prompts, prepend_bos=True)
        _, cache = model.run_with_cache(tokens)

        for layer in range(n_layers):
            target_lp, _ = logit_lens_at_layer(model, cache, layer, target_tokens)
            pre_edit_probs[batch_start:batch_start+len(batch), layer] = np.exp(target_lp)

        # Save residual at critical layer
        resid = cache[f"blocks.{critical_layer}.hook_resid_post"][:, -1, :].cpu().numpy()
        pre_edit_resid.append(resid)

        del cache
        torch.cuda.empty_cache()

    pre_edit_resid = np.concatenate(pre_edit_resid, axis=0)

    # Step 3: Perform rank-one editing
    # We use a simplified approach: for each fact, compute the direction that
    # the MLP at the critical layer adds for the correct answer, then subtract it.
    # This is a simplified version of ROME's rank-one update.

    # Collect MLP outputs at critical layer for edit facts
    mlp_outputs = []
    for batch_start in range(0, n_edit, batch_size):
        batch = edit_facts[batch_start:batch_start+batch_size]
        prompts = [f["prompt"] for f in batch]
        tokens = model.to_tokens(prompts, prepend_bos=True)
        _, cache = model.run_with_cache(tokens)
        mlp_out = cache[f"blocks.{critical_layer}.hook_mlp_out"][:, -1, :].cpu()
        mlp_outputs.append(mlp_out)
        del cache
        torch.cuda.empty_cache()

    mlp_outputs = torch.cat(mlp_outputs, dim=0)  # (n_edit, d_model)
    mean_mlp_direction = mlp_outputs.mean(dim=0)  # average "fact direction"
    mean_mlp_direction = mean_mlp_direction / mean_mlp_direction.norm()

    # Step 4: Apply deletion hook - project out the mean fact direction
    def deletion_hook(value, hook):
        """Project out the mean fact direction from MLP output."""
        direction = mean_mlp_direction.to(value.device)
        # For each position, remove the component along the fact direction
        proj = torch.einsum("bsd,d->bs", value, direction).unsqueeze(-1) * direction
        return value - proj

    # Step 5: Collect post-edit logit lens profiles
    post_edit_probs = np.zeros((n_edit, n_layers))
    post_edit_resid = []

    hook_name = f"blocks.{critical_layer}.hook_mlp_out"

    for batch_start in range(0, n_edit, batch_size):
        batch = edit_facts[batch_start:batch_start+batch_size]
        prompts = [f["prompt"] for f in batch]
        target_tokens = [f["target_token"] for f in batch]

        tokens = model.to_tokens(prompts, prepend_bos=True)

        # Run with deletion hook
        logits = model.run_with_hooks(
            tokens,
            fwd_hooks=[(hook_name, deletion_hook)],
            return_type="logits"
        )

        # We need cache too for logit lens - run again with hook and cache
        # Use a simpler approach: hook into the residual stream
        def cache_and_delete(tokens_input):
            stored = {}
            def make_resid_hook(l):
                def h(value, hook):
                    stored[f"blocks.{l}.hook_resid_post"] = value.detach()
                    return value
                return h

            hooks = [(hook_name, deletion_hook)]
            for l in range(n_layers):
                hooks.append((f"blocks.{l}.hook_resid_post", make_resid_hook(l)))

            model.run_with_hooks(tokens_input, fwd_hooks=hooks)
            return stored

        stored_cache = cache_and_delete(tokens)

        for layer in range(n_layers):
            resid = stored_cache[f"blocks.{layer}.hook_resid_post"][:, -1, :]
            normed = model.ln_final(resid)
            logits_l = model.unembed(normed)
            log_probs_l = torch.log_softmax(logits_l, dim=-1)
            for j, tt in enumerate(target_tokens):
                post_edit_probs[batch_start+j, layer] = np.exp(log_probs_l[j, tt].item())

        # Save post-edit residual at critical layer
        post_resid = stored_cache[f"blocks.{critical_layer}.hook_resid_post"][:, -1, :].cpu().numpy()
        post_edit_resid.append(post_resid)

        del stored_cache
        torch.cuda.empty_cache()

    post_edit_resid = np.concatenate(post_edit_resid, axis=0)

    # Step 6: Analysis and visualization

    # 6a: Pre vs post logit lens profile
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    layers = np.arange(n_layers)
    pre_mean = pre_edit_probs.mean(axis=0)
    post_mean = post_edit_probs.mean(axis=0)
    pre_sem = pre_edit_probs.std(axis=0) / np.sqrt(n_edit)
    post_sem = post_edit_probs.std(axis=0) / np.sqrt(n_edit)

    axes[0].plot(layers, pre_mean, "b-o", markersize=4, label="Pre-edit")
    axes[0].fill_between(layers, pre_mean-1.96*pre_sem, pre_mean+1.96*pre_sem, alpha=0.2, color="blue")
    axes[0].plot(layers, post_mean, "r-s", markersize=4, label="Post-edit")
    axes[0].fill_between(layers, post_mean-1.96*post_sem, post_mean+1.96*post_sem, alpha=0.2, color="red")
    axes[0].axvline(x=critical_layer, color="gray", linestyle="--", alpha=0.5, label=f"Edit layer ({critical_layer})")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("P(correct target)")
    axes[0].set_title("Logit Lens: Pre vs Post Deletion")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 6b: Probability drop per layer
    drop = pre_mean - post_mean
    axes[1].bar(layers, drop, color=["red" if d > 0 else "blue" for d in drop], alpha=0.7)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("ΔP (pre - post)")
    axes[1].set_title("Probability Drop per Layer After Deletion")
    axes[1].axvline(x=critical_layer, color="gray", linestyle="--", alpha=0.5)
    axes[1].grid(True, alpha=0.3)

    # 6c: PCA comparing pre vs post geometry at critical layer
    combined = np.vstack([pre_edit_resid, post_edit_resid])
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(combined)

    axes[2].scatter(transformed[:n_edit, 0], transformed[:n_edit, 1],
                    s=15, alpha=0.6, label="Pre-edit", color="blue")
    axes[2].scatter(transformed[n_edit:, 0], transformed[n_edit:, 1],
                    s=15, alpha=0.6, label="Post-edit", color="red")
    # Draw arrows from pre to post
    for i in range(min(20, n_edit)):
        axes[2].annotate("", xy=transformed[n_edit+i], xytext=transformed[i],
                         arrowprops=dict(arrowstyle="->", color="gray", alpha=0.3))
    axes[2].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[2].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    axes[2].set_title(f"Residual Stream Geometry at Layer {critical_layer}")
    axes[2].legend()

    plt.suptitle("Experiment 4: Effect of Fact Deletion on Residual Stream", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "exp4_deletion_geometry.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Compute geometric statistics
    # Cosine similarity between pre and post
    pre_norm = pre_edit_resid / (np.linalg.norm(pre_edit_resid, axis=1, keepdims=True) + 1e-10)
    post_norm = post_edit_resid / (np.linalg.norm(post_edit_resid, axis=1, keepdims=True) + 1e-10)
    cos_sims = (pre_norm * post_norm).sum(axis=1)

    # L2 distance
    l2_dists = np.linalg.norm(pre_edit_resid - post_edit_resid, axis=1)

    # Statistical test: paired t-test on probability at final layer
    t_stat, p_val = stats.ttest_rel(pre_edit_probs[:, -1], post_edit_probs[:, -1])
    cohens_d = (pre_edit_probs[:, -1].mean() - post_edit_probs[:, -1].mean()) / np.sqrt(
        (pre_edit_probs[:, -1].std()**2 + post_edit_probs[:, -1].std()**2) / 2
    )

    results = {
        "critical_layer": critical_layer,
        "n_edit_facts": n_edit,
        "pre_mean_final_prob": float(pre_mean[-1]),
        "post_mean_final_prob": float(post_mean[-1]),
        "prob_drop_per_layer": drop.tolist(),
        "mean_cosine_similarity": float(cos_sims.mean()),
        "std_cosine_similarity": float(cos_sims.std()),
        "mean_l2_distance": float(l2_dists.mean()),
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "cohens_d": float(cohens_d),
        "pca_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }

    with open(RESULTS_DIR / "data" / "exp4_deletion.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Pre-edit final prob: {pre_mean[-1]:.4f}")
    print(f"Post-edit final prob: {post_mean[-1]:.4f}")
    print(f"Mean cos similarity (pre vs post): {cos_sims.mean():.4f}")
    print(f"Mean L2 distance: {l2_dists.mean():.4f}")
    print(f"Paired t-test: t={t_stat:.3f}, p={p_val:.6f}")
    print(f"Cohen's d: {cohens_d:.3f}")

    return results


def main():
    """Run all experiments."""
    print("="*60)
    print("Characterizing Granular Deletes via Logit Lens")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load model
    print("\nLoading GPT-2-small via TransformerLens...")
    import transformer_lens
    print(f"TransformerLens: installed")
    model = transformer_lens.HookedTransformer.from_pretrained("gpt2", device=DEVICE)
    print(f"Model: {model.cfg.model_name}, {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")

    # Load and filter data
    print("\nLoading CounterFact dataset...")
    dataset = load_counterfact()
    facts = filter_single_token_facts(model, dataset, max_examples=200)

    if len(facts) < 50:
        print(f"WARNING: Only {len(facts)} single-token facts found. Results may be noisy.")

    # Save config
    config = {
        "seed": SEED,
        "device": DEVICE,
        "model": "gpt2",
        "n_layers": model.cfg.n_layers,
        "d_model": model.cfg.d_model,
        "n_facts": len(facts),
        "batch_size": 32,
    }
    with open(RESULTS_DIR / "data" / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Run experiments
    print("\n" + "="*60)
    print("Starting Experiments")
    print("="*60)

    # Experiment 1
    layer_probs, exp1_results = experiment1_logit_lens_profiling(model, facts)

    # Experiment 2
    impact, exp2_results = experiment2_ablation_study(model, facts)

    # Experiment 3
    activations, exp3_results = experiment3_geometry(model, facts)

    # Experiment 4
    exp4_results = experiment4_deletion_geometry(model, facts)

    # Summary
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")

    # Print summary statistics
    print("\n--- Summary ---")
    print(f"Exp 1: Info enters residual stream primarily at layers with max enrichment")
    enrichment = np.diff(exp1_results["mean_prob_per_layer"])
    print(f"  Max enrichment layer: {np.argmax(enrichment)+1}")
    print(f"  Final layer prob: {exp1_results['mean_prob_per_layer'][-1]:.4f}")

    print(f"Exp 2: Most impactful MLP layer for factual recall:")
    top_layer = np.argmax(exp2_results["mean_impact_per_layer"])
    print(f"  Layer {top_layer} (ΔP={exp2_results['mean_impact_per_layer'][top_layer]:.4f})")

    print(f"Exp 3: Residual stream geometry evolves across layers")
    for layer in exp3_results["key_layers"]:
        evr = exp3_results["pca"][f"layer_{layer}"]["explained_variance_ratio"][:3]
        print(f"  Layer {layer}: top-3 PCA = {[f'{v:.3f}' for v in evr]}")

    print(f"Exp 4: Deletion effect")
    print(f"  Pre-edit prob: {exp4_results['pre_mean_final_prob']:.4f}")
    print(f"  Post-edit prob: {exp4_results['post_mean_final_prob']:.4f}")
    print(f"  Cohen's d: {exp4_results['cohens_d']:.3f}")
    print(f"  p-value: {exp4_results['p_value']:.6f}")


if __name__ == "__main__":
    main()
