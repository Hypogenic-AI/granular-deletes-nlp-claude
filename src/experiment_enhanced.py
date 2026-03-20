"""
Enhanced Experiments: Using log-probability and rank-based metrics.
Also uses GPT-2-medium for better factual recall, and adds
per-component (attention vs MLP) analysis.
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


def load_and_filter(model, max_examples=200):
    """Load CounterFact, filter to single-token targets with decent recall."""
    from datasets import load_dataset
    ds = load_dataset("azhx/counterfact", split="train")
    print(f"Loaded {len(ds)} examples")

    filtered = []
    for i in range(len(ds)):
        if len(filtered) >= max_examples * 3:  # oversample to filter later
            break
        rw = ds[i]["requested_rewrite"]
        prompt = rw["prompt"].replace("{}", rw["subject"])
        target_str = rw["target_true"]["str"]
        if not prompt or not target_str:
            continue
        toks = model.to_tokens(" " + target_str, prepend_bos=False)[0]
        if len(toks) == 1:
            filtered.append({
                "prompt": prompt,
                "target": target_str,
                "target_token": toks[0].item(),
                "index": i,
            })

    # Score by model's actual recall and keep best examples
    print(f"Pre-filter: {len(filtered)} single-token facts")

    # Compute model's probability for each
    scored = []
    batch_size = 64
    for batch_start in range(0, len(filtered), batch_size):
        batch = filtered[batch_start:batch_start+batch_size]
        prompts = [f["prompt"] for f in batch]
        tokens = model.to_tokens(prompts, prepend_bos=True)
        with torch.no_grad():
            logits = model(tokens)
        log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
        for j, f in enumerate(batch):
            lp = log_probs[j, f["target_token"]].item()
            f["log_prob"] = lp
            f["prob"] = np.exp(lp)
            # Rank
            rank = (log_probs[j] > lp).sum().item()
            f["rank"] = rank
            scored.append(f)
        del logits
        torch.cuda.empty_cache()

    # Sort by probability (descending) and take top max_examples
    scored.sort(key=lambda x: x["prob"], reverse=True)
    selected = scored[:max_examples]

    print(f"Selected {len(selected)} facts")
    probs = [f["prob"] for f in selected]
    ranks = [f["rank"] for f in selected]
    print(f"  Prob range: {min(probs):.6f} - {max(probs):.6f}")
    print(f"  Mean prob: {np.mean(probs):.6f}")
    print(f"  Median rank: {np.median(ranks):.0f}")
    print(f"  Facts in top-10: {sum(1 for r in ranks if r < 10)}")
    print(f"  Facts in top-100: {sum(1 for r in ranks if r < 100)}")

    return selected


def run_logit_lens_detailed(model, facts, batch_size=32):
    """Detailed logit lens: track prob, rank, and logit at every layer."""
    n_layers = model.cfg.n_layers
    n_facts = len(facts)

    layer_probs = np.zeros((n_facts, n_layers))
    layer_logprobs = np.zeros((n_facts, n_layers))
    layer_ranks = np.zeros((n_facts, n_layers))

    # Also track MLP vs attention contribution per layer
    mlp_contribution = np.zeros((n_facts, n_layers))
    attn_contribution = np.zeros((n_facts, n_layers))

    for batch_start in tqdm(range(0, n_facts, batch_size), desc="Detailed logit lens"):
        batch = facts[batch_start:batch_start+batch_size]
        prompts = [f["prompt"] for f in batch]
        target_tokens = [f["target_token"] for f in batch]
        bs = len(batch)

        tokens = model.to_tokens(prompts, prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)

        for layer in range(n_layers):
            # Logit lens on residual stream post this layer
            resid = cache[f"blocks.{layer}.hook_resid_post"][:, -1, :]
            normed = model.ln_final(resid)
            logits = model.unembed(normed)
            log_probs = torch.log_softmax(logits, dim=-1)

            for j in range(bs):
                tt = target_tokens[j]
                lp = log_probs[j, tt].item()
                layer_logprobs[batch_start+j, layer] = lp
                layer_probs[batch_start+j, layer] = np.exp(lp)
                layer_ranks[batch_start+j, layer] = (log_probs[j] > lp).sum().item()

            # MLP contribution: project MLP output through unembedding
            mlp_out = cache[f"blocks.{layer}.hook_mlp_out"][:, -1, :]
            normed_mlp = model.ln_final(mlp_out)  # approximate
            mlp_logits = model.unembed(normed_mlp)
            for j in range(bs):
                tt = target_tokens[j]
                mlp_contribution[batch_start+j, layer] = mlp_logits[j, tt].item()

            # Attention contribution
            attn_out = cache[f"blocks.{layer}.hook_attn_out"][:, -1, :]
            normed_attn = model.ln_final(attn_out)
            attn_logits = model.unembed(normed_attn)
            for j in range(bs):
                tt = target_tokens[j]
                attn_contribution[batch_start+j, layer] = attn_logits[j, tt].item()

        del cache
        torch.cuda.empty_cache()

    return {
        "probs": layer_probs,
        "logprobs": layer_logprobs,
        "ranks": layer_ranks,
        "mlp_contribution": mlp_contribution,
        "attn_contribution": attn_contribution,
    }


def run_ablation_detailed(model, facts, batch_size=32):
    """Ablation: zero MLP or attention at each layer, measure impact on all layers."""
    n_layers = model.cfg.n_layers
    n_facts = len(facts)

    # Baseline final logit lens probability
    baseline_probs = np.zeros(n_facts)
    for batch_start in range(0, n_facts, batch_size):
        batch = facts[batch_start:batch_start+batch_size]
        prompts = [f["prompt"] for f in batch]
        tokens = model.to_tokens(prompts, prepend_bos=True)
        with torch.no_grad():
            logits = model(tokens)
        lps = torch.log_softmax(logits[:, -1, :], dim=-1)
        for j, f in enumerate(batch):
            baseline_probs[batch_start+j] = np.exp(lps[j, f["target_token"]].item())
        del logits
        torch.cuda.empty_cache()

    # MLP ablation
    mlp_ablated_probs = np.zeros((n_facts, n_layers))
    for ablate_l in tqdm(range(n_layers), desc="MLP ablation"):
        def zero_hook(value, hook):
            return torch.zeros_like(value)
        for batch_start in range(0, n_facts, batch_size):
            batch = facts[batch_start:batch_start+batch_size]
            prompts = [f["prompt"] for f in batch]
            tokens = model.to_tokens(prompts, prepend_bos=True)
            with torch.no_grad():
                logits = model.run_with_hooks(tokens,
                    fwd_hooks=[(f"blocks.{ablate_l}.hook_mlp_out", zero_hook)])
            lps = torch.log_softmax(logits[:, -1, :], dim=-1)
            for j, f in enumerate(batch):
                mlp_ablated_probs[batch_start+j, ablate_l] = np.exp(lps[j, f["target_token"]].item())
            del logits
            torch.cuda.empty_cache()

    # Attention ablation
    attn_ablated_probs = np.zeros((n_facts, n_layers))
    for ablate_l in tqdm(range(n_layers), desc="Attn ablation"):
        def zero_hook(value, hook):
            return torch.zeros_like(value)
        for batch_start in range(0, n_facts, batch_size):
            batch = facts[batch_start:batch_start+batch_size]
            prompts = [f["prompt"] for f in batch]
            tokens = model.to_tokens(prompts, prepend_bos=True)
            with torch.no_grad():
                logits = model.run_with_hooks(tokens,
                    fwd_hooks=[(f"blocks.{ablate_l}.hook_attn_out", zero_hook)])
            lps = torch.log_softmax(logits[:, -1, :], dim=-1)
            for j, f in enumerate(batch):
                attn_ablated_probs[batch_start+j, ablate_l] = np.exp(lps[j, f["target_token"]].item())
            del logits
            torch.cuda.empty_cache()

    return {
        "baseline_probs": baseline_probs,
        "mlp_ablated_probs": mlp_ablated_probs,
        "attn_ablated_probs": attn_ablated_probs,
    }


def run_deletion_experiment(model, facts, critical_layers, batch_size=32):
    """Delete facts by projecting out the mean fact direction at critical layers.
    Test with single-layer and multi-layer deletion."""
    n_layers = model.cfg.n_layers
    n_facts = min(100, len(facts))
    edit_facts = facts[:n_facts]

    # Collect MLP outputs at critical layers
    layer_mlp_outputs = {l: [] for l in critical_layers}
    for batch_start in range(0, n_facts, batch_size):
        batch = edit_facts[batch_start:batch_start+batch_size]
        prompts = [f["prompt"] for f in batch]
        tokens = model.to_tokens(prompts, prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        for l in critical_layers:
            layer_mlp_outputs[l].append(cache[f"blocks.{l}.hook_mlp_out"][:, -1, :].cpu())
        del cache
        torch.cuda.empty_cache()

    # Compute mean fact direction per layer
    fact_directions = {}
    for l in critical_layers:
        outputs = torch.cat(layer_mlp_outputs[l], dim=0)
        direction = outputs.mean(dim=0)
        direction = direction / direction.norm()
        fact_directions[l] = direction

    results = {}

    # Test different deletion strategies
    for strategy_name, target_layers in [
        ("single_best", [critical_layers[0]]),
        ("top_2", critical_layers[:2]),
        ("all_critical", critical_layers),
    ]:
        def make_delete_hooks(target_ls):
            hooks = []
            for l in target_ls:
                direction = fact_directions[l]
                def make_hook(d):
                    def hook_fn(value, hook):
                        d_dev = d.to(value.device)
                        proj = torch.einsum("bsd,d->bs", value, d_dev).unsqueeze(-1) * d_dev
                        return value - proj
                    return hook_fn
                hooks.append((f"blocks.{l}.hook_mlp_out", make_hook(direction)))
            return hooks

        # Pre-deletion logit lens
        pre_probs = np.zeros((n_facts, n_layers))
        post_probs = np.zeros((n_facts, n_layers))
        pre_resid = {l: [] for l in [0, n_layers//2, n_layers-1]}
        post_resid = {l: [] for l in [0, n_layers//2, n_layers-1]}

        for batch_start in range(0, n_facts, batch_size):
            batch = edit_facts[batch_start:batch_start+batch_size]
            prompts = [f["prompt"] for f in batch]
            target_tokens = [f["target_token"] for f in batch]
            bs = len(batch)

            tokens = model.to_tokens(prompts, prepend_bos=True)

            # Pre-deletion
            with torch.no_grad():
                _, cache_pre = model.run_with_cache(tokens)

            for layer in range(n_layers):
                resid = cache_pre[f"blocks.{layer}.hook_resid_post"][:, -1, :]
                normed = model.ln_final(resid)
                logits = model.unembed(normed)
                lps = torch.log_softmax(logits, dim=-1)
                for j in range(bs):
                    pre_probs[batch_start+j, layer] = np.exp(lps[j, target_tokens[j]].item())

            for l in pre_resid:
                pre_resid[l].append(cache_pre[f"blocks.{l}.hook_resid_post"][:, -1, :].cpu().numpy())

            del cache_pre
            torch.cuda.empty_cache()

            # Post-deletion
            delete_hooks = make_delete_hooks(target_layers)
            resid_hooks = []
            stored = {}
            for l in range(n_layers):
                def make_store(layer_idx):
                    def store_fn(value, hook):
                        stored[layer_idx] = value.detach()
                        return value
                    return store_fn
                resid_hooks.append((f"blocks.{l}.hook_resid_post", make_store(l)))

            with torch.no_grad():
                model.run_with_hooks(tokens, fwd_hooks=delete_hooks + resid_hooks)

            for layer in range(n_layers):
                resid = stored[layer][:, -1, :]
                normed = model.ln_final(resid)
                logits = model.unembed(normed)
                lps = torch.log_softmax(logits, dim=-1)
                for j in range(bs):
                    post_probs[batch_start+j, layer] = np.exp(lps[j, target_tokens[j]].item())

            for l in post_resid:
                post_resid[l].append(stored[l][:, -1, :].cpu().numpy())

            del stored
            torch.cuda.empty_cache()

        for l in pre_resid:
            pre_resid[l] = np.concatenate(pre_resid[l], axis=0)
            post_resid[l] = np.concatenate(post_resid[l], axis=0)

        # Compute geometry changes
        geo_stats = {}
        for l in pre_resid:
            pre_n = pre_resid[l] / (np.linalg.norm(pre_resid[l], axis=1, keepdims=True) + 1e-10)
            post_n = post_resid[l] / (np.linalg.norm(post_resid[l], axis=1, keepdims=True) + 1e-10)
            cos_sims = (pre_n * post_n).sum(axis=1)
            l2_dists = np.linalg.norm(pre_resid[l] - post_resid[l], axis=1)
            geo_stats[l] = {
                "mean_cosine": float(cos_sims.mean()),
                "mean_l2": float(l2_dists.mean()),
            }

        # Statistical test
        t_stat, p_val = stats.ttest_rel(pre_probs[:, -1], post_probs[:, -1])
        d = pre_probs[:, -1].mean() - post_probs[:, -1].mean()
        pooled_std = np.sqrt((pre_probs[:, -1].std()**2 + post_probs[:, -1].std()**2) / 2)
        cohens_d = d / pooled_std if pooled_std > 0 else 0.0

        results[strategy_name] = {
            "target_layers": target_layers,
            "pre_mean": pre_probs.mean(axis=0).tolist(),
            "post_mean": post_probs.mean(axis=0).tolist(),
            "pre_final": float(pre_probs[:, -1].mean()),
            "post_final": float(post_probs[:, -1].mean()),
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "cohens_d": float(cohens_d),
            "geometry": {str(k): v for k, v in geo_stats.items()},
        }

        print(f"\n{strategy_name} (layers {target_layers}):")
        print(f"  Pre prob: {pre_probs[:, -1].mean():.6f}, Post: {post_probs[:, -1].mean():.6f}")
        print(f"  t={t_stat:.3f}, p={p_val:.6f}, d={cohens_d:.3f}")

    return results, pre_resid, post_resid


def create_comprehensive_figures(logit_data, ablation_data, deletion_results, model_name, n_layers):
    """Create publication-quality figures."""

    # Figure 1: Comprehensive logit lens profile
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    layers = np.arange(n_layers)

    # 1a: Probability profile
    mean_p = logit_data["probs"].mean(axis=0)
    sem_p = logit_data["probs"].std(axis=0) / np.sqrt(logit_data["probs"].shape[0])
    axes[0, 0].plot(layers, mean_p, "b-o", markersize=5)
    axes[0, 0].fill_between(layers, mean_p-1.96*sem_p, mean_p+1.96*sem_p, alpha=0.3)
    axes[0, 0].set_xlabel("Layer")
    axes[0, 0].set_ylabel("P(correct target)")
    axes[0, 0].set_title("a) Logit Lens: Target Probability by Layer")
    axes[0, 0].grid(True, alpha=0.3)

    # 1b: Log-probability profile
    mean_lp = logit_data["logprobs"].mean(axis=0)
    sem_lp = logit_data["logprobs"].std(axis=0) / np.sqrt(logit_data["logprobs"].shape[0])
    axes[0, 1].plot(layers, mean_lp, "r-o", markersize=5)
    axes[0, 1].fill_between(layers, mean_lp-1.96*sem_lp, mean_lp+1.96*sem_lp, alpha=0.3, color="red")
    axes[0, 1].set_xlabel("Layer")
    axes[0, 1].set_ylabel("log P(correct target)")
    axes[0, 1].set_title("b) Logit Lens: Target Log-Probability by Layer")
    axes[0, 1].grid(True, alpha=0.3)

    # 1c: Rank profile
    mean_rank = logit_data["ranks"].mean(axis=0)
    median_rank = np.median(logit_data["ranks"], axis=0)
    axes[1, 0].plot(layers, mean_rank, "g-o", markersize=5, label="Mean rank")
    axes[1, 0].plot(layers, median_rank, "g--s", markersize=5, label="Median rank")
    axes[1, 0].set_xlabel("Layer")
    axes[1, 0].set_ylabel("Rank of correct target")
    axes[1, 0].set_title("c) Logit Lens: Target Rank by Layer")
    axes[1, 0].legend()
    axes[1, 0].set_yscale("log")
    axes[1, 0].grid(True, alpha=0.3)

    # 1d: MLP vs Attention contribution
    mean_mlp = logit_data["mlp_contribution"].mean(axis=0)
    mean_attn = logit_data["attn_contribution"].mean(axis=0)
    width = 0.35
    axes[1, 1].bar(layers - width/2, mean_mlp, width, label="MLP", alpha=0.7, color="blue")
    axes[1, 1].bar(layers + width/2, mean_attn, width, label="Attention", alpha=0.7, color="orange")
    axes[1, 1].set_xlabel("Layer")
    axes[1, 1].set_ylabel("Logit contribution")
    axes[1, 1].set_title("d) Per-Component Contribution to Target Logit")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f"Logit Lens Analysis: {model_name} on CounterFact", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_logit_lens_comprehensive.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Figure 2: Ablation results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    baseline_mean = ablation_data["baseline_probs"].mean()
    mlp_impact = baseline_mean - ablation_data["mlp_ablated_probs"].mean(axis=0)
    attn_impact = baseline_mean - ablation_data["attn_ablated_probs"].mean(axis=0)

    axes[0].bar(layers - 0.175, mlp_impact, 0.35, label="MLP ablation", color="blue", alpha=0.7)
    axes[0].bar(layers + 0.175, attn_impact, 0.35, label="Attention ablation", color="orange", alpha=0.7)
    axes[0].set_xlabel("Ablated Layer")
    axes[0].set_ylabel("ΔP(correct) = baseline - ablated")
    axes[0].set_title("a) Impact of Layer Ablation on Factual Recall")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Cumulative importance
    mlp_cumul = np.cumsum(np.sort(np.abs(mlp_impact))[::-1])
    attn_cumul = np.cumsum(np.sort(np.abs(attn_impact))[::-1])
    axes[1].plot(range(1, n_layers+1), mlp_cumul / mlp_cumul[-1], "b-o", label="MLP", markersize=5)
    axes[1].plot(range(1, n_layers+1), attn_cumul / attn_cumul[-1], "-s", color="orange", label="Attention", markersize=5)
    axes[1].set_xlabel("Number of layers (sorted by impact)")
    axes[1].set_ylabel("Cumulative fraction of total impact")
    axes[1].set_title("b) Cumulative Layer Importance")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"Ablation Study: {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_ablation_study.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Figure 3: Deletion effects on logit lens
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (name, data) in enumerate(deletion_results.items()):
        pre = data["pre_mean"]
        post = data["post_mean"]
        axes[idx].plot(layers, pre, "b-o", markersize=4, label="Pre-deletion")
        axes[idx].plot(layers, post, "r-s", markersize=4, label="Post-deletion")
        for tl in data["target_layers"]:
            axes[idx].axvline(x=tl, color="gray", linestyle="--", alpha=0.5)
        axes[idx].set_xlabel("Layer")
        axes[idx].set_ylabel("P(correct target)")
        axes[idx].set_title(f"{name}\n(layers {data['target_layers']}, d={data['cohens_d']:.3f})")
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.suptitle(f"Deletion Effects on Logit Lens: {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_deletion_effects.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Figure 4: Enrichment heatmap
    enrichment = np.diff(logit_data["probs"], axis=1)  # (n_facts, n_layers-1)
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(enrichment[:50, :], aspect="auto", cmap="RdBu_r",
                   vmin=-np.percentile(np.abs(enrichment), 95),
                   vmax=np.percentile(np.abs(enrichment), 95))
    ax.set_xlabel("Layer transition (l → l+1)")
    ax.set_ylabel("Fact index")
    ax.set_title(f"Per-Fact Information Enrichment Across Layers ({model_name})")
    ax.set_xticks(range(n_layers-1))
    ax.set_xticklabels([f"{l}→{l+1}" for l in range(n_layers-1)], rotation=45)
    plt.colorbar(im, label="ΔP(correct target)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig4_enrichment_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("="*60)
    print("Enhanced Logit Lens + Geometry Experiments")
    print("="*60)
    print(f"Device: {DEVICE}")

    # Try GPT-2-medium for better factual recall
    import transformer_lens
    model_name = "gpt2-medium"
    print(f"\nLoading {model_name}...")
    model = transformer_lens.HookedTransformer.from_pretrained(model_name, device=DEVICE)
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    print(f"Model: {n_layers} layers, d_model={d_model}")

    # Load and filter data - prioritize facts with higher recall
    facts = load_and_filter(model, max_examples=200)

    # Save config
    config = {
        "seed": SEED,
        "device": DEVICE,
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_facts": len(facts),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    with open(RESULTS_DIR / "data" / "enhanced_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Experiment 1: Detailed logit lens
    print("\n--- Experiment 1: Detailed Logit Lens ---")
    logit_data = run_logit_lens_detailed(model, facts)

    # Save numerical results
    np.save(RESULTS_DIR / "data" / "enhanced_layer_probs.npy", logit_data["probs"])
    np.save(RESULTS_DIR / "data" / "enhanced_layer_ranks.npy", logit_data["ranks"])

    # Identify critical layers (highest enrichment)
    enrichment = np.diff(logit_data["probs"].mean(axis=0))
    top_enrichment_layers = np.argsort(enrichment)[::-1][:3] + 1  # top 3
    print(f"Top enrichment layers: {top_enrichment_layers.tolist()}")
    print(f"Enrichment values: {enrichment[top_enrichment_layers-1].tolist()}")

    # Experiment 2: Ablation
    print("\n--- Experiment 2: Ablation Study ---")
    ablation_data = run_ablation_detailed(model, facts)

    # Identify most impactful layers
    mlp_impact = ablation_data["baseline_probs"].mean() - ablation_data["mlp_ablated_probs"].mean(axis=0)
    attn_impact = ablation_data["baseline_probs"].mean() - ablation_data["attn_ablated_probs"].mean(axis=0)
    top_mlp_layers = np.argsort(mlp_impact)[::-1][:3]
    top_attn_layers = np.argsort(attn_impact)[::-1][:3]
    print(f"Top MLP impact layers: {top_mlp_layers.tolist()}")
    print(f"Top Attention impact layers: {top_attn_layers.tolist()}")

    # Choose critical layers for deletion: union of top enrichment and top MLP impact
    critical_layers = sorted(set(top_enrichment_layers.tolist()) | set(top_mlp_layers.tolist()))[:3]
    print(f"\nCritical layers for deletion: {critical_layers}")

    # Experiment 3: Deletion
    print("\n--- Experiment 3: Deletion Geometry ---")
    deletion_results, pre_resid, post_resid = run_deletion_experiment(
        model, facts, critical_layers
    )

    # Experiment 4: Geometry visualization
    print("\n--- Creating Visualizations ---")
    create_comprehensive_figures(logit_data, ablation_data, deletion_results, model_name, n_layers)

    # Geometry: PCA of residual stream at multiple layers
    key_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    geo_results = {}

    layer_acts = {}
    for batch_start in range(0, len(facts), 64):
        batch = facts[batch_start:batch_start+64]
        prompts = [f["prompt"] for f in batch]
        tokens = model.to_tokens(prompts, prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        for l in key_layers:
            if l not in layer_acts:
                layer_acts[l] = []
            layer_acts[l].append(cache[f"blocks.{l}.hook_resid_post"][:, -1, :].cpu().numpy())
        del cache
        torch.cuda.empty_cache()

    fig, axes = plt.subplots(1, len(key_layers), figsize=(4*len(key_layers), 4))
    for idx, l in enumerate(key_layers):
        acts = np.concatenate(layer_acts[l], axis=0)
        pca = PCA(n_components=2)
        t = pca.fit_transform(acts)
        axes[idx].scatter(t[:, 0], t[:, 1], s=8, alpha=0.5, c=range(len(t)), cmap="viridis")
        axes[idx].set_title(f"Layer {l}\nPC1:{pca.explained_variance_ratio_[0]:.1%}")
        axes[idx].set_xlabel("PC1")
        axes[idx].set_ylabel("PC2")
        geo_results[f"layer_{l}"] = {
            "pc1_var": float(pca.explained_variance_ratio_[0]),
            "pc2_var": float(pca.explained_variance_ratio_[1]),
            "top10_var": float(sum(PCA(n_components=10).fit(acts).explained_variance_ratio_)),
        }
    plt.suptitle(f"Residual Stream Geometry: {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_geometry_pca.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Cross-layer cosine similarity
    n_kl = len(key_layers)
    cos_mat = np.zeros((n_kl, n_kl))
    for i in range(n_kl):
        for j in range(n_kl):
            a = np.concatenate(layer_acts[key_layers[i]], axis=0)
            b = np.concatenate(layer_acts[key_layers[j]], axis=0)
            an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
            bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
            cos_mat[i, j] = (an * bn).sum(axis=1).mean()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cos_mat, annot=True, fmt=".3f",
                xticklabels=[f"L{l}" for l in key_layers],
                yticklabels=[f"L{l}" for l in key_layers],
                cmap="RdYlBu_r", ax=ax)
    ax.set_title(f"Cross-Layer Cosine Similarity: {model_name}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig6_cross_layer_similarity.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Norm evolution
    norm_by_layer = {}
    for batch_start in range(0, len(facts), 64):
        batch = facts[batch_start:batch_start+64]
        prompts = [f["prompt"] for f in batch]
        tokens = model.to_tokens(prompts, prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        for l in range(n_layers):
            resid = cache[f"blocks.{l}.hook_resid_post"][:, -1, :]
            norms = torch.norm(resid, dim=-1).cpu().numpy()
            if l not in norm_by_layer:
                norm_by_layer[l] = []
            norm_by_layer[l].append(norms)
        del cache
        torch.cuda.empty_cache()

    norm_means = [float(np.concatenate(norm_by_layer[l]).mean()) for l in range(n_layers)]
    norm_stds = [float(np.concatenate(norm_by_layer[l]).std()) for l in range(n_layers)]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(n_layers), norm_means, "g-o", markersize=5)
    ax.fill_between(range(n_layers),
                    [m-s for m, s in zip(norm_means, norm_stds)],
                    [m+s for m, s in zip(norm_means, norm_stds)], alpha=0.2, color="green")
    ax.set_xlabel("Layer")
    ax.set_ylabel("L2 Norm")
    ax.set_title(f"Residual Stream Norm Evolution: {model_name}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig7_norm_evolution.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Save all results
    all_results = {
        "config": config,
        "logit_lens": {
            "mean_prob_per_layer": logit_data["probs"].mean(axis=0).tolist(),
            "mean_logprob_per_layer": logit_data["logprobs"].mean(axis=0).tolist(),
            "mean_rank_per_layer": logit_data["ranks"].mean(axis=0).tolist(),
            "median_rank_per_layer": np.median(logit_data["ranks"], axis=0).tolist(),
            "mean_mlp_contrib": logit_data["mlp_contribution"].mean(axis=0).tolist(),
            "mean_attn_contrib": logit_data["attn_contribution"].mean(axis=0).tolist(),
            "enrichment": enrichment.tolist(),
            "top_enrichment_layers": top_enrichment_layers.tolist(),
        },
        "ablation": {
            "baseline_mean_prob": float(ablation_data["baseline_probs"].mean()),
            "mlp_impact": mlp_impact.tolist(),
            "attn_impact": attn_impact.tolist(),
            "top_mlp_layers": top_mlp_layers.tolist(),
            "top_attn_layers": top_attn_layers.tolist(),
        },
        "deletion": deletion_results,
        "geometry": geo_results,
        "norms": {"means": norm_means, "stds": norm_stds},
        "cross_layer_cosine": cos_mat.tolist(),
    }

    with open(RESULTS_DIR / "data" / "enhanced_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*60)
    print("ALL ENHANCED EXPERIMENTS COMPLETE")
    print("="*60)
    print(f"Results: {RESULTS_DIR / 'data' / 'enhanced_results.json'}")
    print(f"Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
