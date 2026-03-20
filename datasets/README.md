# Downloaded Datasets

This directory contains datasets for the research project. Data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset 1: CounterFact

### Overview
- **Source**: HuggingFace (`azhx/counterfact`)
- **Size**: 19,728 train + 2,191 test examples
- **Format**: HuggingFace Dataset (Arrow)
- **Task**: Factual knowledge editing/deletion evaluation
- **Features**: case_id, pararel_idx, requested_rewrite, paraphrase_prompts, neighborhood_prompts, attribute_prompts, generation_prompts
- **License**: MIT (via ROME project)

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset
dataset = load_dataset("azhx/counterfact")
dataset.save_to_disk("datasets/counterfact/data")
```

**Alternative (via ROME repo):**
```bash
git clone https://github.com/kmeng01/rome.git
# CounterFact is auto-downloaded by ROME's scripts
```

### Loading the Dataset

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/counterfact/data")
```

### Sample Data

See `counterfact/samples.json` for example records. Each record contains:
- A `requested_rewrite` with subject, relation, target_true, and target_new
- Paraphrase prompts for generalization testing
- Neighborhood prompts for specificity testing

### Notes
- Standard benchmark for factual knowledge editing (ROME, MEMIT)
- Directly applicable to logit lens analysis of where facts are stored
- Use `NeelNanda/counterfact-tracing` variant for pre-computed causal tracing data

---

## Dataset 2: TOFU (Task of Fictitious Unlearning)

### Overview
- **Source**: HuggingFace (`locuslab/TOFU`, config: `full`)
- **Size**: 4,000 QA pairs (200 fictitious authors × 20 questions each)
- **Format**: HuggingFace Dataset (Arrow)
- **Task**: Machine unlearning evaluation
- **Features**: question, answer
- **License**: MIT

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset
dataset = load_dataset("locuslab/TOFU", "full")
dataset.save_to_disk("datasets/tofu/data")
```

### Loading the Dataset

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/tofu/data")
```

### Sample Data

See `tofu/samples.json` for example records. Each record is a (question, answer) pair about a fictitious author.

### Notes
- Most widely used benchmark for LLM unlearning
- Synthetic knowledge enables precise measurement of deletion
- Forget/retain sets defined by author subsets
- Models must be fine-tuned on TOFU data before unlearning experiments
- Pre-fine-tuned checkpoints available via OpenUnlearning (https://huggingface.co/open-unlearning)
