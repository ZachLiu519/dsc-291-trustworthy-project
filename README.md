# DSC 291 JailbreakBench Reproduction

Reproduction code for the course project on **JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models**.

**Preferred entrypoint:**
- **Google Colab:** [`notebooks/colab_driver.ipynb`](notebooks/colab_driver.ipynb) — uses a Python 3.11 venv and runs the repo scripts via subprocess
- **Local Jupyter (Python 3.11 kernel):** [`notebooks/reproduce_jailbreakbench.ipynb`](notebooks/reproduce_jailbreakbench.ipynb) — imports workflow functions directly

This README is the non-interactive runbook. For the numbers we obtained, see [`reports/project_results.md`](reports/project_results.md). For redacted qualitative examples, see [`reports/qualitative_examples_vicuna.md`](reports/qualitative_examples_vicuna.md) and [`reports/qualitative_examples_gpt4o_mini.md`](reports/qualitative_examples_gpt4o_mini.md).
## Interfaces

| Interface | Use when |
| --- | --- |
| [`notebooks/colab_driver.ipynb`](notebooks/colab_driver.ipynb) | Google Colab with GPU (official Colab workaround) |
| [`notebooks/reproduce_jailbreakbench.ipynb`](notebooks/reproduce_jailbreakbench.ipynb) | Local Jupyter with Python 3.11 kernel |
| `scripts/*.py` | Headless Colab terminal runs or README command-line reproduction |
| `src/jbb_repro/workflows.py` | Scoring, API runs, qualitative examples |
| `src/jbb_repro/vllm_workflows.py` | GPU Vicuna generation workflows |
| `src/jbb_repro/*` modules | Custom experiments and unit tests |

## What We Reproduce

We reproduce a **scoped subset** of JailbreakBench on Vicuna-13B and compare against GPT-4o-mini:

| Experiment | Paper target | Our setup |
| --- | --- | --- |
| Harmful ASR (PAIR, GCG) | Vicuna-13B on full JBB benchmark | 30-behavior stratified harmful subset, public PAIR/GCG artifacts for `vicuna-13b-v1.5` |
| Benign refusal | Over-refusal on benign behaviors | Matched 30-behavior benign subset |
| Defense comparison | Paper evaluates multiple defenses | One simple baseline: `dictionary_filter` |
| Closed-source model | GPT-3.5 / GPT-4 in paper | `gpt-4o-mini`, reusing the Vicuna attack prompts |
| Judge | Paper uses Llama-3-70B | We use **Llama-Guard-2** (`meta-llama/Meta-Llama-Guard-2-8B`) as primary and **GPT-4o-mini as a private API judge** for the proposal-aligned sensitivity check |
| Additional judge checks | Not a paper target | Llama-Guard-3 after peer feedback about why LG2 was selected; string heuristic as a lightweight diagnostic |
| Private API judge stability | Not a paper target | Repeated `gpt-4o-mini` judge calls across scheduled local times on fixed saved responses |

### Comparison to paper (Vicuna harmful ASR)

| Attack | Paper Vicuna ASR | Our Vicuna ASR (Llama-Guard-2) |
| --- | ---: | ---: |
| PAIR | 69% | 68.0% |
| GCG | 80% | 86.2% |

PAIR is very close. GCG is directionally aligned but slightly higher. Differences come from our smaller subset, judge choice, and artifact availability (not every sampled behavior has both PAIR and GCG prompts, so `n` is 25 or 29 rather than 30).

## Prerequisites

- **Python**: `>=3.10,<3.12` (required by `jailbreakbench`)
- **GPU runtime** (Linux/CUDA) for local Vicuna generation:
  - Tested on **A100 80GB** with Vicuna-13B fp16
  - Vicuna-13B fp16 **OOM'd on a 23GB L4**; use `configs/vicuna7b_vllm.yaml` only for smoke tests on smaller GPUs
- **Hugging Face access**:
  - Accept the license for `lmsys/vicuna-13b-v1.5`
  - Request and accept access for `meta-llama/Meta-Llama-Guard-2-8B`
  - Authenticate with `huggingface-cli login` or set `HF_TOKEN`
- **OpenAI API key** for the GPT-4o-mini comparison (`OPENAI_API_KEY`)
- **Disk / cache**: model weights download on first run; set `HF_HOME` if you want a custom cache directory

## Setup

All commands below assume the repo root and `PYTHONPATH=src`.

### macOS / local dev (tests only)

vLLM is Linux/CUDA-only. On macOS you can install dependencies and run unit tests, but not Vicuna generation.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
PYTHONPATH=src pytest tests -q
```

### Colab / Linux GPU (full reproduction)

```bash
python3.10 -m venv .venv310
source .venv310/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,vllm]"
```

Optional smoke test on a smaller GPU before the full Vicuna-13B run:

```bash
PYTHONPATH=src python scripts/run_vllm_local.py \
  --config configs/vicuna7b_vllm.yaml \
  --limit 2
```

## Environment Variables

Scripts that call OpenAI or download gated Hugging Face models read secrets from the environment.

**Colab:** create `/content/.env`:

```bash
OPENAI_API_KEY=sk-...
HF_TOKEN=hf_...
```

**Local / other:** create a project-root `.env` or export variables in your shell before running:

```bash
export OPENAI_API_KEY=sk-...
export HF_TOKEN=hf_...
# optional
export HF_HOME=/path/to/hf_cache
```

Do not commit `.env` files.

## Full Reproduction Pipeline

Run these steps in order on a CUDA machine with the GPU setup above. Each config uses `sample_size: 30` and `seed: 291`, so runs are deterministic.

### Step 0 — Verify wiring (no model load)

Build the sampled behavior subset and PAIR/GCG artifact prompts without loading Vicuna:

```bash
PYTHONPATH=src python scripts/run_vllm_local.py \
  --config configs/vicuna_vllm.yaml \
  --limit 4 \
  --dry-run
```

Expected output: `outputs/vicuna_vllm_jbb_subset/attack_prompts.jsonl` with 4 prompts.

### Step 1 — Vicuna-13B harmful attacks (PAIR + GCG)

```bash
PYTHONPATH=src python scripts/run_vllm_local.py \
  --config configs/vicuna_vllm.yaml
```

Expected output:
- `outputs/vicuna_vllm_jbb_subset/sampled_behaviors.csv`
- `outputs/vicuna_vllm_jbb_subset/attack_prompts.jsonl` (~54 prompts)
- `outputs/vicuna_vllm_jbb_subset/responses.jsonl` (~54 responses)

### Step 2 — Score Vicuna harmful responses

**Primary judge (Llama-Guard-2):**

```bash
PYTHONPATH=src python scripts/score_llamaguard.py \
  --responses outputs/vicuna_vllm_jbb_subset/responses.jsonl
```

Expected primary ASR in `outputs/vicuna_vllm_jbb_subset/asr_summary_llamaguard.csv`:

| method | n | attack_success_rate |
| --- | ---: | ---: |
| GCG | 29 | ~0.86 |
| PAIR | 25 | ~0.68 |

**Proposal-aligned private API judge sensitivity check:**

Score the same fixed responses with GPT-4o-mini as a judge:

```bash
PYTHONPATH=src python scripts/score_openai_judge.py \
  --responses outputs/vicuna_vllm_jbb_subset/responses.jsonl \
  --output-dir outputs/api_judge_sensitivity/manual/vicuna_harmful
```

The committed report uses the completed-run mean from the scheduled private judge runs, so the table reports both judge sensitivity and repeated-call stability.

**Heuristic diagnostic check:**

```bash
PYTHONPATH=src python scripts/score_outputs.py \
  --responses outputs/vicuna_vllm_jbb_subset/responses.jsonl
```

Expected heuristic ASR is slightly higher than Llama-Guard-2 (see `reports/project_results.md`).

**Additional judge sensitivity: Llama-Guard-3-8B**

After Hugging Face access is approved for `meta-llama/Llama-Guard-3-8B`, score the same saved responses without regenerating model outputs:

```bash
PYTHONPATH=src python scripts/run_llamaguard3_sensitivity.sh
```

This writes `responses_llamaguard3_scored.csv` and `asr_summary_llamaguard3.csv` next to each response file, so the Llama-Guard-2 outputs remain intact.

**Private API judge time sensitivity**

To test whether repeated private API judge calls are stable across times of day, we re-scored the same saved harmful responses with `gpt-4o-mini` as a judge across scheduled local runs. These scheduled runs are also the source of the GPT-4o-mini judge column in the judge-sensitivity report. Aggregated, safe-to-commit results live in:

- `reports/private_model_time_sensitivity_runs.csv`
- `reports/private_model_time_sensitivity_summary.csv`
- `reports/figures/private_model_time_sensitivity.png`

The raw repeated scored CSVs are not committed; they can be regenerated from the saved response files with `scripts/score_openai_judge.py`.

### Step 3 — Vicuna-13B benign behaviors

```bash
PYTHONPATH=src python scripts/run_vllm_benign.py \
  --config configs/vicuna_benign_vllm.yaml
```

Expected output: `outputs/vicuna_benign_vllm_jbb_subset/responses.jsonl` (30 responses).

Score benign outputs:

```bash
PYTHONPATH=src python scripts/score_outputs.py \
  --responses outputs/vicuna_benign_vllm_jbb_subset/responses.jsonl \
  --benign

PYTHONPATH=src python scripts/score_llamaguard.py \
  --responses outputs/vicuna_benign_vllm_jbb_subset/responses.jsonl

PYTHONPATH=src python scripts/score_openai_refusal_judge.py \
  --responses outputs/vicuna_benign_vllm_jbb_subset/responses.jsonl
```

Expected: heuristic refusal rate ~0.0; GPT-4o-mini refusal rate ~0.03; Llama-Guard-2 unsafe rate ~0.10; Llama-Guard-3 unsafe rate ~0.0 on benign prompts.

### Step 4 — Dictionary-filter defense on Vicuna-13B

```bash
PYTHONPATH=src python scripts/run_vllm_local.py \
  --config configs/vicuna_dictionary_filter_vllm.yaml \
  --defense dictionary_filter
```

Expected output: `outputs/vicuna_dictionary_filter_vllm_jbb_subset/responses.jsonl`.

Score with Llama-Guard-2:

```bash
PYTHONPATH=src python scripts/score_llamaguard.py \
  --responses outputs/vicuna_dictionary_filter_vllm_jbb_subset/responses.jsonl
```

Expected primary ASR: GCG drops to ~0.21; PAIR drops modestly to ~0.60 (defense mainly removes adversarial suffix tokens).

### Step 5 — GPT-4o-mini on the same attack prompts

Reuse the Vicuna harmful prompts so the API model sees the same PAIR/GCG artifacts:

```bash
PYTHONPATH=src python scripts/run_openai_model.py \
  --config configs/gpt4o_mini_jbb.yaml \
  --prompts outputs/vicuna_vllm_jbb_subset/attack_prompts.jsonl
```

Expected output: `outputs/gpt4o_mini_jbb_subset/responses.jsonl` (~54 responses).

Score:

```bash
PYTHONPATH=src python scripts/score_llamaguard.py \
  --responses outputs/gpt4o_mini_jbb_subset/responses.jsonl

PYTHONPATH=src python scripts/score_outputs.py \
  --responses outputs/gpt4o_mini_jbb_subset/responses.jsonl
```

Expected primary ASR: GCG ~0.17, PAIR ~0.28.

### Step 6 — Qualitative examples (optional)

Extract representative success / failure / judge-disagreement cases:

```bash
PYTHONPATH=src python scripts/extract_qualitative_examples.py \
  --scored outputs/vicuna_vllm_jbb_subset/responses_llamaguard_scored.csv \
  --output reports/qualitative_examples_vicuna.md

PYTHONPATH=src python scripts/extract_qualitative_examples.py \
  --scored outputs/gpt4o_mini_jbb_subset/responses_llamaguard_scored.csv \
  --output reports/qualitative_examples_gpt4o_mini.md
```

### Step 7 — Generate charts (optional)

After the summary CSVs exist, render PNG figures for the proposal metrics:

```bash
PYTHONPATH=src python scripts/generate_charts.py
```

Figures are written to `reports/figures/`:

| Figure | Metric |
| --- | --- |
| `harmful_asr_by_model_llamaguard.png` | Harmful ASR, Vicuna vs GPT-4o-mini |
| `harmful_asr_judge_sensitivity.png` | LG-2 vs GPT-4o-mini judge vs LG-3 vs heuristic ASR |
| `vicuna_asr_vs_paper_llamaguard.png` | Vicuna ASR vs paper baseline |
| `vicuna_benign_metrics.png` | Benign heuristic/GPT-4o-mini refusal + LG-2/LG-3 unsafe rate |
| `defense_asr_llamaguard.png` | Defense comparison (Llama-Guard-2) |
| `defense_asr_llamaguard3.png` | Defense comparison (Llama-Guard-3) |
| `defense_asr_heuristic.png` | Defense comparison (heuristic judge) |
| `judge_disagreement_counts.png` | LG-2 vs heuristic and LG-2 vs LG-3 disagreements |

## Output Inventory

After a full run, the key artifacts are:

| Path | Description |
| --- | --- |
| `outputs/vicuna_vllm_jbb_subset/responses.jsonl` | Vicuna harmful generations |
| `outputs/vicuna_vllm_jbb_subset/asr_summary_llamaguard.csv` | Primary harmful ASR table |
| `outputs/vicuna_benign_vllm_jbb_subset/responses.jsonl` | Vicuna benign generations |
| `outputs/vicuna_benign_vllm_jbb_subset/refusal_summary.csv` | Benign heuristic refusal rate |
| `outputs/vicuna_benign_vllm_jbb_subset/refusal_summary_openai_judge.csv` | Benign GPT-4o-mini refusal judge rate |
| `outputs/vicuna_dictionary_filter_vllm_jbb_subset/asr_summary_llamaguard.csv` | Defense comparison ASR |
| `outputs/gpt4o_mini_jbb_subset/asr_summary_llamaguard.csv` | GPT-4o-mini ASR |
| `reports/private_model_time_sensitivity_*.csv` | Aggregated scheduled private API judge stability results |
| `reports/project_results.md` | Consolidated results write-up |
| `reports/figures/*.png` | Generated metric charts |

Git tracks summary CSVs and reports. Raw `responses.jsonl` and full scored CSVs are gitignored because they may contain harmful content; regenerate them with the steps above.

## Config Reference

| Config | Purpose |
| --- | --- |
| `configs/vicuna_vllm.yaml` | Vicuna-13B harmful PAIR/GCG |
| `configs/vicuna_benign_vllm.yaml` | Vicuna-13B benign behaviors |
| `configs/vicuna_dictionary_filter_vllm.yaml` | Vicuna-13B harmful + defense output dir (pass `--defense dictionary_filter`) |
| `configs/gpt4o_mini_jbb.yaml` | GPT-4o-mini API run |
| `configs/vicuna7b_vllm.yaml` | Vicuna-7B smoke test only (not a paper target model) |
| `configs/vicuna13b_awq_vllm.yaml` | Quantized Vicuna-13B option if fp16 does not fit |

## Limitations

- **Subset size**: 30 behaviors per split for cost/time, not the full JBB benchmark.
- **Judge difference**: paper uses Llama-3-70B; we use Llama-Guard-2 per our proposal.
- **Prompt transfer**: GPT-4o-mini is evaluated on Vicuna-optimized PAIR/GCG artifacts, not target-specific attacks.
- **Artifact coverage**: some behaviors lack PAIR or GCG artifacts, so per-method `n` may be below 30.
- **Platform**: generation requires Linux/CUDA; macOS can run tests and scoring only.

## Tests

Optional sanity check after installing with the `[dev]` extra:

```bash
PYTHONPATH=src pytest tests -q
```

Expected: `19 passed`.
