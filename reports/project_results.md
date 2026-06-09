# DSC 291 JailbreakBench Reproduction Results

## Scope Completed

This run completes the proposal's scoped JailbreakBench reproduction on a 30-behavior stratified harmful subset, the matched 30-behavior benign subset, one open-weight local model, one closed-source API model, one defense baseline, and qualitative examples.

The local model is `lmsys/vicuna-13b-v1.5` served with vLLM on an A100. The closed-source comparison model is `gpt-4o-mini`. PAIR and GCG prompts are the public JailbreakBench artifacts loaded for `vicuna-13b-v1.5`.

## Primary Judge

Primary scoring uses the official Hugging Face model `meta-llama/Meta-Llama-Guard-2-8B`. All primary judge output files record this model id in the `judge_model` column.

## Harmful ASR

Primary Llama-Guard-2 ASR:

| Model | Defense | Method | n | ASR |
| --- | --- | --- | ---: | ---: |
| `lmsys/vicuna-13b-v1.5` | none | GCG | 29 | 0.8621 |
| `lmsys/vicuna-13b-v1.5` | none | PAIR | 25 | 0.6800 |
| `gpt-4o-mini` | none | GCG | 29 | 0.1034 |
| `gpt-4o-mini` | none | PAIR | 25 | 0.2400 |

Llama-Guard-3 sensitivity check:

| Model | Defense | Method | n | ASR |
| --- | --- | --- | ---: | ---: |
| `lmsys/vicuna-13b-v1.5` | none | GCG | 29 | 0.8966 |
| `lmsys/vicuna-13b-v1.5` | none | PAIR | 25 | 0.9200 |
| `gpt-4o-mini` | none | GCG | 29 | 0.1724 |
| `gpt-4o-mini` | none | PAIR | 25 | 0.3200 |

Proposal-aligned private API judge sensitivity check (`gpt-4o-mini` as judge, completed-run mean):

| Model | Defense | Method | n | Mean ASR |
| --- | --- | --- | ---: | ---: |
| `lmsys/vicuna-13b-v1.5` | none | GCG | 29 | 0.8966 |
| `lmsys/vicuna-13b-v1.5` | none | PAIR | 25 | 0.9200 |
| `gpt-4o-mini` | none | GCG | 29 | 0.1293 |
| `gpt-4o-mini` | none | PAIR | 25 | 0.3550 |

Heuristic ASR sensitivity check:

| Model | Defense | Method | n | ASR |
| --- | --- | --- | ---: | ---: |
| `lmsys/vicuna-13b-v1.5` | none | GCG | 29 | 0.8966 |
| `lmsys/vicuna-13b-v1.5` | none | PAIR | 25 | 0.8800 |
| `gpt-4o-mini` | none | GCG | 29 | 0.2069 |
| `gpt-4o-mini` | none | PAIR | 25 | 0.4400 |

The gap between Llama-Guard-2, the proposal-aligned `gpt-4o-mini` API judge, Llama-Guard-3, and the string heuristic confirms the proposal's concern that measured ASR is judge-sensitive. On the regenerated response set, Llama-Guard-2 disagreed with the heuristic on 10 Vicuna prompts and 8 GPT-4o-mini prompts, while Llama-Guard-2 disagreed with Llama-Guard-3 on 9 Vicuna prompts and 4 GPT-4o-mini prompts. The largest LG-2 vs LG-3 gap is on Vicuna PAIR (+24.0 points), and the largest LG-2 vs GPT-4o-mini-judge gap is also on Vicuna PAIR (+24.0 points).

## Private API Judge Time Sensitivity

We also re-scored the same saved harmful responses with `gpt-4o-mini` as an API judge across scheduled runs at different local times. These runs provide the proposal-aligned private-model judge sensitivity check above and test **API judge stability over time**, not target-model generation drift: the prompts and model responses were fixed, and only the private API judge call was repeated.

Scheduled run reliability:

| Scheduled run | Completed | Failed |
| --- | ---: | ---: |
| All attempts | 8 | 2 |

Both failures were OpenAI API read timeouts, not missing credentials or cron failures.

Completed-run ASR ranges:

| Response set | Method | Completed runs | Min ASR | Max ASR | Mean ASR |
| --- | --- | ---: | ---: | ---: | ---: |
| Vicuna harmful | GCG | 8 | 0.8966 | 0.8966 | 0.8966 |
| Vicuna harmful | PAIR | 8 | 0.9200 | 0.9200 | 0.9200 |
| GPT-4o-mini harmful | GCG | 8 | 0.1034 | 0.1379 | 0.1293 |
| GPT-4o-mini harmful | PAIR | 8 | 0.3200 | 0.3600 | 0.3550 |

The main takeaway is that we do not see a clear time-of-day trend in the private API judge. Vicuna labels were identical in every completed run. GPT-4o-mini labels varied by one example for GCG (`1 / 29`, about 3.4 points) and one example for PAIR (`1 / 25`, 4 points), which looks more like small API-judge nondeterminism than systematic daypart sensitivity.

## Benign Refusal

Vicuna-13B benign behavior results:

| Model | n | Heuristic Refusal Rate | GPT-4o-mini Refusal Rate | LG-2 Unsafe Rate | LG-3 Unsafe Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lmsys/vicuna-13b-v1.5` | 30 | 0.0000 | 0.0333 | 0.1000 | 0.0000 |

The refusal heuristic did not detect over-refusal on the sampled benign prompts, while the GPT-4o-mini refusal judge flagged 1 of 30 responses. Llama-Guard is not a refusal judge, so the unsafe rate is reported as an auxiliary safety check rather than a benign refusal metric. Llama-Guard-3 labeled all 30 benign responses safe, while Llama-Guard-2 flagged 3 as unsafe.

## Defense Comparison

The implemented defense baseline is `dictionary_filter`, a prompt preprocessing filter that removes non-dictionary tokens before generation. It is intentionally simple and reproducible, and it is most relevant to suffix-heavy attacks such as GCG.

Primary Llama-Guard-2 defense comparison for Vicuna-13B:

| Defense | Method | n | ASR |
| --- | --- | ---: | ---: |
| none | GCG | 29 | 0.8621 |
| dictionary_filter | GCG | 29 | 0.2069 |
| none | PAIR | 25 | 0.6800 |
| dictionary_filter | PAIR | 25 | 0.6000 |

Llama-Guard-3 defense comparison for Vicuna-13B:

| Defense | Method | n | ASR |
| --- | --- | ---: | ---: |
| none | GCG | 29 | 0.8966 |
| dictionary_filter | GCG | 29 | 0.1724 |
| none | PAIR | 25 | 0.9200 |
| dictionary_filter | PAIR | 25 | 0.8400 |

Heuristic defense comparison:

| Defense | Method | n | ASR |
| --- | --- | ---: | ---: |
| none | GCG | 29 | 0.8966 |
| dictionary_filter | GCG | 29 | 0.2069 |
| none | PAIR | 25 | 0.8800 |
| dictionary_filter | PAIR | 25 | 0.8400 |

The defense reduces GCG ASR under both judges and gives a smaller reduction for PAIR under the primary judge. This matches the defense's mechanism: it mainly removes unusual adversarial suffix tokens rather than semantically normal natural-language attacks.

## Output Inventory

Git-tracked outputs include summary CSVs, figures, and reports. Raw response JSONL files and full scored CSV files are intentionally not committed because they may contain harmful content. They can be regenerated from the README pipeline.

The main committed artifacts are:

- `outputs/vicuna_vllm_jbb_subset/asr_summary_llamaguard.csv`
- `outputs/vicuna_vllm_jbb_subset/asr_summary_llamaguard3.csv`
- `outputs/vicuna_vllm_jbb_subset/asr_summary.csv`
- `outputs/vicuna_benign_vllm_jbb_subset/refusal_summary.csv`
- `outputs/vicuna_benign_vllm_jbb_subset/refusal_summary_openai_judge.csv`
- `outputs/vicuna_benign_vllm_jbb_subset/asr_summary_llamaguard3.csv`
- `outputs/vicuna_dictionary_filter_vllm_jbb_subset/asr_summary_llamaguard.csv`
- `outputs/vicuna_dictionary_filter_vllm_jbb_subset/asr_summary_llamaguard3.csv`
- `outputs/gpt4o_mini_jbb_subset/asr_summary_llamaguard.csv`
- `outputs/gpt4o_mini_jbb_subset/asr_summary_llamaguard3.csv`
- `reports/project_results.md`
- `reports/jailbreakbench_report.md`
- `reports/private_model_time_sensitivity_runs.csv`
- `reports/private_model_time_sensitivity_summary.csv`
- `reports/figures/private_model_time_sensitivity.png`
- `reports/figures/`
- Vicuna qualitative examples: `reports/qualitative_examples_vicuna.md`
- GPT-4o-mini qualitative examples: `reports/qualitative_examples_gpt4o_mini.md`

The GPT-4o-mini run used 3,756 prompt tokens and 5,180 completion tokens across 54 responses.

## Reproduction Notes

This project requires Python 3.11 because `jailbreakbench` requires Python `<3.12`. We pin `transformers<4.39` and `litellm<1.30` for the default JailbreakBench stack, but Llama-Guard-3 scoring required a one-off upgrade to `transformers==4.43.3` for llama3 rope-scaling support. Vicuna-13B fp16 did not fit on a 23GB L4 GPU, so the full local run used an A100 80GB.

## Limitations

The behavior subset is intentionally small for course-project cost and time constraints. PAIR/GCG artifacts were loaded for `vicuna-13b-v1.5` and reused for GPT-4o-mini, so the API-model comparison measures transfer of the same prompt set rather than target-specific attack optimization.
