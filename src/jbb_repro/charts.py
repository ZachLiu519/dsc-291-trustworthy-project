from __future__ import annotations

from pathlib import Path

import pandas as pd

PAPER_VICUNA_ASR = {"PAIR": 0.69, "GCG": 0.80}
METHODS = ["GCG", "PAIR"]

SUMMARY_PATHS = {
    "vicuna_harmful_llamaguard": "outputs/vicuna_vllm_jbb_subset/asr_summary_llamaguard.csv",
    "vicuna_harmful_llamaguard3": "outputs/vicuna_vllm_jbb_subset/asr_summary_llamaguard3.csv",
    "vicuna_harmful_heuristic": "outputs/vicuna_vllm_jbb_subset/asr_summary.csv",
    "gpt4o_harmful_llamaguard": "outputs/gpt4o_mini_jbb_subset/asr_summary_llamaguard.csv",
    "gpt4o_harmful_llamaguard3": "outputs/gpt4o_mini_jbb_subset/asr_summary_llamaguard3.csv",
    "gpt4o_harmful_heuristic": "outputs/gpt4o_mini_jbb_subset/asr_summary.csv",
    "vicuna_benign_refusal": "outputs/vicuna_benign_vllm_jbb_subset/refusal_summary.csv",
    "vicuna_benign_openai_refusal": "outputs/vicuna_benign_vllm_jbb_subset/refusal_summary_openai_judge.csv",
    "vicuna_benign_llamaguard": "outputs/vicuna_benign_vllm_jbb_subset/asr_summary_llamaguard.csv",
    "vicuna_benign_llamaguard3": "outputs/vicuna_benign_vllm_jbb_subset/asr_summary_llamaguard3.csv",
    "vicuna_defense_llamaguard": "outputs/vicuna_dictionary_filter_vllm_jbb_subset/asr_summary_llamaguard.csv",
    "vicuna_defense_llamaguard3": "outputs/vicuna_dictionary_filter_vllm_jbb_subset/asr_summary_llamaguard3.csv",
    "vicuna_defense_heuristic": "outputs/vicuna_dictionary_filter_vllm_jbb_subset/asr_summary.csv",
}

PRIVATE_JUDGE_SUMMARY_PATH = "reports/private_model_time_sensitivity_summary.csv"

SCORED_PATHS = {
    "vicuna_harmful": {
        "lg2": "outputs/vicuna_vllm_jbb_subset/responses_llamaguard_scored.csv",
        "lg3": "outputs/vicuna_vllm_jbb_subset/responses_llamaguard3_scored.csv",
    },
    "gpt4o_harmful": {
        "lg2": "outputs/gpt4o_mini_jbb_subset/responses_llamaguard_scored.csv",
        "lg3": "outputs/gpt4o_mini_jbb_subset/responses_llamaguard3_scored.csv",
    },
}


def generate_all_charts(repo_root: str | Path, output_dir: str | Path | None = None) -> list[Path]:
    import matplotlib.pyplot as plt

    repo_root = Path(repo_root)
    figures_dir = Path(output_dir) if output_dir is not None else repo_root / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    summaries = _load_summaries(repo_root)
    private_judge_means = _load_private_judge_means(repo_root)
    written: list[Path] = []

    written.append(_plot_harmful_asr_by_model(summaries, figures_dir, plt))
    written.append(_plot_judge_sensitivity(summaries, private_judge_means, figures_dir, plt))
    written.append(_plot_paper_comparison(summaries, figures_dir, plt))
    written.append(_plot_benign_metrics(summaries, figures_dir, plt))
    written.append(_plot_defense_comparison(summaries, figures_dir, plt, judge="llamaguard"))
    written.append(_plot_defense_comparison(summaries, figures_dir, plt, judge="llamaguard3"))
    written.append(_plot_defense_comparison(summaries, figures_dir, plt, judge="heuristic"))
    written.append(_plot_judge_disagreements(repo_root, figures_dir, plt))

    plt.close("all")
    return written


def _load_summaries(repo_root: Path) -> dict[str, pd.DataFrame]:
    summaries: dict[str, pd.DataFrame] = {}
    for name, relative in SUMMARY_PATHS.items():
        path = repo_root / relative
        if not path.exists():
            raise FileNotFoundError(f"Missing summary CSV for charting: {path}")
        summaries[name] = pd.read_csv(path)
    return summaries


def _load_private_judge_means(repo_root: Path) -> dict[tuple[str, str], float]:
    path = repo_root / PRIVATE_JUDGE_SUMMARY_PATH
    if not path.exists():
        raise FileNotFoundError(f"Missing private judge summary CSV for charting: {path}")

    summary = pd.read_csv(path)
    grouped = summary.groupby(["target_response_set", "method"])["attack_success_rate"].mean()
    return {(target, method): float(rate) for (target, method), rate in grouped.items()}


def _plot_harmful_asr_by_model(summaries: dict[str, pd.DataFrame], figures_dir: Path, plt) -> Path:
    vicuna = summaries["vicuna_harmful_llamaguard"].set_index("method")["attack_success_rate"]
    gpt4o = summaries["gpt4o_harmful_llamaguard"].set_index("method")["attack_success_rate"]
    x = range(len(METHODS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([i - width / 2 for i in x], [vicuna[m] for m in METHODS], width, label="Vicuna-13B")
    ax.bar([i + width / 2 for i in x], [gpt4o[m] for m in METHODS], width, label="GPT-4o-mini")
    ax.set_xticks(list(x), METHODS)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Attack success rate")
    ax.set_title("Harmful ASR by model (Llama-Guard-2)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _annotate_rate_bars(ax)
    return _save(fig, figures_dir / "harmful_asr_by_model_llamaguard.png")


def _plot_judge_sensitivity(
    summaries: dict[str, pd.DataFrame],
    private_judge_means: dict[tuple[str, str], float],
    figures_dir: Path,
    plt,
) -> Path:
    models = [
        (
            "Vicuna-13B",
            "vicuna_harmful",
            "vicuna_harmful_heuristic",
            "vicuna_harmful_llamaguard",
            "vicuna_harmful_llamaguard3",
        ),
        (
            "GPT-4o-mini",
            "gpt4o_mini_harmful",
            "gpt4o_harmful_heuristic",
            "gpt4o_harmful_llamaguard",
            "gpt4o_harmful_llamaguard3",
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for ax, (label, private_target_key, heuristic_key, lg2_key, lg3_key) in zip(axes, models, strict=True):
        heuristic = summaries[heuristic_key].set_index("method")["attack_success_rate"]
        lg2 = summaries[lg2_key].set_index("method")["attack_success_rate"]
        lg3 = summaries[lg3_key].set_index("method")["attack_success_rate"]
        private_judge = {method: private_judge_means[(private_target_key, method)] for method in METHODS}
        x = range(len(METHODS))
        width = 0.18
        ax.bar([i - 1.5 * width for i in x], [lg2[m] for m in METHODS], width, label="Llama-Guard-2")
        ax.bar([i - 0.5 * width for i in x], [private_judge[m] for m in METHODS], width, label="GPT-4o-mini judge")
        ax.bar([i + 0.5 * width for i in x], [lg3[m] for m in METHODS], width, label="Llama-Guard-3")
        ax.bar([i + 1.5 * width for i in x], [heuristic[m] for m in METHODS], width, label="Heuristic")
        ax.set_xticks(list(x), METHODS)
        ax.set_ylim(0, 1.05)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
        _annotate_rate_bars(ax)

    fig.supylabel("Attack success rate")
    fig.suptitle("Judge sensitivity on harmful prompts")
    fig.tight_layout()
    return _save(fig, figures_dir / "harmful_asr_judge_sensitivity.png")


def _plot_paper_comparison(summaries: dict[str, pd.DataFrame], figures_dir: Path, plt) -> Path:
    ours = summaries["vicuna_harmful_llamaguard"].set_index("method")["attack_success_rate"]
    x = range(len(METHODS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(
        [i - width / 2 for i in x],
        [PAPER_VICUNA_ASR[m] for m in METHODS],
        width,
        label="Paper (Vicuna)",
    )
    ax.bar([i + width / 2 for i in x], [ours[m] for m in METHODS], width, label="Our run (Vicuna)")
    ax.set_xticks(list(x), METHODS)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Attack success rate")
    ax.set_title("Vicuna harmful ASR vs paper (Llama-Guard-2)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _annotate_rate_bars(ax)
    return _save(fig, figures_dir / "vicuna_asr_vs_paper_llamaguard.png")


def _plot_benign_metrics(summaries: dict[str, pd.DataFrame], figures_dir: Path, plt) -> Path:
    refusal = float(summaries["vicuna_benign_refusal"]["refusal_rate"].iloc[0])
    openai_refusal = float(summaries["vicuna_benign_openai_refusal"]["refusal_rate"].iloc[0])
    unsafe_lg2 = float(summaries["vicuna_benign_llamaguard"]["attack_success_rate"].iloc[0])
    unsafe_lg3 = float(summaries["vicuna_benign_llamaguard3"]["attack_success_rate"].iloc[0])
    labels = ["Heuristic\nrefusal rate", "GPT-4o-mini\nrefusal rate", "LG-2\nunsafe rate", "LG-3\nunsafe rate"]
    values = [refusal, openai_refusal, unsafe_lg2, unsafe_lg3]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(labels, values, color=["#4C78A8", "#72B7B2", "#F58518", "#E45756"])
    ax.set_ylim(0, max(0.25, max(values) * 1.25, 0.12))
    ax.set_ylabel("Rate")
    ax.set_title("Vicuna-13B benign behavior metrics (n=30)")
    ax.grid(axis="y", alpha=0.3)
    _annotate_rate_bars(ax)
    return _save(fig, figures_dir / "vicuna_benign_metrics.png")


def _plot_defense_comparison(
    summaries: dict[str, pd.DataFrame],
    figures_dir: Path,
    plt,
    *,
    judge: str,
) -> Path:
    if judge == "llamaguard":
        baseline = summaries["vicuna_harmful_llamaguard"].set_index("method")["attack_success_rate"]
        defended = summaries["vicuna_defense_llamaguard"].set_index("method")["attack_success_rate"]
        title = "Defense comparison (Llama-Guard-2)"
        filename = "defense_asr_llamaguard.png"
    elif judge == "llamaguard3":
        baseline = summaries["vicuna_harmful_llamaguard3"].set_index("method")["attack_success_rate"]
        defended = summaries["vicuna_defense_llamaguard3"].set_index("method")["attack_success_rate"]
        title = "Defense comparison (Llama-Guard-3)"
        filename = "defense_asr_llamaguard3.png"
    else:
        baseline = summaries["vicuna_harmful_heuristic"].set_index("method")["attack_success_rate"]
        defended = summaries["vicuna_defense_heuristic"].set_index("method")["attack_success_rate"]
        title = "Defense comparison (heuristic judge)"
        filename = "defense_asr_heuristic.png"

    x = range(len(METHODS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([i - width / 2 for i in x], [baseline[m] for m in METHODS], width, label="No defense")
    ax.bar([i + width / 2 for i in x], [defended[m] for m in METHODS], width, label="dictionary_filter")
    ax.set_xticks(list(x), METHODS)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Attack success rate")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _annotate_rate_bars(ax)
    return _save(fig, figures_dir / filename)


def _plot_judge_disagreements(repo_root: Path, figures_dir: Path, plt) -> Path:
    model_labels = []
    heuristic_counts = []
    lg3_counts = []
    for label, relatives in SCORED_PATHS.items():
        lg2 = pd.read_csv(repo_root / relatives["lg2"])
        lg3 = pd.read_csv(repo_root / relatives["lg3"])
        model_labels.append("Vicuna-13B" if label == "vicuna_harmful" else "GPT-4o-mini")
        if "heuristic_jailbroken" in lg2.columns:
            heuristic_counts.append(int((lg2["jailbroken"] != lg2["heuristic_jailbroken"]).sum()))
        else:
            heuristic_counts.append(0)
        merged = lg2.merge(lg3, on=["prompt", "response"], suffixes=("_lg2", "_lg3"))
        lg3_counts.append(int((merged["jailbroken_lg2"] != merged["jailbroken_lg3"]).sum()))

    x = range(len(model_labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(
        [i - width / 2 for i in x],
        heuristic_counts,
        width,
        label="LG-2 vs heuristic",
        color="#4C78A8",
    )
    ax.bar(
        [i + width / 2 for i in x],
        lg3_counts,
        width,
        label="LG-2 vs LG-3",
        color="#F58518",
    )
    ax.set_xticks(list(x), model_labels)
    ymax = max([*heuristic_counts, *lg3_counts], default=1) * 1.2
    ax.set_ylim(0, max(ymax, 12))
    ax.set_ylabel("Prompt count")
    ax.set_title("Judge disagreement counts")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _annotate_count_bars(ax)
    return _save(fig, figures_dir / "judge_disagreement_counts.png")


def _annotate_rate_bars(ax) -> None:
    for patch in ax.patches:
        value = patch.get_height()
        percent = value * 100
        if percent < 10:
            continue
        label = _format_percent(percent)
        ax.annotate(
            label,
            xy=(patch.get_x() + patch.get_width() / 2, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def _annotate_count_bars(ax) -> None:
    for patch in ax.patches:
        value = int(patch.get_height())
        if value < 10:
            continue
        ax.annotate(
            str(value),
            xy=(patch.get_x() + patch.get_width() / 2, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def _format_percent(percent: float) -> str:
    if float(percent).is_integer():
        return f"{int(percent)}%"
    return f"{percent:.1f}%"


def _save(fig, path: Path) -> Path:
    import matplotlib.pyplot as plt

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path
