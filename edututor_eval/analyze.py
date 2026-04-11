"""Analysis and visualization of evaluation results.

Generates a Markdown report with:
- Score distributions per evaluator
- Per-dimension breakdowns
- Evaluator agreement analysis
- Flag frequency analysis
- Recommendations for improvement

Usage:
    python -m edututor_eval.analyze \
        --results results/eval_results.json \
        --output results/analysis_report.md
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from edututor_eval.datatypes import EvalResult, DimensionScore
from edututor_eval.utils import results_to_dataframe

logger = logging.getLogger(__name__)


def load_results(path: str | Path) -> list[EvalResult]:
    """Load evaluation results from JSON."""
    with open(path) as f:
        data = json.load(f)
    results = []
    for item in data:
        if "dimension_scores" in item:
            item["dimension_scores"] = [
                DimensionScore(**ds) if isinstance(ds, dict) else ds
                for ds in item["dimension_scores"]
            ]
        results.append(EvalResult(**item))
    return results


def generate_report(
    results: list[EvalResult],
    agreement_path: str | None = None,
) -> str:
    """Generate a comprehensive Markdown analysis report."""
    df = results_to_dataframe(results)
    evaluators = df["evaluator"].unique()

    sections = []
    sections.append("# EduTutor Evaluation Analysis Report\n")

    # --- Overview ---
    sections.append("## Overview\n")
    sections.append(f"- **Total evaluations**: {len(results)}")
    sections.append(f"- **Unique responses evaluated**: {df['response_id'].nunique()}")
    sections.append(f"- **Evaluators used**: {', '.join(evaluators)}")
    sections.append("")

    # --- Score Distribution per Evaluator ---
    sections.append("## Score Distribution by Evaluator\n")
    sections.append("| Evaluator | Mean | Std | Min | Median | Max | N |")
    sections.append("|-----------|------|-----|-----|--------|-----|---|")
    for ev in evaluators:
        ev_df = df[df["evaluator"] == ev]
        scores = ev_df["overall_score"]
        sections.append(
            f"| {ev} | {scores.mean():.2f} | {scores.std():.2f} | "
            f"{scores.min():.1f} | {scores.median():.1f} | "
            f"{scores.max():.1f} | {len(scores)} |"
        )
    sections.append("")

    # --- Per-Dimension Scores ---
    dim_cols = [c for c in df.columns if c.startswith("score_")]
    if dim_cols:
        sections.append("## Per-Dimension Scores\n")
        for ev in evaluators:
            ev_df = df[df["evaluator"] == ev]
            sections.append(f"### {ev}\n")
            sections.append("| Dimension | Mean | Std |")
            sections.append("|-----------|------|-----|")
            for col in dim_cols:
                dim_name = col.replace("score_", "")
                if col in ev_df.columns:
                    vals = ev_df[col].dropna()
                    if len(vals) > 0:
                        sections.append(
                            f"| {dim_name} | {vals.mean():.2f} | {vals.std():.2f} |"
                        )
            sections.append("")

    # --- Score Tier Breakdown ---
    sections.append("## Quality Tier Distribution\n")
    sections.append("| Evaluator | Low (1-2) | Medium (2-3.5) | High (3.5-5) |")
    sections.append("|-----------|-----------|----------------|--------------|")
    for ev in evaluators:
        scores = df[df["evaluator"] == ev]["overall_score"]
        low = (scores < 2.0).sum()
        med = ((scores >= 2.0) & (scores < 3.5)).sum()
        high = (scores >= 3.5).sum()
        total = len(scores)
        sections.append(
            f"| {ev} | {low} ({100*low/total:.0f}%) | "
            f"{med} ({100*med/total:.0f}%) | "
            f"{high} ({100*high/total:.0f}%) |"
        )
    sections.append("")

    # --- Flag Analysis ---
    sections.append("## Flags Detected\n")
    all_flags: list[str] = []
    for r in results:
        all_flags.extend(r.flags)

    if all_flags:
        flag_counts = Counter(all_flags)
        sections.append("| Flag | Count | % of Evaluations |")
        sections.append("|------|-------|-------------------|")
        for flag, count in flag_counts.most_common(15):
            pct = 100 * count / len(results)
            sections.append(f"| {flag} | {count} | {pct:.1f}% |")
    else:
        sections.append("No flags detected.")
    sections.append("")

    # --- Agreement Analysis ---
    if agreement_path and Path(agreement_path).exists():
        with open(agreement_path) as f:
            agreement = json.load(f)
        if agreement:
            sections.append("## Evaluator–Human Agreement\n")
            sections.append(
                "| Evaluator | Pearson r | Spearman ρ | MAE | Adjacent Agreement |"
            )
            sections.append(
                "|-----------|-----------|------------|-----|-------------------|"
            )
            for ev, metrics in agreement.items():
                sections.append(
                    f"| {ev} | {metrics['pearson_r']:.3f} | "
                    f"{metrics['spearman_r']:.3f} | "
                    f"{metrics['mean_absolute_error']:.3f} | "
                    f"{metrics['adjacent_agreement']:.1%} |"
                )
            sections.append("")

    # --- Key Findings ---
    sections.append("## Key Findings\n")

    # Find the dimension with lowest average score
    if dim_cols:
        all_dim_means = {}
        for col in dim_cols:
            vals = df[col].dropna()
            if len(vals) > 0:
                all_dim_means[col.replace("score_", "")] = vals.mean()

        if all_dim_means:
            weakest = min(all_dim_means, key=all_dim_means.get)
            strongest = max(all_dim_means, key=all_dim_means.get)
            sections.append(
                f"1. **Strongest dimension**: {strongest} "
                f"(mean {all_dim_means[strongest]:.2f})"
            )
            sections.append(
                f"2. **Weakest dimension**: {weakest} "
                f"(mean {all_dim_means[weakest]:.2f})"
            )

    # Flag prevalence
    if all_flags:
        most_common_flag = flag_counts.most_common(1)[0]
        sections.append(
            f"3. **Most common issue**: '{most_common_flag[0]}' "
            f"(detected {most_common_flag[1]} times)"
        )

    sections.append("")
    sections.append("## Recommendations\n")
    sections.append(
        "1. Focus prompt improvement efforts on the weakest scoring dimension\n"
        "2. Investigate responses flagged with quality issues for root cause patterns\n"
        "3. Use evaluator agreement data to identify where automated scoring "
        "is unreliable and human review is needed\n"
        "4. Consider the cost-quality trade-off: rule-based for pre-filtering, "
        "LLM-judge for spot-checking, learned model for production scoring"
    )

    return "\n".join(sections)


def main():
    parser = argparse.ArgumentParser(description="Analyze evaluation results")
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--agreement", type=str, default=None)
    parser.add_argument("--output", type=str, default="results/analysis_report.md")
    args = parser.parse_args()

    results = load_results(args.results)
    report = generate_report(
        results,
        agreement_path=args.agreement
        or str(Path(args.results).parent / "agreement_report.json"),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    print(f"Analysis report saved to {output_path}")


if __name__ == "__main__":
    main()
