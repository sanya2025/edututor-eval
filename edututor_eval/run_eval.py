"""Main evaluation pipeline: run all evaluators on a dataset.

This is the entry point for running evaluations. It loads data,
runs each evaluator, computes agreement metrics against human labels,
and saves structured results.

Usage:
    python -m edututor_eval.run_eval \
        --data data/synthetic_responses.json \
        --rubric configs/rubric_default.yaml \
        --output results/eval_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from edututor_eval.datatypes import TutorResponse, EvalResult
from edututor_eval.rubrics import load_rubric
from edututor_eval.metrics.rule_based import RuleBasedEvaluator
from edututor_eval.metrics.llm_judge import LLMJudgeEvaluator
from edututor_eval.metrics.learned import LearnedEvaluator
from edututor_eval.utils import (
    setup_logging,
    load_responses,
    save_results,
    results_to_dataframe,
    compute_agreement,
)

logger = logging.getLogger(__name__)


def run_evaluation(
    responses: list[TutorResponse],
    rubric_path: str,
    evaluators: list[str] | None = None,
    model_path: str | None = None,
    llm_provider: str = "openai",
    llm_model: str = "gpt-4o",
) -> dict[str, list[EvalResult]]:
    """Run specified evaluators on all responses.

    Returns dict mapping evaluator name to list of results.
    """
    if evaluators is None:
        evaluators = ["rule_based", "llm_judge", "learned"]

    rubric = load_rubric(rubric_path)
    all_results: dict[str, list[EvalResult]] = {}

    for eval_name in evaluators:
        logger.info("Running evaluator: %s on %d responses", eval_name, len(responses))
        start = time.time()

        if eval_name == "rule_based":
            evaluator = RuleBasedEvaluator()
        elif eval_name == "llm_judge":
            evaluator = LLMJudgeEvaluator(
                rubric=rubric, provider=llm_provider, model=llm_model
            )
        elif eval_name == "learned":
            evaluator = LearnedEvaluator(model_path=model_path)
        else:
            logger.warning("Unknown evaluator: %s", eval_name)
            continue

        results = []
        for response in responses:
            result = evaluator.evaluate(response)
            results.append(result)

        elapsed = time.time() - start
        all_results[eval_name] = results
        logger.info(
            "  %s: %d results in %.1fs (%.0f ms/response)",
            eval_name, len(results), elapsed,
            (elapsed / len(responses)) * 1000 if responses else 0,
        )

    return all_results


def compute_evaluator_agreement(
    responses: list[TutorResponse],
    all_results: dict[str, list[EvalResult]],
) -> dict:
    """Compare each evaluator's scores against human labels."""
    human_scores = [r.human_score for r in responses if r.human_score is not None]
    if not human_scores:
        logger.warning("No human scores available for agreement computation")
        return {}

    agreement_report = {}
    for eval_name, results in all_results.items():
        eval_scores = [r.overall_score for r in results]

        # Align: only compare where we have both
        paired = [
            (h, e) for h, e, r in zip(human_scores, eval_scores, responses)
            if r.human_score is not None
        ]
        if not paired:
            continue

        h_scores, e_scores = zip(*paired)
        metrics = compute_agreement(list(h_scores), list(e_scores))
        agreement_report[eval_name] = metrics
        logger.info(
            "  %s vs human: pearson=%.3f, spearman=%.3f, MAE=%.3f",
            eval_name, metrics["pearson_r"], metrics["spearman_r"],
            metrics["mean_absolute_error"],
        )

    return agreement_report


def main():
    parser = argparse.ArgumentParser(description="Run AI tutor response evaluation")
    parser.add_argument("--data", type=str, required=True, help="Path to response data JSON")
    parser.add_argument("--rubric", type=str, default="configs/rubric_default.yaml")
    parser.add_argument("--output", type=str, default="results/eval_results.json")
    parser.add_argument(
        "--evaluators", nargs="+",
        default=["rule_based", "llm_judge", "learned"],
        choices=["rule_based", "llm_judge", "learned"],
    )
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--llm_provider", type=str, default="openai")
    parser.add_argument("--llm_model", type=str, default="gpt-4o")
    args = parser.parse_args()

    setup_logging()

    # Load data
    logger.info("Loading responses from %s", args.data)
    responses = load_responses(args.data)
    logger.info("Loaded %d responses", len(responses))

    # Run evaluations
    all_results = run_evaluation(
        responses=responses,
        rubric_path=args.rubric,
        evaluators=args.evaluators,
        model_path=args.model_path,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
    )

    # Compute agreement with human labels
    logger.info("Computing evaluator-human agreement...")
    agreement = compute_evaluator_agreement(responses, all_results)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Flatten all results for saving
    flat_results = []
    for eval_name, results in all_results.items():
        flat_results.extend(results)
    save_results(flat_results, output_path)

    # Save agreement report
    agreement_path = output_path.parent / "agreement_report.json"
    with open(agreement_path, "w") as f:
        json.dump(agreement, f, indent=2)

    logger.info("Results saved to %s", output_path)
    logger.info("Agreement report saved to %s", agreement_path)

    # Print summary table
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    for eval_name, results in all_results.items():
        scores = [r.overall_score for r in results]
        mean_s = sum(scores) / len(scores)
        print(f"\n{eval_name}:")
        print(f"  Mean score: {mean_s:.2f}")
        if eval_name in agreement:
            a = agreement[eval_name]
            print(f"  vs Human — Pearson r: {a['pearson_r']:.3f}, "
                  f"Spearman ρ: {a['spearman_r']:.3f}, "
                  f"MAE: {a['mean_absolute_error']:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
