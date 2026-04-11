"""
Three-Way Comparison Study: GSM8K × 3 Prompt Styles × 3 Evaluators

Usage:
    # Full study (Steps 1-3): generate data + run evaluators
    python run_study.py

    # Skip generation if you already ran it, just re-run evaluators
    python run_study.py --skip-generation

    # After filling out data/manual_annotations.csv, compute agreement
    python run_study.py --agreement-only

Requirements:
    pip install openai datasets
    export OPENAI_API_KEY="sk-..."
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("❌ API key NOT loaded")
    raise ValueError("Missing OPENAI_API_KEY in environment")

api_key = api_key.strip()

print("✅ API key loaded")
print("Length:", len(api_key))

client = OpenAI(api_key=api_key)

try:
    models = client.models.list()
    print("✅ API key works")
except Exception as e:
    print("❌ API key issue:", e)


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

N_PROBLEMS = 100  # How many GSM8K problems to use
GENERATION_MODEL = "gpt-4o-mini"  # Cheaper model for generating tutor responses
JUDGE_MODEL = "gpt-4o-mini"  # Model for LLM-as-judge evaluation
TEMPERATURE = 0.3
SLEEP_BETWEEN_CALLS = 0.3  # Seconds between API calls (avoid rate limits)

DATA_PATH = Path("data/gsm8k_300.json")
RESULTS_PATH = Path("results/gsm8k_eval.json")
AGREEMENT_PATH = Path("results/agreement_report.json")
REPORT_PATH = Path("results/gsm8k_study_report.md")
ANNOTATIONS_PATH = Path("data/manual_annotations.csv")
ANNOTATION_SAMPLE_PATH = Path("data/annotation_sample.csv")  # pre-filled template

# ---------------------------------------------------------------------------
# PROMPT STYLES
# ---------------------------------------------------------------------------

PROMPT_STYLES = {
    "direct_answer": {
        "system": (
            "You are a math assistant. When given a problem, give the final "
            "answer directly and concisely. Do not show your reasoning or "
            "explain any steps. Just state the answer."
        ),
        "expected_quality": "low",
        "description": "Gives answer immediately, no teaching",
    },
    "step_by_step": {
        "system": (
            "You are a math tutor. When given a problem, solve it step by step, "
            "showing all your work clearly so the student can follow along. "
            "Label each step. Explain what you're doing and why."
        ),
        "expected_quality": "medium",
        "description": "Shows work but does the thinking for the student",
    },
    "socratic": {
        "system": (
            "You are a Socratic math tutor helping a student who is stuck. "
            "Do NOT give the answer or solve the problem. Instead, ask 1-2 "
            "guiding questions that help the student figure out the next step "
            "themselves. Offer a small hint if needed. Be encouraging and warm. "
            "Your goal is to help them think, not to think for them."
        ),
        "expected_quality": "high",
        "description": "Guides student to discover the answer themselves",
    },
}


# ---------------------------------------------------------------------------
# STEP 1 + 2: LOAD GSM8K AND GENERATE RESPONSES
# ---------------------------------------------------------------------------


def load_gsm8k(n: int = 100) -> list[dict]:
    """Load the first N problems from GSM8K test split."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' not installed. Run: pip install datasets")
        sys.exit(1)

    print(f"Loading {n} problems from GSM8K...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = [
        {
            "id": f"gsm_{i:03d}",
            "question": ds[i]["question"],
            "answer": ds[i]["answer"],  # Ground truth (for reference only)
        }
        for i in range(min(n, len(ds)))
    ]
    print(f"Loaded {len(problems)} problems.")
    return problems


def generate_response(client, question: str, style: str) -> str:
    """Call the LLM with a specific prompt style. Returns the response text."""
    system_prompt = PROMPT_STYLES[style]["system"]
    completion = client.chat.completions.create(
        model=GENERATION_MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    )
    return completion.choices[0].message.content


def generate_all_responses(problems: list[dict]) -> list[dict]:
    """Generate 3 responses per problem (one per prompt style). 300 total."""
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: 'openai' not installed. Run: pip install openai")
        sys.exit(1)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Run: export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    responses = []
    total = len(problems) * len(PROMPT_STYLES)
    count = 0

    print(
        f"\nGenerating {total} responses ({len(problems)} problems × {len(PROMPT_STYLES)} styles)..."
    )
    print(f"Model: {GENERATION_MODEL} | Est. cost: ~$1-3\n")

    for prob in problems:
        for style in PROMPT_STYLES:
            count += 1
            print(
                f"  [{count}/{total}] Problem {prob['id']} | style: {style}", end="\r"
            )

            try:
                text = generate_response(client, prob["question"], style)
            except Exception as e:
                print(
                    f"\nWARNING: Failed to generate {prob['id']}/{style}: {e}. Using placeholder."
                )
                text = f"[Generation failed: {e}]"

            responses.append(
                {
                    "id": f"{prob['id']}_{style}",
                    "student_question": prob["question"],
                    "tutor_response": text,
                    "subject": "math",
                    "grade_level": "3-5",
                    "topic": "word_problems",
                    "prompt_style": style,
                    "human_score": None,
                    "human_labels": None,
                }
            )

            time.sleep(SLEEP_BETWEEN_CALLS)

    print(f"\nGenerated {len(responses)} responses.")
    return responses


# ---------------------------------------------------------------------------
# STEP 3: RUN ALL THREE EVALUATORS
# ---------------------------------------------------------------------------


def run_evaluators(responses: list[dict]) -> list[dict]:
    """Run rule_based, learned, and llm_judge on all responses."""
    # Import evaluators from the edututor_eval package
    sys.path.insert(0, str(Path(__file__).parent))
    from edututor_eval.datatypes import GradeLevel, Subject, TutorResponse
    from edututor_eval.metrics.learned import LearnedEvaluator
    from edututor_eval.metrics.llm_judge import LLMJudgeEvaluator
    from edututor_eval.metrics.rule_based import RuleBasedEvaluator
    from edututor_eval.rubrics import load_rubric

    rubric = load_rubric("configs/rubric_default.yaml")

    evaluators = {
        "rule_based": RuleBasedEvaluator(),
        "learned": LearnedEvaluator(),
        "llm_judge": LLMJudgeEvaluator(
            rubric=rubric,
            provider="openai",
            model=JUDGE_MODEL,
            temperature=0.1,
        ),
    }

    all_results = []
    total = len(responses) * len(evaluators)
    count = 0

    print(
        f"\nRunning {len(evaluators)} evaluators on {len(responses)} responses ({total} total)..."
    )
    print(f"LLM judge model: {JUDGE_MODEL} | Est. cost: ~$1-2\n")

    for ev_name, evaluator in evaluators.items():
        print(f"  Running: {ev_name}")
        for resp_data in responses:
            count += 1
            print(f"    [{count}/{total}]", end="\r")

            # Convert grade_level string to GradeLevel enum
            grade_map = {
                "K-2": GradeLevel.K_2,
                "3-5": GradeLevel.GRADE_3_5,
                "6-8": GradeLevel.GRADE_6_8,
                "9-12": GradeLevel.GRADE_9_12,
            }
            subj_map = {"math": Subject.MATH, "science": Subject.SCIENCE}

            r = TutorResponse(
                id=resp_data["id"],
                student_question=resp_data["student_question"],
                tutor_response=resp_data["tutor_response"],
                subject=subj_map.get(resp_data["subject"], Subject.MATH),
                grade_level=grade_map.get(
                    resp_data["grade_level"], GradeLevel.GRADE_3_5
                ),
                topic=resp_data.get("topic", ""),
                human_score=resp_data.get("human_score"),
            )

            try:
                result = evaluator.evaluate(r)
            except Exception as e:
                print(
                    f"\nWARNING: Evaluator {ev_name} failed on {resp_data['id']}: {e}"
                )
                continue

            result_dict = {
                "response_id": result.response_id,
                "evaluator": result.evaluator,
                "overall_score": result.overall_score,
                "flags": result.flags,
                "prompt_style": resp_data.get("prompt_style", ""),
            }
            for ds in result.dimension_scores:
                result_dict[f"score_{ds.dimension}"] = ds.score

            all_results.append(result_dict)

            if ev_name == "llm_judge":
                time.sleep(SLEEP_BETWEEN_CALLS)

    print(f"\nCompleted {len(all_results)} evaluations.")
    return all_results


# ---------------------------------------------------------------------------
# STEP 4 HELPER: Generate annotation sample CSV
# ---------------------------------------------------------------------------


def generate_annotation_sample(responses: list[dict], n: int = 50) -> None:
    """
    Create a CSV template for manual annotation.

    Sampling strategy: stratified by prompt style + includes some
    high-disagreement responses (most informative to annotate).
    """
    # Try to load existing eval results to find disagreements
    disagreement_ids = set()
    if RESULTS_PATH.exists():
        results_df = pd.read_json(RESULTS_PATH)
        rule = results_df[results_df.evaluator == "rule_based"][
            ["response_id", "overall_score"]
        ].rename(columns={"overall_score": "rule"})
        llm = results_df[results_df.evaluator.str.contains("llm")][
            ["response_id", "overall_score"]
        ].rename(columns={"overall_score": "llm"})
        if not rule.empty and not llm.empty:
            merged = rule.merge(llm, on="response_id")
            merged["diff"] = abs(merged["rule"] - merged["llm"])
            top_disagreements = merged.nlargest(10, "diff")["response_id"].tolist()
            disagreement_ids = set(top_disagreements)
            print(
                f"  Found {len(disagreement_ids)} high-disagreement responses to prioritize"
            )

    # Stratified sample: ~17 per style, prioritize disagreements
    df = pd.DataFrame(responses)
    sampled_ids = set()

    # First, add disagreement cases
    sampled_ids.update(disagreement_ids)

    # Then fill up with stratified sample
    per_style = (n - len(sampled_ids)) // len(PROMPT_STYLES)
    for style in PROMPT_STYLES:
        style_responses = df[df["prompt_style"] == style]["id"].tolist()
        # Exclude already sampled
        remaining = [r for r in style_responses if r not in sampled_ids]
        sampled_ids.update(remaining[:per_style])
        if len(sampled_ids) >= n:
            break

    sample = df[df["id"].isin(list(sampled_ids)[:n])]

    # Build annotation template
    annotation_rows = []
    for _, row in sample.iterrows():
        annotation_rows.append(
            {
                "response_id": row["id"],
                "prompt_style": row["prompt_style"],
                "student_question": row["student_question"],
                "tutor_response_preview": row["tutor_response"],
                # --- Fill these in manually ---
                "score_correctness": "",
                "score_pedagogical_alignment": "",
                "score_curriculum_grounding": "",
                "score_engagement": "",
                "score_safety": "",
                "human_overall": "",
                "notes": "",
            }
        )

    annotation_df = pd.DataFrame(annotation_rows)
    ANNOTATION_SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    annotation_df.to_csv(ANNOTATION_SAMPLE_PATH, index=False)
    print(f"\nAnnotation template saved to: {ANNOTATION_SAMPLE_PATH}")
    print(f"  {len(annotation_rows)} responses to annotate")
    print(
        "  Fill in the score_* and human_overall columns using configs/rubric_default.yaml"
    )
    print("  Save as data/manual_annotations.csv when done")


# ---------------------------------------------------------------------------
# STEP 5: COMPUTE EVALUATOR-HUMAN AGREEMENT
# ---------------------------------------------------------------------------


def compute_agreement_report() -> None:
    """Load manual annotations and compare with each evaluator's scores."""
    if not ANNOTATIONS_PATH.exists():
        print(f"ERROR: {ANNOTATIONS_PATH} not found.")
        print("Complete the manual annotation step first:")
        print("  1. Open data/annotation_sample.csv")
        print("  2. Score each response using configs/rubric_default.yaml")
        print("  3. Save as data/manual_annotations.csv")
        return

    if not RESULTS_PATH.exists():
        print(f"ERROR: {RESULTS_PATH} not found. Run the full study first.")
        return

    manual = pd.read_csv(ANNOTATIONS_PATH)
    # Drop rows where human_overall is empty (not yet annotated)
    manual = manual.dropna(subset=["human_overall"])
    manual["human_overall"] = pd.to_numeric(manual["human_overall"], errors="coerce")
    manual = manual.dropna(subset=["human_overall"])
    print(f"\nLoaded {len(manual)} completed annotations.")

    results_df = pd.read_json(RESULTS_PATH)
    evaluators = results_df["evaluator"].unique()

    agreement_rows = []
    for ev in evaluators:
        ev_df = results_df[results_df["evaluator"] == ev][
            ["response_id", "overall_score"]
        ]
        merged = manual.merge(ev_df, on="response_id")
        if merged.empty:
            continue

        human = merged["human_overall"].values
        auto = merged["overall_score"].values

        pearson = float(np.corrcoef(human, auto)[0, 1])
        mae = float(np.mean(np.abs(human - auto)))
        adjacent = float(np.mean(np.abs(human - auto) <= 0.5))

        # Spearman (manual fallback if scipy unavailable)
        try:
            from scipy.stats import spearmanr

            spearman, _ = spearmanr(human, auto)
            spearman = float(spearman)
        except ImportError:
            ranks_h = np.argsort(np.argsort(human)).astype(float)
            ranks_a = np.argsort(np.argsort(auto)).astype(float)
            spearman = float(np.corrcoef(ranks_h, ranks_a)[0, 1])

        agreement_rows.append(
            {
                "evaluator": ev,
                "n": len(merged),
                "pearson_r": round(pearson, 3),
                "spearman_rho": round(spearman, 3),
                "mae": round(mae, 3),
                "adjacent_agreement": f"{adjacent:.1%}",
            }
        )

        print(
            f"  {ev:25s}  Pearson={pearson:.3f}  Spearman={spearman:.3f}  "
            f"MAE={mae:.3f}  Adjacent={adjacent:.1%}"
        )

    # Save agreement report
    agreement_df = pd.DataFrame(agreement_rows)
    AGREEMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AGREEMENT_PATH, "w") as f:
        json.dump(agreement_rows, f, indent=2)

    # Also compute per-style breakdown
    print("\nAgreement by prompt style:")
    style_rows = []
    for style in PROMPT_STYLES:
        style_manual = (
            manual[manual["prompt_style"] == style]
            if "prompt_style" in manual.columns
            else manual
        )
        if style_manual.empty:
            continue
        for ev in evaluators:
            ev_df = results_df[results_df["evaluator"] == ev][
                ["response_id", "overall_score"]
            ]
            merged = style_manual.merge(ev_df, on="response_id")
            if len(merged) < 3:
                continue
            human = merged["human_overall"].values
            auto = merged["overall_score"].values
            mae = float(np.mean(np.abs(human - auto)))
            style_rows.append(
                {
                    "style": style,
                    "evaluator": ev,
                    "n": len(merged),
                    "mae": round(mae, 3),
                }
            )
            print(f"  {ev:25s} | {style:15s} | n={len(merged):2d} | MAE={mae:.3f}")

    # Write full report
    _write_report(agreement_rows, style_rows)


def _write_report(agreement_rows: list[dict], style_rows: list[dict]) -> None:
    """Write the final Markdown report."""
    lines = [
        "# Three-Way Comparison Study Results\n",
        "## Dataset: GSM8K × 3 Prompt Styles\n",
        f"- **Problems:** {N_PROBLEMS} GSM8K test problems",
        f"- **Responses:** {N_PROBLEMS * 3} total (3 prompt styles × {N_PROBLEMS})",
        "- **Prompt styles:** `direct_answer` (low quality) · `step_by_step` (medium) · `socratic` (high)\n",
        "## Evaluator–Human Agreement\n",
        "| Evaluator | N | Pearson r | Spearman ρ | MAE | Adjacent (±0.5) |",
        "|-----------|---|-----------|------------|-----|-----------------|",
    ]

    for row in agreement_rows:
        lines.append(
            f"| {row['evaluator']} | {row['n']} | {row['pearson_r']} | "
            f"{row['spearman_rho']} | {row['mae']} | {row['adjacent_agreement']} |"
        )

    lines += [
        "\n## Agreement by Prompt Style\n",
        "| Evaluator | Style | N | MAE |",
        "|-----------|-------|---|-----|",
    ]
    for row in style_rows:
        lines.append(
            f"| {row['evaluator']} | {row['style']} | {row['n']} | {row['mae']} |"
        )

    lines += [
        "\n## Interpretation\n",
        "- **Rule-based**: Fast and free, best for pre-filtering obvious failures",
        "- **Learned (fallback)**: Better than rule-based without model training cost",
        "- **LLM-as-judge**: Highest agreement with humans, best for auditing and training data generation",
        "\nIn production: use rule-based as a real-time gate, LLM-judge for periodic auditing,",
        "and the learned model (once trained on LLM-judge labels) for volume scoring.",
    ]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"\nFull report saved to: {REPORT_PATH}")


# ---------------------------------------------------------------------------
# SCORE DISTRIBUTION CHECK (sanity check after generation + evaluation)
# ---------------------------------------------------------------------------


def print_score_distribution() -> None:
    """Print mean scores per evaluator per prompt style (sanity check)."""
    if not RESULTS_PATH.exists():
        return
    df = pd.read_json(RESULTS_PATH)
    print("\nMean scores by evaluator × prompt style:")
    print("-" * 55)
    for ev in df["evaluator"].unique():
        ev_df = df[df["evaluator"] == ev]
        row = f"  {ev:25s}"
        for style in ["direct_answer", "step_by_step", "socratic"]:
            style_df = ev_df[ev_df["prompt_style"] == style]
            if not style_df.empty:
                mean = style_df["overall_score"].mean()
                row += f"  {style[:6]}: {mean:.2f}"
        print(row)
    print("-" * 55)
    print("  Expected order: direct_answer < step_by_step < socratic")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Three-way evaluator comparison study")
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip data generation (use existing data/gsm8k_300.json)",
    )
    parser.add_argument(
        "--agreement-only",
        action="store_true",
        help="Only compute agreement (requires manual_annotations.csv)",
    )
    parser.add_argument("--n-problems", type=int, default=N_PROBLEMS)
    args = parser.parse_args()

    Path("data").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    if args.agreement_only:
        print("=== Step 5: Computing Evaluator-Human Agreement ===")
        compute_agreement_report()
        return

    # ---- Steps 1 & 2: Generate data ----
    if args.skip_generation and DATA_PATH.exists():
        print(f"Skipping generation. Loading existing data from {DATA_PATH}...")
        with open(DATA_PATH) as f:
            responses = json.load(f)["responses"]
        print(f"Loaded {len(responses)} responses.")
    else:
        problems = load_gsm8k(n=args.n_problems)
        responses = generate_all_responses(problems)
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DATA_PATH, "w") as f:
            json.dump({"responses": responses}, f, indent=2)
        print(f"Saved to {DATA_PATH}")

    # ---- Step 3: Run evaluators ----
    print("\n=== Step 3: Running Evaluators ===")
    eval_results = run_evaluators(responses)
    results_df = pd.DataFrame(eval_results)
    results_df.to_json(RESULTS_PATH, orient="records", indent=2)
    print(f"Saved {len(eval_results)} results to {RESULTS_PATH}")

    # Sanity check
    print_score_distribution()

    # ---- Step 4: Prepare annotation template ----
    print("\n=== Step 4: Preparing Annotation Template ===")
    generate_annotation_sample(responses, n=50)

    print("\n" + "=" * 60)
    print("NEXT STEP: Manual Annotation")
    print("=" * 60)
    print(f"1. Open: {ANNOTATION_SAMPLE_PATH}")
    print(f"2. Open: configs/rubric_default.yaml  (scoring guide)")
    print("3. Score each response on all 5 dimensions (1-5)")
    print("4. Fill in 'human_overall' column")
    print(f"5. Save as: {ANNOTATIONS_PATH}")
    print("6. Then run: python run_study.py --agreement-only")


if __name__ == "__main__":
    main()
