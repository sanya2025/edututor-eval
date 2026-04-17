# edututor-eval

**A lightweight evaluation framework for AI tutoring responses, built for education-focused LLM systems.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Why This Exists

AI tutors powered by LLMs face a unique evaluation challenge: a response can be *factually correct* but *pedagogically harmful* — for example, giving away the answer instead of scaffolding the student toward it. Standard NLP metrics (BLEU, ROUGE) miss this entirely.

**edututor-eval** provides a structured framework for evaluating AI tutor responses across the dimensions that actually matter in education:

| Dimension | What It Measures |
|-----------|-----------------|
| **Correctness** | Is the math/science content accurate? |
| **Pedagogical Alignment** | Does it scaffold rather than give answers? |
| **Curriculum Grounding** | Is it appropriate for the grade level? |
| **Engagement** | Is the tone encouraging and age-appropriate? |
| **Safety** | Does it avoid harmful or misleading content? |

## Quick Start

```bash
git clone https://github.com/sanya2025/edututor-eval.git
cd edututor-eval
pip install -e .
```

### Run evaluation on sample data

```bash
python -m edututor_eval.run_eval \
    --data data/sample_responses.json \
    --rubric configs/rubric_default.yaml \
    --output results/eval_results.json
```

### Generate synthetic test data

```bash
python -m edututor_eval.generate_data \
    --n_samples 200 \
    --output data/synthetic_responses.json
```

### Analyze results

```bash
python -m edututor_eval.analyze \
    --results results/eval_results.json \
    --output results/analysis_report.md
```

## Architecture

```
Student Question + AI Response + Curriculum Context
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   Rule-Based   LLM-as-Judge  Learned
    Metrics      Scoring      Classifier
        │           │           │
        └───────────┼───────────┘
                    ▼
            Score Aggregation
          (weighted ensemble)
                    ▼
         Per-Dimension Scores
    + Overall Quality Score (1–5)
    + Actionable Feedback
```

Three evaluation strategies, compared head-to-head:

1. **Rule-based metrics** — Fast heuristics: response length, question echo, answer leakage detection, reading level checks
2. **LLM-as-Judge** — GPT-4 / Claude scoring against a structured rubric with chain-of-thought reasoning
3. **Learned classifier** — Fine-tuned DeBERTa trained on human-annotated tutor response quality labels

## Project Structure

```
edututor-eval/
├── edututor_eval/
│   ├── __init__.py
│   ├── run_eval.py          # Main evaluation pipeline
│   ├── generate_data.py     # Synthetic data generator
│   ├── analyze.py           # Results analysis + visualization
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── rule_based.py    # Heuristic scoring functions
│   │   ├── llm_judge.py     # LLM-as-judge evaluator
│   │   └── learned.py       # Fine-tuned model scorer
│   ├── rubrics.py           # Rubric loading and validation
│   ├── datatypes.py         # Data models (Pydantic)
│   └── utils.py             # Shared helpers
├── configs/
│   └── rubric_default.yaml  # Default evaluation rubric
├── data/
│   └── sample_responses.json
├── results/
├── tests/
│   └── test_metrics.py
├── setup.py
├── requirements.txt
└── README.md
```

## Key Design Decisions

These are deliberate choices, each with trade-offs worth discussing:

- **Multi-strategy evaluation** — No single method is sufficient. Rule-based catches obvious failures cheaply; LLM-as-judge handles nuance; learned models scale. The framework compares all three so you can pick the right cost/quality trade-off.
- **Rubric-driven** — Evaluation criteria are externalized in YAML, not hardcoded. This lets educators iterate on what "good" means without touching code.
- **Education-specific dimensions** — Generic quality scores hide what matters. Separating correctness from pedagogy from engagement reveals *why* a response fails, not just *that* it fails.
- **Synthetic data with controllable quality** — Real tutoring data is hard to get (privacy, scale). The generator creates realistic responses with known quality levels, enabling rapid prototyping.

## Requirements

```
python >= 3.10
torch >= 2.1
transformers >= 4.36
scikit-learn >= 1.3
pandas >= 2.1
pyyaml >= 6.0
pydantic >= 2.5
textstat >= 0.7
```

Optional (for LLM-as-judge): `openai >= 1.6` or `anthropic >= 0.18`


## License

MIT

## Citation

```bibtex
@software{edututor_eval_2026,
  title={edututor-eval: An Evaluation Framework for AI Tutoring Responses},
  author={Sanja Damjanovic},
  year={2026},
  url={https://github.com/sanya2025/edututor-eval}
}
```
