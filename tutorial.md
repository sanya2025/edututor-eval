# Building an Evaluation Framework for AI Tutors: A Practical Guide

*How to measure whether an AI tutor is actually teaching — not just answering.*

---

## The Problem Nobody Warns You About

When Khan Academy launched Khanmigo, they didn't just need an AI that could solve math problems — they needed one that could *teach* math. That distinction is the entire challenge.

Consider this student question: **"How do I solve 3x + 7 = 22?"**

**Response A:** "x = 5."

**Response B:** "Great question! Let's work through this together. First, what do you think we should do to isolate x? Hint: look at the +7. What operation would undo adding 7?"

Both are correct. Only one is tutoring. Standard NLP evaluation metrics — BLEU, ROUGE, perplexity — would score these almost identically, or might even prefer Response A for its precision. But any educator would tell you Response B is dramatically better.

This is the evaluation gap that `edututor-eval` addresses: **how do you build automated quality scoring for AI tutors that captures pedagogical quality, not just factual accuracy?**

This tutorial walks through building such a system from scratch, the design trade-offs involved, and what I learned along the way.

---

## Part 1: Defining "Quality" in Education AI

### Why Generic LLM Evaluation Fails

Most LLM evaluation frameworks (MT-Bench, AlpacaEval, LMSYS Arena) measure general helpfulness. But educational AI has inverted requirements:

| General AI Assistant | AI Tutor |
|---------------------|----------|
| Give the best answer quickly | Don't give the answer — scaffold toward it |
| Be comprehensive | Match the student's level (less can be more) |
| Sound authoritative | Sound encouraging and approachable |
| Solve the problem | Help the student solve the problem |

This inversion means we need education-specific evaluation dimensions. After reviewing learning science literature (Vygotsky's zone of proximal development, Bloom's taxonomy, the Socratic method), I settled on five:

1. **Correctness** — Is the content accurate?
2. **Pedagogical Alignment** — Does it scaffold rather than tell?
3. **Curriculum Grounding** — Is it appropriate for the grade level?
4. **Engagement** — Is the tone encouraging?
5. **Safety** — Is it free from harmful patterns?

These are encoded in a YAML rubric (`configs/rubric_default.yaml`) that defines 5-point scoring criteria for each dimension. Externalizing this is a deliberate design choice: educators should be able to modify what "good" means without touching Python.

### The Rubric as a Contract

Here's a snippet of the pedagogical alignment dimension:

```yaml
- name: pedagogical_alignment
  description: "Does the response scaffold learning rather than give away the answer?"
  weight: 1.5  # Weighted highest — this is the hardest and most important dimension
  score_levels:
    - level: 1
      description: "Directly gives the answer with no explanation or scaffolding"
    - level: 3
      description: "Explains the concept but doesn't actively guide the student"
    - level: 5
      description: "Masterful scaffolding — guides student to discover the answer themselves"
```

The weight of 1.5 (vs. 1.0 for correctness) reflects a key insight: **in tutoring, pedagogy matters more than correctness.** A response that's slightly imprecise but guides the student to think is better than one that's perfectly correct but does the thinking for them.

---

## Part 2: Three Evaluation Strategies

Rather than betting on a single approach, I implemented three strategies and compared them. This mirrors what you'd do in production: use cheap methods for bulk screening and expensive methods for quality assurance.

### Strategy 1: Rule-Based Heuristics

The simplest approach: pattern matching and text statistics.

```python
from edututor_eval.metrics.rule_based import RuleBasedEvaluator
from edututor_eval.utils import load_responses

responses = load_responses("data/sample_responses.json")
evaluator = RuleBasedEvaluator()

result = evaluator.evaluate(responses[0])
print(f"Overall: {result.overall_score}")
print(f"Flags: {result.flags}")
for dim in result.dimension_scores:
    print(f"  {dim.dimension}: {dim.score}")
```

What the rule-based evaluator checks:

- **Answer leakage**: Does the response start with "The answer is..."?
- **Scaffolding signals**: Does it contain guiding questions ("What do you think...?"), step-by-step markers, hints?
- **Reading level**: Is the Flesch-Kincaid grade appropriate for the student's level?
- **Engagement cues**: Encouragement patterns, conversational tone, direct address
- **Safety flags**: Discouraging language, meta-AI references

**Strengths:** Fast (~0.1ms per response), deterministic, interpretable, zero cost.

**Weaknesses:** Can't understand meaning. A response saying "What do you think the answer might be? Just kidding, it's 5" would score well on scaffolding heuristics despite being pedagogically bad.

### Strategy 2: LLM-as-Judge

Use a strong LLM (GPT-4, Claude) to evaluate responses against the rubric.

```python
from edututor_eval.rubrics import load_rubric
from edututor_eval.metrics.llm_judge import LLMJudgeEvaluator

rubric = load_rubric("configs/rubric_default.yaml")
evaluator = LLMJudgeEvaluator(
    rubric=rubric,
    provider="openai",
    model="gpt-4o",
    temperature=0.1,  # Low temp for consistency
)

result = evaluator.evaluate(responses[0])
print(result.metadata["reasoning"])  # Chain-of-thought explanation
```

Key design decisions in the LLM-as-judge implementation:

1. **Structured JSON output** — Requesting JSON reduces parsing failures from ~15% to ~2%
2. **Chain-of-thought before scoring** — Asking the LLM to reason before assigning numbers reduces position bias and improves calibration
3. **Rubric in the prompt** — The YAML rubric is serialized into the prompt, ensuring the LLM evaluates against the same criteria humans use
4. **Retry with fallback** — Parse failures are retried; after 3 failures, a default score is returned rather than crashing the pipeline

**Strengths:** Captures semantic nuance, understands pedagogical context, produces human-readable explanations.

**Weaknesses:** Expensive (~$0.03/evaluation with GPT-4o), slow (~2-3s), non-deterministic, has known biases (verbosity bias, position bias).

### Strategy 3: Learned Classifier

Fine-tune a transformer on human-annotated quality labels.

```python
from edututor_eval.metrics.learned import LearnedEvaluator

# With a trained model:
evaluator = LearnedEvaluator(model_path="models/quality_scorer_v1")

# Without a trained model (feature-based fallback):
evaluator = LearnedEvaluator()  # Uses calibrated feature weights

result = evaluator.evaluate(responses[0])
```

The architecture fuses three signal types:

1. **Text encoding** — DeBERTa-v3-base encodes the [question, response] pair
2. **Pedagogical features** — 7 hand-crafted indicators (scaffolding, question-asking, step structure, etc.)
3. **Curriculum context** — Subject and grade level embeddings

The fallback mode uses the pedagogical features with weights calibrated on pilot annotations. This lets the full pipeline run before you've collected enough labeled data to train the transformer.

**Strengths:** Fast at inference (~10ms on GPU), consistent, scales to millions of evaluations.

**Weaknesses:** Requires labeled training data (expensive to create), less interpretable than LLM judge, may not generalize to new response styles.

---

## Part 3: Running the Full Pipeline

### Step 1: Generate Synthetic Data

```bash
python -m edututor_eval.generate_data --n_samples 200 --output data/synthetic_responses.json
```

The generator creates responses across three quality tiers (low/medium/high) with correlated features — high-quality responses use more scaffolding, lower reading levels, and encouraging tone. The distribution defaults to 20% low, 40% medium, 40% high, which matches the realistic skew in production AI tutors (most responses are okay, not terrible or excellent).

### Step 2: Run Evaluation

```bash
python -m edututor_eval.run_eval \
    --data data/synthetic_responses.json \
    --rubric configs/rubric_default.yaml \
    --evaluators rule_based learned \
    --output results/eval_results.json
```

This runs both evaluators on all 200 responses, computes agreement with human labels, and saves structured results.

### Step 3: Analyze Results

```bash
python -m edututor_eval.analyze \
    --results results/eval_results.json \
    --output results/analysis_report.md
```

The analysis report includes score distributions, per-dimension breakdowns, flag frequencies, and evaluator-human agreement metrics.

---

## Part 4: What I Learned

### Lesson 1: Pedagogy is the hardest dimension to automate

Correctness can be verified with a calculator. Engagement can be approximated with sentiment analysis. But "does this response scaffold learning?" requires understanding the student's current knowledge state, the learning objective, and what a good next step would be. This is where LLM-as-judge currently dominates rule-based approaches.

### Lesson 2: The rubric is the product

The most impactful work wasn't the code — it was the rubric design. Getting educators to agree on what "level 3 pedagogical alignment" means took multiple iterations. In production, rubric versioning and A/B testing rubric changes would be as important as A/B testing model changes.

### Lesson 3: You need all three strategies

In a production system at Khan Academy's scale (181M+ learners), you'd likely use:
- **Rule-based** as a real-time pre-filter (block obvious failures before they reach students)
- **Learned model** as the primary scorer (fast enough for real-time, good enough for most cases)
- **LLM-as-judge** for periodic auditing and generating training data for the learned model

This creates a flywheel: LLM-judge labels train the learned model, which handles production volume, while rule-based catches edge cases both miss.

### Lesson 4: Synthetic data is more useful than expected

The synthetic generator was initially a prototyping tool, but it turned out to be valuable for:
- **Regression testing** — generate known-quality responses to verify evaluators after code changes
- **Bias detection** — generate identical content at different grade levels to test if the evaluator is grade-fair
- **Stress testing** — generate adversarial responses (prompt injections, off-topic, extremely long) to find evaluator failure modes

### Lesson 5: Agreement metrics tell you where to invest

If rule-based and LLM-judge agree, the rule is probably sufficient (save money). If they disagree, that's where human review adds the most value. This is essentially an active learning strategy for evaluation.

---

## Part 5: Extensions and Future Work

**Conversation-level evaluation.** The current system scores individual responses. But tutoring quality is really about the *conversation* — did the student learn? Multi-turn evaluation requires tracking the student's progress across exchanges.

**Curriculum-grounded hallucination detection.** A response can be factually correct but teach content that's out of scope for the grade level. Connecting the evaluator to curriculum standards (like Common Core) would catch cases where a 3rd-grade math tutor accidentally explains calculus.

**Adaptive rubrics.** A Socratic approach works well for problem-solving but poorly for factual recall. The rubric should adapt based on the type of question and learning objective.

**Fairness auditing.** Do AI tutors provide equally good scaffolding across student demographics? Building evaluation slices by demographic proxy (school type, geographic region) would surface systematic quality gaps.

---

## Interview Discussion Points

This project is designed to demonstrate several competencies relevant to the Khan Academy Senior AI Engineer role. Here are the key discussion threads:

### On Evaluation Design
- *"Why five dimensions instead of a single quality score?"* — Because a single score hides *why* something fails. A response can be correct but pedagogically harmful. Separating dimensions makes the evaluation actionable: you know exactly what to fix.
- *"How would you validate the rubric itself?"* — Inter-annotator agreement studies. Have multiple educators rate the same responses, compute Cohen's kappa, and iterate on rubric wording until kappa > 0.7 for each dimension.

### On LLM-as-Judge
- *"What are the failure modes of using an LLM to judge another LLM?"* — Verbosity bias (longer responses score higher), position bias (in pairwise comparison), sycophancy (the judge agrees with confident-sounding responses). Mitigation: structured rubrics, chain-of-thought, temperature control, calibration against human labels.
- *"How would you scale this to millions of responses?"* — You wouldn't. LLM-as-judge is for generating training data and auditing. The learned model handles volume. The architecture is designed for this handoff.

### On Production Deployment
- *"How would this integrate with Khanmigo's production system?"* — Real-time path: rule-based pre-filter → learned model scorer → response served (or blocked if score < threshold). Async path: sample N% of responses → LLM-judge audit → update training data → retrain learned model → deploy.
- *"How would you handle evaluation drift?"* — Monitor score distributions over time. If the mean drops or variance increases, trigger investigation. Compare evaluator agreement with fresh human annotations monthly.

### On Research
- *"What's the hardest open problem here?"* — Evaluating scaffolding quality without knowing the student's actual learning state. Current approaches use proxies (question difficulty, grade level), but real scaffolding quality depends on what the individual student already knows.
- *"Where does this connect to existing research?"* — MT-Bench (Zheng et al. 2023) for LLM evaluation methodology, RLHF literature for learning from human preferences, intelligent tutoring systems research (VanLehn 2011) for pedagogical quality metrics.

### On Ethics and Bias
- *"How would you ensure the evaluator doesn't have demographic bias?"* — Systematic testing: generate equivalent responses with varied student contexts, check for score differences. Separate the evaluator's assessment of response quality from any assumptions about student capability.
- *"What's the risk of optimizing tutor responses for the evaluator rather than for learning?"* — Goodhart's Law. The evaluator becomes the target, and the tutor learns to game it. Mitigation: rotate rubric emphasis, use human spot-checks, and ultimately measure downstream learning outcomes (did the student get the next problem right?).

---

## Running the Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Expected output:

```
tests/test_metrics.py::TestRuleBasedEvaluator::test_returns_valid_result PASSED
tests/test_metrics.py::TestRuleBasedEvaluator::test_good_response_scores_higher PASSED
tests/test_metrics.py::TestRuleBasedEvaluator::test_answer_leak_detected PASSED
tests/test_metrics.py::TestRuleBasedEvaluator::test_scaffolding_rewarded PASSED
tests/test_metrics.py::TestRuleBasedEvaluator::test_short_response_penalized PASSED
tests/test_metrics.py::TestRuleBasedEvaluator::test_all_dimensions_scored PASSED
tests/test_metrics.py::TestLearnedEvaluator::test_fallback_returns_valid_result PASSED
tests/test_metrics.py::TestLearnedEvaluator::test_feature_extraction PASSED
tests/test_metrics.py::TestPedagogicalFeatures::test_scaffolding_detected PASSED
tests/test_metrics.py::TestPedagogicalFeatures::test_answer_leak_detected PASSED
tests/test_metrics.py::TestPedagogicalFeatures::test_empty_text PASSED
```

---

## Acknowledgments

This project was inspired by the evaluation challenges facing AI-powered education platforms. The rubric design draws on learning science frameworks from Vygotsky (zone of proximal development), Bloom (taxonomy of learning objectives), and VanLehn (intelligent tutoring systems).
