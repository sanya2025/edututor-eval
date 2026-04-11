"""Synthetic dataset generator for AI tutor response evaluation.

Generates realistic tutoring exchanges with controllable quality levels.
The generator creates correlated features: high-quality responses tend to
use scaffolding, ask questions back, match the grade level, and avoid
giving away answers — just like real tutoring data.

Usage:
    python -m edututor_eval.generate_data --n_samples 200 --output data/synthetic_responses.json
"""

from __future__ import annotations

import argparse
import json
import random
import uuid
from pathlib import Path

from edututor_eval.datatypes import Subject, GradeLevel

# --- Question templates by subject and grade ---

QUESTIONS = {
    "math": {
        "K-2": [
            "What is 7 + 5?",
            "If I have 3 apples and get 4 more, how many do I have?",
            "What number comes after 19?",
            "Can you help me count by 2s?",
            "What is 10 - 6?",
        ],
        "3-5": [
            "How do I multiply 12 × 8?",
            "What is a fraction? Can you explain with an example?",
            "How do I find the area of a rectangle?",
            "What's the difference between perimeter and area?",
            "I don't understand how to divide 144 by 12.",
        ],
        "6-8": [
            "How do I solve 3x + 7 = 22?",
            "What is the Pythagorean theorem and when do I use it?",
            "Can you explain what a negative exponent means?",
            "How do I convert a fraction to a decimal?",
            "What's the difference between a ratio and a proportion?",
        ],
        "9-12": [
            "How do I find the derivative of f(x) = x³ + 2x?",
            "Can you explain what a logarithm is?",
            "How do I solve a system of equations with substitution?",
            "What is the quadratic formula and when should I use it?",
            "How do I find the area under a curve?",
        ],
    },
    "science": {
        "K-2": [
            "Why is the sky blue?",
            "What do plants need to grow?",
            "Why do we have seasons?",
            "What is gravity?",
            "How do magnets work?",
        ],
        "3-5": [
            "What is the water cycle?",
            "How does the food chain work?",
            "What are the phases of the moon?",
            "What's the difference between a solid, liquid, and gas?",
            "How do volcanoes erupt?",
        ],
        "6-8": [
            "What is photosynthesis and why is it important?",
            "How does natural selection work?",
            "What is the difference between an element and a compound?",
            "How do tectonic plates cause earthquakes?",
            "Can you explain the law of conservation of energy?",
        ],
        "9-12": [
            "How does meiosis differ from mitosis?",
            "Can you explain Newton's third law with examples?",
            "What is the ideal gas law and how do I use it?",
            "How does CRISPR gene editing work?",
            "What is the relationship between wavelength and frequency?",
        ],
    },
}

# --- Response templates by quality level ---
# Each is a function that takes (question, subject, grade) and returns text

def _high_quality_response(question: str, subject: str, grade: str) -> str:
    """Quality 4-5: Scaffolding, encouraging, grade-appropriate."""
    templates = [
        (
            "Great question! Let's work through this together. "
            "First, think about what you already know about {topic}. "
            "What's the first step you'd try? "
            "Here's a hint: {hint} "
            "Take your time — there's no rush. Once you try the first step, "
            "I'll help you check your work and move to the next one."
        ),
        (
            "I can see you're thinking carefully about this — that's exactly "
            "the right approach! Let me guide you through it step by step. "
            "Step 1: {step1} "
            "Step 2: {step2} "
            "Can you try Step 1 and tell me what you get? "
            "Remember, making mistakes is how we learn!"
        ),
        (
            "That's a really interesting question! Let's explore it together. "
            "Have you noticed that {observation}? "
            "What do you think would happen if {hypothetical}? "
            "Let's think about why that might be the case. "
            "Here's something to consider: {consideration}"
        ),
    ]
    template = random.choice(templates)
    fills = _get_fills(subject, grade)
    return template.format(**fills)


def _medium_quality_response(question: str, subject: str, grade: str) -> str:
    """Quality 2.5-3.5: Correct but may lack scaffolding or engagement."""
    templates = [
        (
            "To solve this, you need to {method}. "
            "The key concept here is {concept}. "
            "Apply {technique} and you should get the answer. "
            "Let me know if you need more help."
        ),
        (
            "This involves {topic}. "
            "Here's how it works: {explanation} "
            "The answer is found by {method}. "
            "Does that make sense?"
        ),
        (
            "Good question. {concept} is about {explanation}. "
            "You can think of it as {analogy}. "
            "Try working through it with that in mind."
        ),
    ]
    template = random.choice(templates)
    fills = _get_fills(subject, grade)
    return template.format(**fills)


def _low_quality_response(question: str, subject: str, grade: str) -> str:
    """Quality 1-2: Answer leaked, off-topic, or pedagogically poor."""
    templates = [
        # Answer dumping
        "The answer is {answer}.",
        "It equals {answer}. Just memorize it.",
        # Dismissive
        "This is simple. You should know this by now. The answer is {answer}.",
        # Off-topic / too complex
        (
            "This relates to the fundamental axioms of {advanced_topic}. "
            "In formal mathematical notation, we express this as a "
            "bijective homomorphism between the domain and codomain, "
            "which trivially yields the result."
        ),
        # Too vague
        "Just think about it more carefully and you'll figure it out.",
        # Meta-AI response
        "As an AI language model, I can tell you that the answer is {answer}. I don't have feelings about this.",
    ]
    template = random.choice(templates)
    fills = _get_fills(subject, grade)
    return template.format(**fills)


def _get_fills(subject: str, grade: str) -> dict:
    """Generate template fill values based on subject and grade."""
    math_fills = {
        "topic": random.choice(["addition", "multiplication", "equations", "geometry"]),
        "hint": random.choice([
            "try breaking the problem into smaller parts",
            "think about what operation you need to use",
            "draw a picture to help you visualize it",
        ]),
        "step1": "Identify what the problem is asking you to find",
        "step2": "Write down the information you're given",
        "observation": "numbers follow patterns",
        "hypothetical": "we changed one of the numbers",
        "consideration": "what would the simplest version of this problem look like?",
        "method": "apply the formula step by step",
        "concept": "order of operations",
        "technique": "the standard algorithm",
        "explanation": "working with numerical relationships",
        "analogy": "a recipe where each step builds on the last",
        "answer": str(random.randint(1, 100)),
        "advanced_topic": "abstract algebra and ring theory",
    }
    science_fills = {
        "topic": random.choice(["energy", "cells", "forces", "ecosystems"]),
        "hint": random.choice([
            "think about what you can observe in everyday life",
            "consider cause and effect",
            "what patterns do you notice?",
        ]),
        "step1": "Make an observation about what you see",
        "step2": "Form a hypothesis about why it happens",
        "observation": "this process happens all around us",
        "hypothetical": "conditions were different",
        "consideration": "how does this connect to what you already know?",
        "method": "apply the scientific method",
        "concept": "conservation of energy",
        "technique": "hypothesis testing",
        "explanation": "how systems interact and change over time",
        "analogy": "a machine where each part has a specific job",
        "answer": random.choice(["photosynthesis", "gravity", "42", "mitosis"]),
        "advanced_topic": "quantum field theory and statistical mechanics",
    }
    return math_fills if subject == "math" else science_fills


def generate_dataset(
    n_samples: int = 200,
    quality_distribution: dict[str, float] | None = None,
    seed: int = 42,
) -> list[dict]:
    """Generate a synthetic evaluation dataset.

    Args:
        n_samples: Number of samples to generate.
        quality_distribution: Probability of each quality tier.
            Default: 20% low, 40% medium, 40% high (realistic skew
            — most AI tutor responses are okay, not great or terrible).
        seed: Random seed for reproducibility.

    Returns:
        List of response dicts matching TutorResponse schema.
    """
    random.seed(seed)

    if quality_distribution is None:
        quality_distribution = {"low": 0.20, "medium": 0.40, "high": 0.40}

    subjects = ["math", "science"]
    grades = ["K-2", "3-5", "6-8", "9-12"]

    samples = []
    for i in range(n_samples):
        subject = random.choice(subjects)
        grade = random.choice(grades)
        question = random.choice(QUESTIONS[subject][grade])

        # Pick quality tier
        tier = random.choices(
            list(quality_distribution.keys()),
            weights=list(quality_distribution.values()),
        )[0]

        if tier == "high":
            response_text = _high_quality_response(question, subject, grade)
            base_score = random.uniform(3.8, 5.0)
        elif tier == "medium":
            response_text = _medium_quality_response(question, subject, grade)
            base_score = random.uniform(2.5, 3.8)
        else:
            response_text = _low_quality_response(question, subject, grade)
            base_score = random.uniform(1.0, 2.5)

        # Per-dimension scores with noise (correlated with overall)
        human_labels = {}
        for dim in ["correctness", "pedagogical_alignment", "curriculum_grounding",
                     "engagement", "safety"]:
            noise = random.gauss(0, 0.3)
            dim_score = max(1.0, min(5.0, base_score + noise))
            human_labels[dim] = round(dim_score, 1)

        # Safety is usually high unless the response is dismissive
        if tier != "low":
            human_labels["safety"] = round(random.uniform(4.0, 5.0), 1)

        sample = {
            "id": f"syn_{i:04d}",
            "student_question": question,
            "tutor_response": response_text,
            "subject": subject,
            "grade_level": grade,
            "topic": f"{subject}_topic_{random.randint(1, 20)}",
            "conversation_history": [],
            "human_score": round(base_score, 1),
            "human_labels": human_labels,
        }
        samples.append(sample)

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic AI tutor response data"
    )
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--output", type=str, default="data/synthetic_responses.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Generating {args.n_samples} synthetic tutor responses...")
    samples = generate_dataset(n_samples=args.n_samples, seed=args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"responses": samples}, f, indent=2)

    # Print quality distribution summary
    scores = [s["human_score"] for s in samples]
    print(f"Saved {len(samples)} responses to {output_path}")
    print(f"Score distribution: mean={sum(scores)/len(scores):.2f}, "
          f"min={min(scores):.1f}, max={max(scores):.1f}")
    low = sum(1 for s in scores if s < 2.5)
    med = sum(1 for s in scores if 2.5 <= s < 3.8)
    high = sum(1 for s in scores if s >= 3.8)
    print(f"Tiers: low={low}, medium={med}, high={high}")


if __name__ == "__main__":
    main()
