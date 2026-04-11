"""Rule-based evaluation metrics for AI tutor responses.

These are fast, deterministic heuristics that catch common failure modes
without any model inference. They serve two roles:

1. **Cheap pre-filters** — flag obviously bad responses before spending
   money on LLM-as-judge calls
2. **Baselines** — establish a floor that learned models must beat to
   justify their complexity

The key insight: in educational AI, many quality failures are *structural*
rather than *semantic*. A response that dumps the answer in the first
sentence, or reads at a college level for a 3rd grader, can be caught
with simple text analysis.
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass

from edututor_eval.datatypes import TutorResponse, EvalResult, DimensionScore


# --- Grade level → approximate target reading level (Flesch-Kincaid) ---
GRADE_READING_TARGETS = {
    "K-2": (1.0, 3.0),
    "3-5": (3.0, 6.0),
    "6-8": (6.0, 9.0),
    "9-12": (9.0, 13.0),
}

# Patterns that suggest the tutor is giving away the answer directly
ANSWER_LEAK_PATTERNS = [
    r"(?i)^(the answer is|the solution is|it equals|that equals)",
    r"(?i)^(simply|just) (put|enter|type|write)\b",
    r"(?i)^(it'?s|that'?s|the answer'?s)\s+\d",
    r"(?i)^(you get|we get|this gives us)\s+\d",
]

# Patterns indicating good pedagogical scaffolding
SCAFFOLDING_PATTERNS = [
    r"(?i)(what do you think|what would happen|can you try|how might you)",
    r"(?i)(let'?s think about|let'?s start with|consider)",
    r"(?i)(first,? try|as a first step|start by)",
    r"(?i)(hint|clue|remember that|recall that)",
    r"(?i)(what if|have you considered|another way to think about)",
]

# Patterns suggesting encouragement
ENCOURAGEMENT_PATTERNS = [
    r"(?i)(great|good|nice|excellent) (question|thinking|try|effort|job|work)",
    r"(?i)(you'?re on the right track|that'?s a good start|keep going)",
    r"(?i)(don'?t worry|it'?s okay|mistakes are|let'?s try)",
]


@dataclass
class RuleBasedEvaluator:
    """Evaluator using deterministic heuristic rules.

    Each method scores one dimension on a 1-5 scale. The heuristics
    are intentionally simple — the goal is interpretability and speed,
    not maximum accuracy.
    """

    def evaluate(self, response: TutorResponse) -> EvalResult:
        text = response.tutor_response
        question = response.student_question

        scores = {}
        flags = []

        # --- Correctness proxy: can't verify math, but can flag red flags ---
        correctness_score, correctness_flags = self._score_correctness_proxy(text)
        scores["correctness"] = correctness_score
        flags.extend(correctness_flags)

        # --- Pedagogical alignment ---
        pedagogy_score, pedagogy_flags = self._score_pedagogy(text, question)
        scores["pedagogical_alignment"] = pedagogy_score
        flags.extend(pedagogy_flags)

        # --- Curriculum grounding (reading level appropriateness) ---
        grade_val = response.grade_level.value if hasattr(response.grade_level, 'value') else str(response.grade_level)
        curriculum_score = self._score_curriculum_grounding(
            text, grade_val
        )
        scores["curriculum_grounding"] = curriculum_score

        # --- Engagement ---
        engagement_score = self._score_engagement(text)
        scores["engagement"] = engagement_score

        # --- Safety ---
        safety_score, safety_flags = self._score_safety(text)
        scores["safety"] = safety_score
        flags.extend(safety_flags)

        # Overall: weighted average (pedagogy matters most for tutoring)
        weights = {
            "correctness": 0.20,
            "pedagogical_alignment": 0.30,
            "curriculum_grounding": 0.20,
            "engagement": 0.15,
            "safety": 0.15,
        }
        overall = sum(scores[k] * weights[k] for k in scores)

        dimension_scores = [
            DimensionScore(
                dimension=dim,
                score=score,
                confidence=0.6,  # rule-based = moderate confidence
                reasoning=f"Rule-based heuristic for {dim}",
            )
            for dim, score in scores.items()
        ]

        return EvalResult(
            response_id=response.id,
            evaluator="rule_based",
            overall_score=round(overall, 2),
            dimension_scores=dimension_scores,
            flags=flags,
            metadata={"method": "rule_based_v1"},
        )

    def _score_correctness_proxy(self, text: str) -> tuple[float, list[str]]:
        """Proxy correctness checks (we can't verify math without a solver).

        Flags: contradictions, hedging, nonsensical patterns.
        """
        flags = []
        score = 4.0  # assume reasonable unless red flags found

        # Very short responses are suspicious
        word_count = len(text.split())
        if word_count < 10:
            score -= 1.5
            flags.append("response_too_short")

        # Excessive hedging suggests uncertainty
        hedge_words = len(re.findall(r"(?i)\b(maybe|perhaps|might|possibly|i think)\b", text))
        if hedge_words >= 3:
            score -= 0.5
            flags.append("excessive_hedging")

        # Self-contradiction detection (simple: "is X" then "is not X")
        if re.search(r"(?i)(is|equals)\s+(\d+).*(?:is not|doesn't equal|isn't)\s+\2", text):
            score -= 2.0
            flags.append("potential_contradiction")

        return max(1.0, min(5.0, score)), flags

    def _score_pedagogy(self, text: str, question: str) -> tuple[float, list[str]]:
        """Score pedagogical alignment.

        The core question: does this response *teach*, or does it just
        *tell the answer*? Good tutoring scaffolds understanding through
        hints, questions, and guided steps.
        """
        flags = []
        score = 3.0

        # Penalize: giving away the answer directly
        for pattern in ANSWER_LEAK_PATTERNS:
            if re.search(pattern, text):
                score -= 1.5
                flags.append("answer_leaked")
                break

        # Reward: scaffolding language
        scaffolding_count = sum(
            1 for p in SCAFFOLDING_PATTERNS if re.search(p, text)
        )
        score += min(scaffolding_count * 0.5, 1.5)

        # Reward: asking the student a question back (Socratic method)
        question_marks_in_response = text.count("?")
        if question_marks_in_response >= 1:
            score += 0.5

        # Penalize: dumping a wall of text (information overload)
        if len(text.split()) > 300:
            score -= 0.5
            flags.append("response_too_long")

        # Reward: step-by-step structure
        step_markers = len(re.findall(r"(?i)(step \d|first|second|third|next|then|finally)", text))
        if step_markers >= 2:
            score += 0.5

        return max(1.0, min(5.0, score)), flags

    def _score_curriculum_grounding(self, text: str, grade_level: str) -> float:
        """Check if response reading level matches the target grade.

        Uses Flesch-Kincaid grade level as a proxy. This is imperfect —
        math notation inflates reading level — but it catches the worst
        cases (college-level explanations for 2nd graders).
        """
        fk_grade = self._flesch_kincaid_grade(text)
        target_min, target_max = GRADE_READING_TARGETS.get(grade_level, (5.0, 10.0))

        if target_min <= fk_grade <= target_max:
            return 5.0
        elif fk_grade < target_min:
            # Slightly below target: probably fine
            return 4.0
        else:
            # Above target: penalize proportionally
            overshoot = fk_grade - target_max
            return max(1.0, 5.0 - overshoot * 0.5)

    def _score_engagement(self, text: str) -> float:
        """Score how engaging and encouraging the response is."""
        score = 3.0

        encouragement_count = sum(
            1 for p in ENCOURAGEMENT_PATTERNS if re.search(p, text)
        )
        score += min(encouragement_count * 0.5, 1.0)

        # Conversational tone (contractions, direct address)
        if re.search(r"(?i)\b(you|your|you're|let's|we)\b", text):
            score += 0.5

        # Penalize: robotic/formal tone
        if re.search(r"(?i)(the student should|one must|it is necessary to)", text):
            score -= 0.5

        return max(1.0, min(5.0, score))

    def _score_safety(self, text: str) -> tuple[float, list[str]]:
        """Check for safety issues specific to educational context."""
        flags = []
        score = 5.0

        # Discouraging language
        if re.search(r"(?i)(you should know this|this is (easy|simple|basic)|obviously)", text):
            score -= 1.0
            flags.append("potentially_discouraging")

        # Off-topic detection (very basic)
        if re.search(r"(?i)(as an ai|i'm a language model|i cannot|i don't have feelings)", text):
            score -= 0.5
            flags.append("meta_ai_reference")

        return max(1.0, min(5.0, score)), flags

    @staticmethod
    def _flesch_kincaid_grade(text: str) -> float:
        """Compute Flesch-Kincaid grade level."""
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()

        if not sentences or not words:
            return 0.0

        # Count syllables (approximation)
        def count_syllables(word: str) -> int:
            word = word.lower().strip(".,!?;:'\"")
            if not word:
                return 0
            count = 0
            vowels = "aeiouy"
            prev_vowel = False
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_vowel:
                    count += 1
                prev_vowel = is_vowel
            if word.endswith("e") and count > 1:
                count -= 1
            return max(1, count)

        total_syllables = sum(count_syllables(w) for w in words)
        n_sentences = max(1, len(sentences))
        n_words = max(1, len(words))

        grade = 0.39 * (n_words / n_sentences) + 11.8 * (total_syllables / n_words) - 15.59
        return max(0.0, grade)
