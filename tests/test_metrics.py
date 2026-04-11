"""Tests for evaluation metrics.

These tests verify that each evaluator:
1. Returns valid EvalResult objects
2. Produces scores in the expected range (1-5)
3. Correctly identifies known quality signals
"""

import pytest

from edututor_eval.datatypes import TutorResponse, Subject, GradeLevel
from edututor_eval.metrics.rule_based import RuleBasedEvaluator
from edututor_eval.metrics.learned import LearnedEvaluator, extract_pedagogical_features


@pytest.fixture
def good_response() -> TutorResponse:
    return TutorResponse(
        id="test_good",
        student_question="How do I solve 3x + 7 = 22?",
        tutor_response=(
            "Great question! Let's work through this step by step. "
            "First, what do you think we should do to get x by itself? "
            "Hint: try to get rid of the 7 first. What operation would undo adding 7? "
            "Take your time — you're doing great!"
        ),
        subject=Subject.MATH,
        grade_level=GradeLevel.GRADE_6_8,
        topic="linear_equations",
        human_score=4.5,
    )


@pytest.fixture
def bad_response() -> TutorResponse:
    return TutorResponse(
        id="test_bad",
        student_question="How do I solve 3x + 7 = 22?",
        tutor_response="The answer is 5.",
        subject=Subject.MATH,
        grade_level=GradeLevel.GRADE_6_8,
        topic="linear_equations",
        human_score=1.5,
    )


class TestRuleBasedEvaluator:
    def setup_method(self):
        self.evaluator = RuleBasedEvaluator()

    def test_returns_valid_result(self, good_response):
        result = self.evaluator.evaluate(good_response)
        assert result.response_id == "test_good"
        assert result.evaluator == "rule_based"
        assert 1.0 <= result.overall_score <= 5.0

    def test_good_response_scores_higher(self, good_response, bad_response):
        good_result = self.evaluator.evaluate(good_response)
        bad_result = self.evaluator.evaluate(bad_response)
        assert good_result.overall_score > bad_result.overall_score

    def test_answer_leak_detected(self, bad_response):
        result = self.evaluator.evaluate(bad_response)
        assert "answer_leaked" in result.flags

    def test_scaffolding_rewarded(self, good_response):
        result = self.evaluator.evaluate(good_response)
        ped_score = next(
            (d.score for d in result.dimension_scores
             if d.dimension == "pedagogical_alignment"),
            None,
        )
        assert ped_score is not None
        assert ped_score >= 3.5  # should be above average

    def test_short_response_penalized(self):
        short = TutorResponse(
            id="test_short",
            student_question="What is 2+2?",
            tutor_response="4.",
            subject=Subject.MATH,
            grade_level=GradeLevel.K_2,
        )
        result = self.evaluator.evaluate(short)
        assert "response_too_short" in result.flags

    def test_all_dimensions_scored(self, good_response):
        result = self.evaluator.evaluate(good_response)
        dims = {d.dimension for d in result.dimension_scores}
        expected = {"correctness", "pedagogical_alignment", "curriculum_grounding",
                    "engagement", "safety"}
        assert dims == expected


class TestLearnedEvaluator:
    def setup_method(self):
        # No model path → uses feature-based fallback
        self.evaluator = LearnedEvaluator()

    def test_fallback_returns_valid_result(self, good_response):
        result = self.evaluator.evaluate(good_response)
        assert result.response_id == "test_good"
        assert 1.0 <= result.overall_score <= 5.0

    def test_feature_extraction(self):
        text = (
            "Great question! Let's think about this step by step. "
            "What do you think the first step would be?"
        )
        features = extract_pedagogical_features(text)
        assert len(features) == 7
        assert features[0] == 1.0  # has_scaffolding
        assert features[1] == 1.0  # has_question_back


class TestPedagogicalFeatures:
    def test_scaffolding_detected(self):
        text = "Let's think about what we know. Can you try the first step?"
        feats = extract_pedagogical_features(text)
        assert feats[0] == 1.0  # scaffolding

    def test_answer_leak_detected(self):
        text = "The answer is 42. Done."
        feats = extract_pedagogical_features(text)
        assert feats[4] == 1.0  # answer_leaked

    def test_empty_text(self):
        feats = extract_pedagogical_features("")
        assert len(feats) == 7
        assert all(f == 0.0 for f in feats[:5])
