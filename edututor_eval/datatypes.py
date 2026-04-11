"""Data models for the evaluation framework.

All data flowing through the pipeline has a well-defined schema.
This makes it easy to validate inputs, serialize results, and
catch schema drift early — which matters in production when
upstream data formats change without warning.

Uses pydantic if available, otherwise falls back to dataclasses.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

try:
    from pydantic import BaseModel, Field

    _PYDANTIC = True
except ImportError:
    # Lightweight fallback when pydantic isn't installed
    from dataclasses import dataclass, field as _dc_field

    _PYDANTIC = False

    class BaseModel:
        """Minimal pydantic-like base using dataclass semantics."""

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
            # Apply defaults for missing fields
            for k, v in getattr(self.__class__, "__field_defaults__", {}).items():
                if not hasattr(self, k):
                    setattr(self, k, v() if callable(v) else v)

        def model_dump(self) -> dict:
            result = {}
            for k in vars(self):
                v = getattr(self, k)
                if isinstance(v, list):
                    result[k] = [
                        item.model_dump() if hasattr(item, "model_dump") else item
                        for item in v
                    ]
                elif isinstance(v, Enum):
                    result[k] = v.value
                elif hasattr(v, "model_dump"):
                    result[k] = v.model_dump()
                else:
                    result[k] = v
            return result

    def Field(default=..., default_factory=None, **kwargs):
        if default_factory is not None:
            return default_factory()
        if default is not ...:
            return default
        return None


class Subject(str, Enum):
    MATH = "math"
    SCIENCE = "science"
    ENGLISH = "english"
    HISTORY = "history"


class GradeLevel(str, Enum):
    K_2 = "K-2"
    GRADE_3_5 = "3-5"
    GRADE_6_8 = "6-8"
    GRADE_9_12 = "9-12"


class TutorResponse(BaseModel):
    """A single AI tutor response to evaluate.

    This mirrors what a tutoring system would log: the student's question,
    the AI's response, and the curriculum context needed to judge whether
    the response was appropriate.
    """

    if not _PYDANTIC:
        __field_defaults__ = {
            "topic": "",
            "conversation_history": list,
            "human_score": None,
            "human_labels": None,
        }

    id: str
    student_question: str
    tutor_response: str
    subject: Subject
    grade_level: GradeLevel
    topic: str = ""
    conversation_history: list = Field(default_factory=list)
    human_score: Optional[float] = None
    human_labels: Optional[dict] = None


class DimensionScore(BaseModel):
    """Score for a single evaluation dimension."""

    if not _PYDANTIC:
        __field_defaults__ = {
            "confidence": 1.0,
            "reasoning": "",
        }

    dimension: str
    score: float
    confidence: float = 1.0
    reasoning: str = ""


class EvalResult(BaseModel):
    """Complete evaluation result for one tutor response."""

    if not _PYDANTIC:
        __field_defaults__ = {
            "dimension_scores": list,
            "flags": list,
            "metadata": dict,
        }

    response_id: str
    evaluator: str
    overall_score: float
    dimension_scores: list = Field(default_factory=list)
    flags: list = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class EvalBatch(BaseModel):
    """A batch of evaluation results with summary statistics."""

    if not _PYDANTIC:
        __field_defaults__ = {
            "mean_score": 0.0,
            "score_std": 0.0,
        }

    results: list
    evaluator: str
    mean_score: float = 0.0
    score_std: float = 0.0

    def compute_summary(self) -> None:
        if self.results:
            scores = [r.overall_score for r in self.results]
            self.mean_score = sum(scores) / len(scores)
            mean = self.mean_score
            self.score_std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
