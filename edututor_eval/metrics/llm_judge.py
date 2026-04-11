"""LLM-as-Judge evaluator for AI tutor responses.

This is the most capable (and most expensive) evaluation strategy.
The idea: use a strong LLM (GPT-4, Claude) as an automated evaluator,
providing it with a structured rubric and asking for per-dimension scores
with chain-of-thought reasoning.

Key design decisions:
- **Structured output** — We ask for JSON to make parsing reliable.
- **Chain-of-thought** — Requiring reasoning before scores reduces
  position bias and improves calibration (Zheng et al., 2023).
- **Rubric in prompt** — The rubric YAML gets serialized into the
  prompt, so the LLM evaluates against the same criteria humans use.
- **Retry logic** — LLM outputs are stochastic; we retry on parse
  failures rather than crashing.

Trade-offs vs. other approaches:
- More expensive than rule-based (~$0.02-0.05 per evaluation with GPT-4)
- Slower (~2-5 seconds per evaluation)
- But captures nuance that rules miss (e.g., "technically correct but
  pedagogically confusing")
- Agreement with human annotators is typically higher than rule-based
  but lower than fine-tuned specialized models
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

from edututor_eval.datatypes import TutorResponse, EvalResult, DimensionScore
from edututor_eval.rubrics import Rubric

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """You are an expert educational AI evaluator. Your task is to assess the quality of an AI tutor's response to a student question.

You will be given:
1. The student's question
2. The AI tutor's response
3. Context (subject, grade level)
4. An evaluation rubric with specific dimensions and scoring criteria

For each dimension in the rubric, provide:
- A score from 1 to 5
- A brief justification (1-2 sentences)

Think step by step before scoring. Consider what a skilled human tutor would do differently.

Respond in JSON format:
{
  "reasoning": "Your overall analysis of the response...",
  "dimensions": {
    "<dimension_name>": {"score": <1-5>, "justification": "..."},
    ...
  },
  "overall_score": <1-5>,
  "flags": ["flag1", "flag2"]
}"""


def build_judge_prompt(
    response: TutorResponse,
    rubric: Rubric,
) -> str:
    """Construct the evaluation prompt for the LLM judge.

    Separating prompt construction from API calls makes this
    testable and provider-agnostic.
    """
    rubric_text = rubric.to_prompt_text()

    prompt = f"""## Student Question
{response.student_question}

## AI Tutor Response
{response.tutor_response}

## Context
- Subject: {response.subject.value}
- Grade Level: {response.grade_level.value}
- Topic: {response.topic}

## Evaluation Rubric
{rubric_text}

Please evaluate the AI tutor's response according to the rubric above. Provide your analysis as JSON."""

    return prompt


class LLMJudgeEvaluator:
    """Evaluate tutor responses using an LLM as judge.

    Supports OpenAI and Anthropic backends. Falls back to a mock
    implementation for testing without API keys.
    """

    def __init__(
        self,
        rubric: Rubric,
        provider: str = "openai",
        model: str = "gpt-4o",
        temperature: float = 0.1,
        max_retries: int = 2,
    ):
        self.rubric = rubric
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self._client = None

    def _get_client(self):
        """Lazy-initialize the API client."""
        if self._client is not None:
            return self._client

        if self.provider == "openai":
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except ImportError:
                logger.warning("openai not installed; using mock evaluator")
                self._client = None
        elif self.provider == "anthropic":
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            except ImportError:
                logger.warning("anthropic not installed; using mock evaluator")
                self._client = None
        return self._client

    def evaluate(self, response: TutorResponse) -> EvalResult:
        """Evaluate a single tutor response."""
        prompt = build_judge_prompt(response, self.rubric)
        client = self._get_client()

        if client is None:
            return self._mock_evaluate(response)

        for attempt in range(self.max_retries + 1):
            try:
                raw = self._call_llm(prompt)
                return self._parse_response(response.id, raw)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Parse failed (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries:
                    logger.error("All retries exhausted; returning default scores")
                    return self._default_result(response.id)

        return self._default_result(response.id)

    def _call_llm(self, prompt: str) -> str:
        """Make the API call. Separated for testability."""
        if self.provider == "openai":
            completion = self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            return completion.choices[0].message.content

        elif self.provider == "anthropic":
            message = self._client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=self.temperature,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text

        raise ValueError(f"Unknown provider: {self.provider}")

    def _parse_response(self, response_id: str, raw: str) -> EvalResult:
        """Parse LLM JSON output into an EvalResult.

        Robust parsing is important — LLMs don't always produce
        perfectly valid JSON, even with response_format set.
        """
        # Try to extract JSON from the response
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            raw = json_match.group()

        data = json.loads(raw)
        dimensions = data.get("dimensions", {})

        dimension_scores = [
            DimensionScore(
                dimension=dim_name,
                score=float(dim_data.get("score", 3.0)),
                confidence=0.8,
                reasoning=dim_data.get("justification", ""),
            )
            for dim_name, dim_data in dimensions.items()
        ]

        return EvalResult(
            response_id=response_id,
            evaluator="llm_judge",
            overall_score=float(data.get("overall_score", 3.0)),
            dimension_scores=dimension_scores,
            flags=data.get("flags", []),
            metadata={
                "model": self.model,
                "reasoning": data.get("reasoning", ""),
            },
        )

    def _mock_evaluate(self, response: TutorResponse) -> EvalResult:
        """Deterministic mock for testing without API keys.

        Uses simple heuristics to produce plausible scores,
        so tests can verify the pipeline end-to-end.
        """
        text = response.tutor_response
        word_count = len(text.split())
        has_question = "?" in text

        # Simple heuristic scores for testing
        base = 3.0
        if word_count > 30:
            base += 0.5
        if has_question:
            base += 0.5
        if word_count > 200:
            base -= 0.3

        dims = ["correctness", "pedagogical_alignment", "curriculum_grounding",
                "engagement", "safety"]
        dimension_scores = [
            DimensionScore(
                dimension=d,
                score=min(5.0, max(1.0, base + (0.2 * i - 0.4))),
                confidence=0.5,
                reasoning="Mock evaluation (no API key configured)",
            )
            for i, d in enumerate(dims)
        ]

        return EvalResult(
            response_id=response.id,
            evaluator="llm_judge_mock",
            overall_score=round(base, 2),
            dimension_scores=dimension_scores,
            flags=[],
            metadata={"model": "mock", "note": "Set OPENAI_API_KEY for real evaluation"},
        )

    def _default_result(self, response_id: str) -> EvalResult:
        """Fallback when parsing fails after all retries."""
        return EvalResult(
            response_id=response_id,
            evaluator="llm_judge",
            overall_score=3.0,
            dimension_scores=[],
            flags=["parse_failure"],
            metadata={"error": "Failed to parse LLM judge output"},
        )
