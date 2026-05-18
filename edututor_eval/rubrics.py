"""Rubric loading and validation.

Rubrics are the heart of the system. By externalizing evaluation criteria
into YAML, we let educators define what "good" means without writing code.
This separation of concerns is critical in EdTech — the people who
understand pedagogy aren't always the ones writing Python.

A rubric defines:
- Which dimensions to evaluate (correctness, pedagogy, engagement, etc.)
- What each score level (1-5) looks like for each dimension
- How to weight dimensions into an overall score
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ScoreLevel:
    """Description of what a score level means for a dimension."""

    level: int  # 1-5
    description: str
    indicators: list[str] = field(default_factory=list)


@dataclass
class Dimension:
    """A single evaluation dimension with scoring rubric."""

    name: str
    description: str
    weight: float = 1.0
    score_levels: list[ScoreLevel] = field(default_factory=list)

    def get_level_description(self, level: int) -> str:
        for sl in self.score_levels:
            if sl.level == level:
                return sl.description
        return ""


@dataclass
class Rubric:
    """Complete evaluation rubric."""

    name: str
    version: str
    description: str
    dimensions: list[Dimension]

    @property
    def dimension_names(self) -> list[str]:
        return [d.name for d in self.dimensions]

    @property
    def total_weight(self) -> float:
        return sum(d.weight for d in self.dimensions)

    def weighted_score(self, scores: dict[str, float]) -> float:
        """Compute weighted overall score from per-dimension scores."""
        total = 0.0
        weight_sum = 0.0
        for dim in self.dimensions:
            if dim.name in scores:
                total += scores[dim.name] * dim.weight
                weight_sum += dim.weight
        return total / weight_sum if weight_sum > 0 else 0.0

    def to_prompt_text(self) -> str:
        """Format rubric as text for inclusion in LLM prompts.

        This is how we bridge the structured rubric definition and
        the LLM-as-judge: the rubric gets serialized into the prompt
        so the LLM knows exactly what to evaluate and how.
        """
        lines = [f"# Evaluation Rubric: {self.name}\n"]
        for dim in self.dimensions:
            lines.append(f"## {dim.name} (weight: {dim.weight})")
            lines.append(f"{dim.description}\n")
            for sl in dim.score_levels:
                indicators = ", ".join(sl.indicators) if sl.indicators else ""
                indicator_text = f" [{indicators}]" if indicators else ""
                lines.append(f"  {sl.level}/5: {sl.description}{indicator_text}")
            lines.append("")
        return "\n".join(lines)


def load_rubric(path: str | Path) -> Rubric:
    """Load a rubric from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Rubric file not found: {path}")
    with open(path) as f:
        data: dict[str, Any] = yaml.safe_load(f)

    if not data:
        raise ValueError(f"Rubric file is empty: {path}")

    dimensions = []
    for i, dim_data in enumerate(data.get("dimensions", [])):
        if "name" not in dim_data:
            raise ValueError(f"Dimension {i} missing required field 'name' in {path}")
        weight = dim_data.get("weight", 1.0)

        if weight < 0:
            raise ValueError(
                f"Dimension '{dim_data['name']}' has negative weight: {weight}"
            )

        score_levels = [
            ScoreLevel(
                level=sl["level"],
                description=sl["description"],
                indicators=sl.get("indicators", []),
            )
            for sl in dim_data.get("score_levels", [])
        ]
        dimensions.append(
            Dimension(
                name=dim_data["name"],
                description=dim_data.get("description", ""),
                weight=dim_data.get("weight", 1.0),
                score_levels=score_levels,
            )
        )
    if not dimensions:
        raise ValueError(f"Rubric must have at least one dimension: {path}")

    total_weight = sum(d.weight for d in dimensions)
    if total_weight == 0:
        raise ValueError(f"Rubric dimensions sum to zero weight in {path}")

    return Rubric(
        name=data.get("name", "Unnamed"),
        version=data.get("version", "1.0"),
        description=data.get("description", ""),
        dimensions=dimensions,
    )
