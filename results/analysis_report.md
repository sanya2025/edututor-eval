# EduTutor Evaluation Analysis Report

## Overview

- **Total evaluations**: 50
- **Unique responses evaluated**: 50
- **Evaluators used**: rule_based

## Score Distribution by Evaluator

| Evaluator | Mean | Std | Min | Median | Max | N |
|-----------|------|-----|-----|--------|-----|---|
| rule_based | 4.03 | 0.30 | 3.0 | 4.0 | 4.7 | 50 |

## Per-Dimension Scores

### rule_based

| Dimension | Mean | Std |
|-----------|------|-----|
| correctness | 3.97 | 0.21 |
| pedagogical_alignment | 3.70 | 0.83 |
| curriculum_grounding | 4.24 | 0.49 |
| engagement | 3.58 | 0.33 |
| safety | 4.94 | 0.22 |

## Quality Tier Distribution

| Evaluator | Low (1-2) | Medium (2-3.5) | High (3.5-5) |
|-----------|-----------|----------------|--------------|
| rule_based | 0 (0%) | 1 (2%) | 49 (98%) |

## Flags Detected

| Flag | Count | % of Evaluations |
|------|-------|-------------------|
| potentially_discouraging | 2 | 4.0% |
| meta_ai_reference | 2 | 4.0% |
| response_too_short | 1 | 2.0% |
| answer_leaked | 1 | 2.0% |

## Key Findings

1. **Strongest dimension**: safety (mean 4.94)
2. **Weakest dimension**: engagement (mean 3.58)
3. **Most common issue**: 'potentially_discouraging' (detected 2 times)

## Recommendations

1. Focus prompt improvement efforts on the weakest scoring dimension
2. Investigate responses flagged with quality issues for root cause patterns
3. Use evaluator agreement data to identify where automated scoring is unreliable and human review is needed
4. Consider the cost-quality trade-off: rule-based for pre-filtering, LLM-judge for spot-checking, learned model for production scoring