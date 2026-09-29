"""Counterfactual explanations for trained pipelines, with CELIA.

Install the optional dependencies with ``pip install "nodeml[counterfactuals]"``.
The configuration and report classes work without them.
"""

from nodeml.core.pipeline.counterfactuals.config import (
    CounterfactualEvaluatorConfig,
    CounterfactualScenario,
    FeatureConstraints,
    InsertionPoint,
    OutputPoint,
    TargetRange,
)
from nodeml.core.pipeline.counterfactuals.evaluator import (
    CounterfactualEvaluator,
    ReferenceData,
)
from nodeml.core.pipeline.counterfactuals.report import (
    CounterfactualRecord,
    CounterfactualReport,
    SampleResult,
    ScenarioSummary,
)

__all__ = [
    "CounterfactualEvaluator",
    "CounterfactualEvaluatorConfig",
    "CounterfactualRecord",
    "CounterfactualReport",
    "CounterfactualScenario",
    "FeatureConstraints",
    "InsertionPoint",
    "OutputPoint",
    "ReferenceData",
    "SampleResult",
    "ScenarioSummary",
    "TargetRange",
]
