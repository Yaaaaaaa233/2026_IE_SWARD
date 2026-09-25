"""Independent task-1 probability distillation route for task 2."""

from .model import (
    CrossFitResult,
    DistilledRuleset,
    DistilledScoring,
    FeatureSpec,
    build_feature_specs,
    cross_fit_distillation,
    evaluate_fidelity,
    fit_ruleset,
    recompute_score,
    score_vehicle,
)

__all__ = [
    "CrossFitResult",
    "DistilledRuleset",
    "DistilledScoring",
    "FeatureSpec",
    "build_feature_specs",
    "cross_fit_distillation",
    "evaluate_fidelity",
    "fit_ruleset",
    "recompute_score",
    "score_vehicle",
]
