"""R3 white-box / black-box fusion: data-free experiment harness.

Implements roadmap D18-D20 mechanics: the alpha-weighted fusion
``S = alpha_q0 * Q0 + (1 - alpha_q0) * Q1``, the same-MOE alpha grid, the
rank-disagreement ammunition for human review, and the driver-facing talking
point format.  No real scores, labels or probabilities live here.
"""

from .fusion import (
    dimension_talking_point,
    fuse_scores,
    rank_disagreements,
    run_alpha_grid,
)

__all__ = [
    "dimension_talking_point",
    "fuse_scores",
    "rank_disagreements",
    "run_alpha_grid",
]
