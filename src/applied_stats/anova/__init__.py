"""Analysis of Variance (ANOVA) and post-hoc tests module."""

from applied_stats.anova.multi_group import (
    ANOVATable,
    one_way_anova,
    tukey_hsd,
    two_way_anova,
)

__all__ = [
    "ANOVATable",
    "one_way_anova",
    "two_way_anova",
    "tukey_hsd",
]
