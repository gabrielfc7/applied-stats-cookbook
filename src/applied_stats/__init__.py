"""applied-stats-cookbook: Computational reference for statistical inference, tests, and resampling."""

__version__ = "0.1.0"

from applied_stats.anova import ANOVATable, one_way_anova, tukey_hsd, two_way_anova
from applied_stats.base import TestResult
from applied_stats.nonparametric import (
    kruskal_wallis_test,
    mann_whitney_u_test,
    wilcoxon_signed_rank_test,
)
from applied_stats.parametric import (
    one_sample_t_test,
    paired_t_test,
    two_sample_t_test,
)
from applied_stats.reporting import PublicationReporter
from applied_stats.resampling import (
    BootstrapResult,
    PermutationResult,
    bootstrap_ci,
    permutation_test,
)

__all__ = [
    "TestResult",
    "one_sample_t_test",
    "two_sample_t_test",
    "paired_t_test",
    "mann_whitney_u_test",
    "wilcoxon_signed_rank_test",
    "kruskal_wallis_test",
    "one_way_anova",
    "two_way_anova",
    "tukey_hsd",
    "ANOVATable",
    "bootstrap_ci",
    "permutation_test",
    "BootstrapResult",
    "PermutationResult",
    "PublicationReporter",
]
