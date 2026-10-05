"""Parametric hypothesis testing module."""

from applied_stats.parametric.t_tests import (
    one_sample_t_test,
    paired_t_test,
    two_sample_t_test,
)

__all__ = [
    "one_sample_t_test",
    "two_sample_t_test",
    "paired_t_test",
]
