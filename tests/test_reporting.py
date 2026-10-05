"""Tests for publication reporting and formatting."""

import pytest

from applied_stats.parametric import one_sample_t_test, two_sample_t_test
from applied_stats.nonparametric import mann_whitney_u_test
from applied_stats.reporting import PublicationReporter


def test_publication_reporter_markdown_and_latex(two_sample_data, one_sample_data):
    s1, s2 = two_sample_data
    r1 = two_sample_t_test(s1, s2)
    r2 = one_sample_t_test(one_sample_data, popmean=5.0)
    r3 = mann_whitney_u_test(s1, s2)

    reporter = PublicationReporter([r1, r2, r3])
    df = reporter.to_dataframe()
    assert len(df) == 3
    assert "Statistic" in df.columns
    assert "Effect Size" in df.columns

    md_str = reporter.to_markdown()
    assert "| Test" in md_str
    assert "Welch's Two-Sample t-test" in md_str

    latex_str = reporter.to_latex()
    assert "\\begin{tabular}" in latex_str

    apa = reporter.apa_summary()
    assert "t(" in apa
