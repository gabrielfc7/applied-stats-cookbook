"""Publication-grade reporting utilities formatting statistical test outputs."""

from __future__ import annotations

from typing import Sequence, Union

import pandas as pd

from applied_stats.base import TestResult


class PublicationReporter:
    """Formatter generating publication-ready summary tables (Markdown, LaTeX, HTML, DataFrame)."""

    def __init__(self, results: Union[TestResult, Sequence[TestResult]] = ()):
        if isinstance(results, TestResult):
            self.results: list[TestResult] = [results]
        else:
            self.results = list(results)

    def add_result(self, result: TestResult) -> None:
        self.results.append(result)

    def to_dataframe(self) -> pd.DataFrame:
        """Format results into a clean tabular DataFrame."""
        rows = []
        for r in self.results:
            p_formatted = "< .001" if r.p_value < 0.001 else f"{r.p_value:.3f}".replace("0.", ".")

            df_str = ""
            if r.df is not None:
                if isinstance(r.df, (tuple, list)):
                    df_str = f"({', '.join(str(d) for d in r.df)})"
                else:
                    df_str = f"({r.df})"

            stat_str = f"{r.statistic_name}{df_str} = {r.statistic:.2f}"

            ci_str = "-"
            if r.ci_lower is not None and r.ci_upper is not None:
                ci_pct = int(r.ci_level * 100)
                ci_str = f"[{r.ci_lower:.2f}, {r.ci_upper:.2f}]"

            es_str = "-"
            if r.effect_size is not None and r.effect_size_name is not None:
                es_str = f"{r.effect_size:.2f} ({r.effect_size_name})"

            n_str = "/".join(str(n) for n in r.sample_sizes) if r.sample_sizes else "-"

            rows.append(
                {
                    "Test": r.test_name,
                    "N": n_str,
                    "Statistic": stat_str,
                    "p-value": p_formatted,
                    "95% CI": ci_str,
                    "Effect Size": es_str,
                    "Significance": "p < .05 *" if r.is_significant else "ns",
                }
            )

        return pd.DataFrame(rows)

    def to_markdown(self) -> str:
        """Format as Markdown table."""
        df = self.to_dataframe()
        return df.to_markdown(index=False)

    def to_latex(self) -> str:
        """Format as LaTeX tabular environment."""
        df = self.to_dataframe()
        return df.to_latex(index=False, escape=False)

    def apa_summary(self) -> str:
        """Format all tests as an APA-formatted paragraph."""
        return "\n".join(f"• {r.apa_format()}" for r in self.results)
