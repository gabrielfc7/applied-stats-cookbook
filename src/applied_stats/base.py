"""Core dataclasses and common utilities for statistical test results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class TestResult:
    """Standard container for hypothesis testing outputs and effect size metrics."""

    test_name: str
    statistic_name: str
    statistic: float
    p_value: float
    df: Optional[Union[float, int, tuple[Union[float, int], ...]]] = None
    effect_size: Optional[float] = None
    effect_size_name: Optional[str] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    ci_level: float = 0.95
    alternative: str = "two-sided"
    null_value: float = 0.0
    sample_sizes: tuple[int, ...] = field(default_factory=tuple)
    extra_info: dict[str, Any] = field(default_factory=dict)

    @property
    def is_significant(self) -> bool:
        """Check if result is statistically significant at alpha = 1 - ci_level."""
        alpha = 1.0 - self.ci_level
        return self.p_value < alpha

    def apa_format(self) -> str:
        """Format the test result in APA 7th edition reporting style."""
        stat_symbol = self.statistic_name
        p_str = "< .001" if self.p_value < 0.001 else f"= {self.p_value:.3f}".replace("0.", ".")

        df_str = ""
        if self.df is not None:
            if isinstance(self.df, (tuple, list)):
                df_str = f"({', '.join(str(d) for d in self.df)})"
            else:
                df_str = f"({self.df})"

        result = f"{stat_symbol}{df_str} = {self.statistic:.2f}, p {p_str}"

        if self.ci_lower is not None and self.ci_upper is not None:
            ci_pct = int(self.ci_level * 100)
            result += f", {ci_pct}% CI [{self.ci_lower:.2f}, {self.ci_upper:.2f}]"

        if self.effect_size is not None and self.effect_size_name is not None:
            result += f", {self.effect_size_name} = {self.effect_size:.2f}"

        return result

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "test_name": self.test_name,
            "statistic_name": self.statistic_name,
            "statistic": float(self.statistic),
            "p_value": float(self.p_value),
            "df": self.df,
            "effect_size_name": self.effect_size_name,
            "effect_size": float(self.effect_size) if self.effect_size is not None else None,
            "ci_level": self.ci_level,
            "ci_lower": float(self.ci_lower) if self.ci_lower is not None else None,
            "ci_upper": float(self.ci_upper) if self.ci_upper is not None else None,
            "alternative": self.alternative,
            "sample_sizes": self.sample_sizes,
            **self.extra_info,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert result to a 1-row pandas DataFrame."""
        return pd.DataFrame([self.to_dict()])

    def __str__(self) -> str:
        return self.apa_format()


def check_sample(data: Any, name: str = "sample", min_size: int = 2) -> np.ndarray:
    """Validate sample array and remove NaN / inf values."""
    arr = np.asarray(data, dtype=np.float64).ravel()
    arr = arr[~np.isnan(arr) & ~np.isinf(arr)]
    if len(arr) < min_size:
        raise ValueError(f"Sample '{name}' must contain at least {min_size} valid observations (got {len(arr)}).")
    return arr
