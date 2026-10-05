"""Test fixtures for applied-stats-cookbook."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def one_sample_data():
    rng = np.random.RandomState(42)
    return rng.normal(loc=5.5, scale=1.5, size=35)


@pytest.fixture
def two_sample_data():
    rng = np.random.RandomState(42)
    s1 = rng.normal(loc=10.0, scale=2.0, size=30)
    s2 = rng.normal(loc=12.5, scale=2.5, size=25)
    return s1, s2


@pytest.fixture
def paired_data():
    rng = np.random.RandomState(42)
    before = rng.normal(loc=100.0, scale=10.0, size=30)
    after = before + rng.normal(loc=4.0, scale=3.0, size=30)
    return before, after


@pytest.fixture
def multi_group_data():
    rng = np.random.RandomState(42)
    g1 = rng.normal(loc=10.0, scale=1.5, size=25)
    g2 = rng.normal(loc=12.0, scale=1.8, size=25)
    g3 = rng.normal(loc=15.0, scale=1.6, size=25)
    return g1, g2, g3


@pytest.fixture
def factorial_data():
    rng = np.random.RandomState(42)
    data = []
    for a in ["Low", "High"]:
        for b in ["Standard", "Novel"]:
            mu = 10.0 + (3.0 if a == "High" else 0.0) + (4.0 if b == "Novel" else 0.0)
            vals = rng.normal(loc=mu, scale=1.5, size=15)
            for v in vals:
                data.append({"FactorA": a, "FactorB": b, "Yield": v})
    return pd.DataFrame(data)
