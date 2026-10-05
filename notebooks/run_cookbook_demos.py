"""Demonstration script executing the statistical testing workflows and printing publication tables."""

import numpy as np
import pandas as pd

from applied_stats.parametric import one_sample_t_test, two_sample_t_test, paired_t_test
from applied_stats.nonparametric import mann_whitney_u_test, wilcoxon_signed_rank_test, kruskal_wallis_test
from applied_stats.anova import one_way_anova, two_way_anova, tukey_hsd
from applied_stats.resampling import bootstrap_ci, permutation_test
from applied_stats.reporting import PublicationReporter


import sys
sys.stdout.reconfigure(encoding='utf-8')

def main():
    print("=" * 80)
    print("           APPLIED-STATS-COOKBOOK DEMONSTRATION & VERIFICATION")
    print("=" * 80)

    rng = np.random.RandomState(42)

    # 1. Parametric Testing: Two-sample Welch vs Student
    s1 = rng.normal(loc=10.0, scale=2.0, size=35)
    s2 = rng.normal(loc=12.2, scale=3.1, size=30)
    t_res_welch = two_sample_t_test(s1, s2, equal_var=False)
    t_res_student = two_sample_t_test(s1, s2, equal_var=True)

    # 2. Non-Parametric Testing: Mann-Whitney U
    mwu_res = mann_whitney_u_test(s1, s2)

    # 3. Paired testing
    pre = rng.normal(100, 10, 25)
    post = pre + rng.normal(4.5, 3.0, 25)
    paired_res = paired_t_test(post, pre)

    # Compile publication table
    reporter = PublicationReporter([t_res_welch, t_res_student, mwu_res, paired_res])
    print("\n### 1. Publication-Ready Summary Table:")
    print(reporter.to_markdown())

    print("\n### 2. APA 7th Edition Summaries:")
    print(reporter.apa_summary())

    # 4. Multi-Group ANOVA and Tukey HSD
    g1 = rng.normal(50, 5, 20)
    g2 = rng.normal(54, 6, 20)
    g3 = rng.normal(62, 5, 20)
    anova_res = one_way_anova(g1, g2, g3)
    print("\n### 3. One-Way ANOVA Summary:")
    print(anova_res.to_dataframe().to_string())

    print(f"\nEta-squared (eta^2): {anova_res.eta_squared:.4f}")
    print(f"Omega-squared (omega^2): {anova_res.omega_squared:.4f}")

    print("\n### 4. Post-Hoc Tukey HSD Pairwise Comparisons:")
    hsd_df = tukey_hsd(g1, g2, g3)
    print(hsd_df.to_markdown(index=False))

    # 5. Resampling Methods: Vectorized Bootstrap & Permutation Test
    boot_mean = bootstrap_ci(g3, statistic_fn=np.mean, n_resamples=5000, method="bca", random_state=42)
    print("\n### 5. Resampling Inference:")
    print(boot_mean.summary())

    perm_res = permutation_test(g1, g3, n_permutations=5000, random_state=42)
    print(perm_res.summary())
    print("=" * 80)


if __name__ == "__main__":
    main()
