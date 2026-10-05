# Mathematical Foundations: Resampling Methods (Bootstrap & Permutation)

This document details the non-parametric resampling methods implemented in `applied_stats.resampling`.

---

## 1. Vectorized Bootstrap Confidence Intervals

The bootstrap approximates the sampling distribution of an arbitrary estimator $\hat{\theta} = T(F_n)$ by repeatedly drawing $B$ samples of size $n$ with replacement from the empirical distribution function $F_n$:

$$\mathbf{x}^{*b} = \{x_1^{*b}, x_2^{*b}, \dots, x_n^{*b}\} \sim F_n$$

$$\hat{\theta}^{*b} = T(\mathbf{x}^{*b}), \quad b = 1, \dots, B$$

### 1.1 Percentile Bootstrap

Given ordered bootstrap estimates $\hat{\theta}^{*(1)} \le \hat{\theta}^{*(2)} \le \dots \le \hat{\theta}^{*(B)}$:

$$\text{CI}_{\text{percentile}} = \left[ \hat{\theta}^{*(\alpha/2)}, \; \hat{\theta}^{*(1 - \alpha/2)} \right]$$

### 1.2 Basic (Empirical) Bootstrap

$$\text{CI}_{\text{basic}} = \left[ 2\hat{\theta} - \hat{\theta}^{*(1 - \alpha/2)}, \; 2\hat{\theta} - \hat{\theta}^{*(\alpha/2)} \right]$$

### 1.3 Bias-Corrected and Accelerated (BCa) Bootstrap

Corrects for median bias and variance stabilization (skewness):

$$\text{CI}_{\text{BCa}} = \left[ \hat{\theta}^{*(\alpha_1)}, \; \hat{\theta}^{*(\alpha_2)} \right]$$

where:

$$\alpha_1 = \Phi\left( z_0 + \frac{z_0 + z_{\alpha/2}}{1 - a(z_0 + z_{\alpha/2})} \right)$$

$$\alpha_2 = \Phi\left( z_0 + \frac{z_0 + z_{1 - \alpha/2}}{1 - a(z_0 + z_{1 - \alpha/2})} \right)$$

* **Bias parameter $z_0$:**
  $$z_0 = \Phi^{-1}\left( \frac{1}{B} \sum_{b=1}^B \mathbf{1}(\hat{\theta}^{*b} < \hat{\theta}) \right)$$

* **Acceleration parameter $a$ via Jackknife Influence Values $\hat{\theta}_{(i)}$:**
  $$a = \frac{\sum_{i=1}^n (\bar{\theta}_{(\cdot)} - \hat{\theta}_{(i)})^3}{6 \left( \sum_{i=1}^n (\bar{\theta}_{(\cdot)} - \hat{\theta}_{(i)})^2 \right)^{3/2}}$$

---

## 2. Two-Sample Permutation Tests

Under the null hypothesis of exchangeability $H_0: F_X = F_Y$, observations from groups $X$ and $Y$ are identical draws from a common distribution.

### 2.1 Test Statistic & Monte Carlo Randomization

Pool all $N = n_1 + n_2$ observations. Generate $B$ random permutations without replacement, partitioning into subsets of size $n_1$ and $n_2$:

$$T^{*b} = T(\mathbf{x}^{*b}, \mathbf{y}^{*b})$$

### 2.2 Exact Empirical p-Value

With Davison & Hinkley (1997) continuity adjustment:

$$p_{\text{two-sided}} = \frac{1 + \sum_{b=1}^B \mathbf{1}(|T^{*b}| \ge |T_{\text{obs}}|)}{1 + B}$$
