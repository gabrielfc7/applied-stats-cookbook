# Mathematical Foundations: Non-Parametric Rank Inference

This document details the rank-based statistical tests implemented in `applied_stats.nonparametric`.

---

## 1. Mann-Whitney U Test (Wilcoxon Rank-Sum)

Tests whether the probability that an observation from population $X$ exceeds an observation from population $Y$ is equal to $0.5$:

$$H_0: P(X > Y) = P(Y > X) = 0.5$$

### 1.1 Ranking and U-Statistic

Pool all $N = n_1 + n_2$ observations and assign fractional ranks. Let $R_1$ denote the sum of ranks for sample 1:

$$U_1 = R_1 - \frac{n_1(n_1 + 1)}{2}$$

$$U_2 = R_2 - \frac{n_2(n_2 + 1)}{2} = n_1 n_2 - U_1$$

The test statistic is $U = \min(U_1, U_2)$ or $U_1$ depending on alternative formulation.

### 1.2 Effect Sizes: Rank-Biserial Correlation & Cliff's $\delta$

The rank-biserial correlation $r$ measures the difference between proportions of concordant and discordant pairs:

$$r = 1 - \frac{2 U_1}{n_1 n_2} = \frac{U_2 - U_1}{n_1 n_2} \in [-1, 1]$$

Cliff's $\delta$ is numerically identical to $r$:

$$\delta = P(X > Y) - P(X < Y)$$

---

## 2. Wilcoxon Signed-Rank Test

Non-parametric analog to the paired t-test. Evaluates whether the median difference of paired observations equals zero.

### 2.1 Ranking Differences

1. Calculate paired differences $d_i = x_i - y_i$.
2. Exclude zero differences under the Wilcoxon zero-method: $d_i \ne 0$, resulting in $n_r$ reduced pairs.
3. Rank the absolute differences $|d_i|$.
4. Sum ranks according to sign:

$$W^+ = \sum_{d_i > 0} \text{Rank}(|d_i|), \quad W^- = \sum_{d_i < 0} \text{Rank}(|d_i|)$$

### 2.2 Matched-Pairs Rank-Biserial Correlation

$$r = \frac{W^+ - W^-}{W^+ + W^-} = \frac{W^+ - W^-}{n_r(n_r + 1) / 2}$$

---

## 3. Kruskal-Wallis H Test

Non-parametric one-way ANOVA across $k \ge 2$ independent groups.

### 3.1 H-Statistic Formulation

Pool all $N = \sum_{j=1}^k n_j$ samples and assign average ranks. Let $R_j$ denote the rank sum of group $j$:

$$H = \frac{12}{N(N + 1)} \sum_{j=1}^k \frac{R_j^2}{n_j} - 3(N + 1)$$

### 3.2 Tie Correction

In the presence of tied observations with group sizes $t_i$:

$$C = 1 - \frac{\sum_i (t_i^3 - t_i)}{N^3 - N}, \quad H_{\text{corrected}} = \frac{H}{C}$$

Under $H_0$, $H_{\text{corrected}} \xrightarrow{d} \chi^2(\nu = k - 1)$.

### 3.3 Effect Size: Epsilon-Squared ($\epsilon^2$)

$$\epsilon^2 = \frac{H - k + 1}{N - k}$$

Represents the proportion of variance in ranks accounted for by group membership.
