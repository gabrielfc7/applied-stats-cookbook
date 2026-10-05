# Mathematical Foundations: Analysis of Variance (ANOVA) & Post-Hoc Tests

This document formalizes sum-of-squares decompositions, effect sizes, and simultaneous inference implemented in `applied_stats.anova`.

---

## 1. One-Way Analysis of Variance

Evaluates equality of population means across $k \ge 2$ independent groups:

$$H_0: \mu_1 = \mu_2 = \dots = \mu_k$$

### 1.1 Sum of Squares Partitioning

Let $y_{ij}$ be the $i$-th sample in group $j$, $\bar{y}_j$ the group mean, and $\bar{y}_{\cdot\cdot}$ the grand mean across all $N = \sum_{j=1}^k n_j$ observations.

$$SS_{\text{total}} = \sum_{j=1}^k \sum_{i=1}^{n_j} (y_{ij} - \bar{y}_{\cdot\cdot})^2 = SS_{\text{between}} + SS_{\text{within}}$$

* **Between-Group (Treatment) SS:**
  $$SS_{\text{between}} = \sum_{j=1}^k n_j (\bar{y}_j - \bar{y}_{\cdot\cdot})^2, \quad df_{\text{between}} = k - 1$$

* **Within-Group (Residual / Error) SS:**
  $$SS_{\text{within}} = \sum_{j=1}^k \sum_{i=1}^{n_j} (y_{ij} - \bar{y}_j)^2, \quad df_{\text{within}} = N - k$$

### 1.2 Mean Squares and F-Statistic

$$MS_{\text{between}} = \frac{SS_{\text{between}}}{k - 1}, \quad MS_{\text{within}} = \frac{SS_{\text{within}}}{N - k}$$

$$F = \frac{MS_{\text{between}}}{MS_{\text{within}}} \sim F(k - 1, N - k)$$

### 1.3 Effect Sizes: Eta-Squared ($\eta^2$) vs. Omega-Squared ($\omega^2$)

$$\eta^2 = \frac{SS_{\text{between}}}{SS_{\text{total}}}$$

Because $\eta^2$ is an upwardly biased estimator of population variance accounted for, **Omega-squared ($\omega^2$)** provides an unbiased shrinkage estimate:

$$\omega^2 = \frac{SS_{\text{between}} - (k - 1) MS_{\text{within}}}{SS_{\text{total}} + MS_{\text{within}}}$$

---

## 2. Two-Way Factorial ANOVA

Evaluates main effects of factors $A$ and $B$ alongside interaction $A \times B$:

$$y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \epsilon_{ijk}$$

### 2.1 Decomposition

$$SS_{\text{total}} = SS_A + SS_B + SS_{AB} + SS_{\text{error}}$$

Degrees of freedom:
* $df_A = a - 1$
* $df_B = b - 1$
* $df_{AB} = (a - 1)(b - 1)$
* $df_{\text{error}} = N - ab$

Partial eta-squared for each effect:

$$\eta_{p, \text{effect}}^2 = \frac{SS_{\text{effect}}}{SS_{\text{effect}} + SS_{\text{error}}}$$

---

## 3. Tukey's Honestly Significant Difference (HSD)

Controls Family-Wise Error Rate (FWER) across all $C = \binom{k}{2}$ pairwise comparisons.

### 3.1 Studentized Range Statistic ($q$)

For any pair of groups $(i, j)$:

$$q = \frac{|\bar{y}_i - \bar{y}_j|}{\sqrt{\frac{MS_{\text{within}}}{2} \left( \frac{1}{n_i} + \frac{1}{n_j} \right)}} \sim q(k, df_{\text{within}})$$

### 3.2 Simultaneous Confidence Intervals

$$\bar{y}_i - \bar{y}_j \pm q_{1 - \alpha, k, df_{\text{within}}} \cdot \sqrt{\frac{MS_{\text{within}}}{2} \left( \frac{1}{n_i} + \frac{1}{n_j} \right)}$$
