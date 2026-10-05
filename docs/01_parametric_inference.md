# Mathematical Foundations: Parametric Statistical Inference

This document details the mathematical formulations and derivations implemented in `applied_stats.parametric`.

---

## 1. One-Sample Student's t-Test

Let $X = \{x_1, x_2, \dots, x_n\}$ be an independent sample drawn from $\mathcal{N}(\mu, \sigma^2)$ with hypothesized population mean $\mu_0$.

### 1.1 Sample Estimators

$$\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i, \quad s = \sqrt{\frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x})^2}$$

Standard error of the sample mean:

$$SE = \frac{s}{\sqrt{n}}$$

### 1.2 Test Statistic and Distribution

Under the null hypothesis $H_0: \mu = \mu_0$:

$$t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}} \sim t(\nu = n - 1)$$

### 1.3 Effect Size: Cohen's $d$

$$d = \frac{\bar{x} - \mu_0}{s}$$

Conventional benchmarks: small ($|d| \approx 0.2$), medium ($|d| \approx 0.5$), large ($|d| \ge 0.8$).

### 1.4 Confidence Interval

$$(1 - \alpha) \cdot 100\% \text{ CI} = \left[ \bar{x} - t_{1 - \alpha/2, \nu} \cdot SE, \; \bar{x} + t_{1 - \alpha/2, \nu} \cdot SE \right]$$

---

## 2. Independent Two-Sample t-Tests

Let $X_1 \sim \mathcal{N}(\mu_1, \sigma_1^2)$ of size $n_1$ and $X_2 \sim \mathcal{N}(\mu_2, \sigma_2^2)$ of size $n_2$.

### 2.1 Student's t-Test (Equal Variance Assumed: $\sigma_1^2 = \sigma_2^2$)

Pooled sample variance:

$$s_p^2 = \frac{(n_1 - 1) s_1^2 + (n_2 - 1) s_2^2}{n_1 + n_2 - 2}$$

Standard error and test statistic:

$$SE_p = s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}, \quad t = \frac{\bar{x}_1 - \bar{x}_2}{SE_p} \sim t(\nu = n_1 + n_2 - 2)$$

#### Hedges' $g$ Correction

Because sample Cohen's $d = \frac{\bar{x}_1 - \bar{x}_2}{s_p}$ exhibits positive small-sample bias, Hedges' $g$ applies an analytical correction factor $J$:

$$g = d \cdot J(n_1 + n_2 - 2) \approx d \cdot \left( 1 - \frac{3}{4(n_1 + n_2) - 9} \right)$$

### 2.2 Welch's t-Test (Unequal Variance: $\sigma_1^2 \ne \sigma_2^2$)

Unpooled standard error:

$$SE_w = \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}, \quad t = \frac{\bar{x}_1 - \bar{x}_2}{SE_w}$$

#### Welch–Satterthwaite Degrees of Freedom

$$\nu = \frac{\left( \frac{s_1^2}{n_1} + \frac{s_2^2}{n_2} \right)^2}{\frac{(s_1^2 / n_1)^2}{n_1 - 1} + \frac{(s_2^2 / n_2)^2}{n_2 - 1}}$$

Welch's t-test protects nominal Type I error rates against heteroskedasticity and unbalanced sample sizes.

---

## 3. Paired Samples t-Test

For paired observations $(x_i, y_i)$, compute differences $d_i = x_i - y_i$. The paired test reduces to a 1-sample test on differences:

$$t = \frac{\bar{d}}{s_d / \sqrt{n}} \sim t(\nu = n - 1)$$

Effect size (Cohen's $d_z$):

$$d_z = \frac{\bar{d}}{s_d}$$
