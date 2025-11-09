# Chapter 4: Bayesian Learning

## Overview & Motivation

Bayesian learning provides a principled probabilistic framework for machine learning. Unlike frequentist approaches that find point estimates, Bayesian methods maintain probability distributions over hypotheses, naturally incorporating uncertainty and prior knowledge. This chapter covers Bayesian inference, Maximum A Posteriori (MAP) estimation, Minimum Description Length (MDL) principle, and the Bayes optimal classifier.

!!! important "Why Bayesian Learning?"
    Bayesian methods offer unique advantages:
    - **Uncertainty Quantification**: Provides probability distributions, not just point estimates
    - **Prior Knowledge**: Naturally incorporates domain expertise
    - **Regularization**: Priors act as natural regularization, preventing overfitting
    - **Optimality**: Bayes optimal classifier is theoretically the best possible
    - **Small Data**: Particularly powerful when data is limited

Bayesian methods are particularly powerful when:
- We have prior knowledge about the problem
- We need to quantify uncertainty
- We want to avoid overfitting naturally
- We need to combine multiple sources of information

## Core Theory & Intuitive Explanation

### Bayesian Philosophy

**Intuition**: Instead of finding "the best" hypothesis, we maintain a probability distribution over all possible hypotheses, updated as we see more data.

**Key Principle**: Start with prior beliefs, update with evidence (data), arrive at posterior beliefs.

### Bayes' Theorem

The foundation of Bayesian learning:

$$P(h|D) = \frac{P(D|h) \cdot P(h)}{P(D)}$$

where:
- $P(h|D)$: **Posterior** - probability of hypothesis $h$ given data $D$
- $P(D|h)$: **Likelihood** - probability of data given hypothesis
- $P(h)$: **Prior** - our belief about $h$ before seeing data
- $P(D)$: **Evidence** - normalizing constant

**Intuition**: "How likely is hypothesis $h$ given the data we observed?"

## Mathematical Foundations

### Bayesian Learning Framework

Given:
- Hypothesis space: $\mathcal{H} = \{h_1, h_2, ..., h_n\}$
- Prior: $P(h_i)$ for each hypothesis
- Training data: $D = \{d_1, d_2, ..., d_m\}$

**Goal**: Find the most probable hypothesis given the data.

### Maximum A Posteriori (MAP) Hypothesis

The MAP hypothesis is:

$$h_{MAP} = \arg\max_{h \in \mathcal{H}} P(h|D) = \arg\max_{h \in \mathcal{H}} P(D|h) \cdot P(h)$$

Since $P(D)$ is constant, we can ignore it.

**Log form** (often more convenient):

$$h_{MAP} = \arg\max_{h \in \mathcal{H}} \log P(D|h) + \log P(h)$$

### Maximum Likelihood (ML) Hypothesis

If we assume uniform prior $P(h) = \text{constant}$, MAP reduces to ML:

$$h_{ML} = \arg\max_{h \in \mathcal{H}} P(D|h)$$

### Bayes Optimal Classifier

Instead of selecting a single hypothesis, use all hypotheses weighted by their posterior:

For classification, predict the most probable class:

$$\text{arg}\max_{v_j \in V} \sum_{h_i \in \mathcal{H}} P(v_j|h_i) \cdot P(h_i|D)$$

where $V$ is the set of possible class values.

**Key Insight**: This is the optimal classifier - no other classifier can achieve lower expected error.

!!! important "Bayes Optimal Classifier"
    The Bayes optimal classifier is **theoretically optimal** - no other classifier can achieve lower expected error. However:
    - **Computationally expensive**: Requires summing over all hypotheses
    - **Requires true posteriors**: Need accurate $P(h_i|D)$ for all hypotheses
    - **Often impractical**: For large hypothesis spaces, we use approximations (MAP, etc.)
    
    This is a theoretical gold standard, not always practical, but guides algorithm design.

### Minimum Description Length (MDL) Principle

MDL connects learning to information theory:

$$h_{MDL} = \arg\min_{h \in \mathcal{H}} L_C(h) + L_C(D|h)$$

where:
- $L_C(h)$: Description length of hypothesis $h$
- $L_C(D|h)$: Description length of data given hypothesis $h$

**Intuition**: Prefer hypotheses that allow compact description of both the hypothesis and the data.

**Connection to MAP**: Under certain coding schemes, MDL is equivalent to MAP with specific priors.

### Naive Bayes Classifier

For classification with features $x = (x_1, x_2, ..., x_n)$, predict:

$$v_{MAP} = \arg\max_{v_j \in V} P(v_j) \prod_{i=1}^n P(x_i|v_j)$$

**Naive Assumption**: Features are conditionally independent given the class:

$$P(x_1, x_2, ..., x_n|v_j) = \prod_{i=1}^n P(x_i|v_j)$$

### Parameter Estimation

For discrete features, estimate probabilities using maximum likelihood:

$$P(x_i|v_j) = \frac{\text{count}(x_i, v_j)}{\text{count}(v_j)}$$

With Laplace smoothing (to handle unseen values):

$$P(x_i|v_j) = \frac{\text{count}(x_i, v_j) + 1}{\text{count}(v_j) + |\text{values}(x_i)|}$$

## Visual/Graphical Illustrations

### Bayesian Update Process

```
Prior P(h)          Data D          Posterior P(h|D)
    │                  │                  │
    │                  │                  │
┌───▼───┐          ┌───▼───┐          ┌───▼───┐
│ h1:0.3│          │ Likeli│          │ h1:0.1│
│ h2:0.5│    ×     │ hood  │    =     │ h2:0.7│
│ h3:0.2│          │ P(D|h)│          │ h3:0.2│
└───────┘          └───────┘          └───────┘
```

### MAP vs ML

```
Hypothesis Space
    │
    │  Prior P(h)
    │    ╱│╲
    │   ╱ │ ╲
    │  ╱  │  ╲
    │ ╱   │   ╲
    │╱    │    ╲
    └─────┼─────┘
          │
          │ Likelihood P(D|h)
          │        │
          │        │
          ▼        ▼
        ML      MAP
      (peak)   (weighted)
```

## Worked Examples

### Example 1: Coin Tossing

**Problem**: Estimate probability of heads for a coin.

**Prior**: $P(\theta) = \text{Beta}(\alpha, \beta)$ (conjugate prior)

**Data**: $D = \{H, H, T, H, T, H\}$ (4 heads, 2 tails)

**Likelihood**: $P(D|\theta) = \theta^4(1-\theta)^2$

**Posterior**: $P(\theta|D) \propto \theta^4(1-\theta)^2 \cdot \theta^{\alpha-1}(1-\theta)^{\beta-1}$

With uniform prior ($\alpha = \beta = 1$):

$$P(\theta|D) = \text{Beta}(5, 3)$$

**MAP estimate**: $\theta_{MAP} = \frac{5-1}{5+3-2} = \frac{4}{6} = \frac{2}{3}$

### Example 2: Text Classification

**Problem**: Classify emails as spam or not spam.

**Features**: Word presence (binary)

**Naive Bayes**:
- $P(\text{spam}) = 0.3$
- $P(\text{"free"}|\text{spam}) = 0.8$
- $P(\text{"free"}|\text{not spam}) = 0.1$

For email containing "free":

$$P(\text{spam}|\text{"free"}) = \frac{0.8 \times 0.3}{0.8 \times 0.3 + 0.1 \times 0.7} = \frac{0.24}{0.31} \approx 0.77$$

## Code Implementation

### Naive Bayes from Scratch

```python
import numpy as np
from collections import defaultdict
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class NaiveBayes:
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.class_probs = {}
        self.feature_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.classes = None
        self.n_features = None
    
    def fit(self, X, y):
        """Train Naive Bayes classifier."""
        self.classes = np.unique(y)
        self.n_features = X.shape[1]
        n_samples = len(y)
        
        # Estimate class priors P(v_j)
        for c in self.classes:
            count = np.sum(y == c)
            self.class_probs[c] = (count + self.smoothing) / (n_samples + self.smoothing * len(self.classes))
        
        # Estimate conditional probabilities P(x_i|v_j)
        for c in self.classes:
            class_mask = (y == c)
            class_X = X[class_mask]
            n_class = len(class_X)
            
            for feature_idx in range(self.n_features):
                feature_values = np.unique(X[:, feature_idx])
                n_values = len(feature_values)
                
                for value in feature_values:
                    count = np.sum(class_X[:, feature_idx] == value)
                    # Laplace smoothing
                    prob = (count + self.smoothing) / (n_class + self.smoothing * n_values)
                    self.feature_probs[c][feature_idx][value] = prob
    
    def predict(self, X):
        """Predict class labels."""
        predictions = []
        
        for x in X:
            best_class = None
            best_score = float('-inf')
            
            for c in self.classes:
                # Log probability to avoid underflow
                log_prob = np.log(self.class_probs[c])
                
                for feature_idx, value in enumerate(x):
                    if value in self.feature_probs[c][feature_idx]:
                        log_prob += np.log(self.feature_probs[c][feature_idx][value])
                    else:
                        # Unseen value - use small probability
                        log_prob += np.log(self.smoothing / (len(self.classes) * self.smoothing))
                
                if log_prob > best_score:
                    best_score = log_prob
                    best_class = c
            
            predictions.append(best_class)
        
        return np.array(predictions)

# Test on synthetic data
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                          n_classes=2, random_state=42)

# Discretize features for Naive Bayes
X_discrete = (X > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X_discrete, y, test_size=0.2, random_state=42
)

# Train and evaluate
nb = NaiveBayes(smoothing=1.0)
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

print(f"Naive Bayes Accuracy: {accuracy_score(y_test, y_pred):.3f}")
```

### MAP Estimation for Linear Regression

```python
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

class BayesianLinearRegression:
    def __init__(self, alpha=1.0, beta=1.0):
        """
        alpha: precision of prior (1/variance of weights)
        beta: precision of noise (1/variance of observations)
        """
        self.alpha = alpha
        self.beta = beta
        self.mean = None
        self.cov = None
    
    def fit(self, X, y):
        """Bayesian linear regression with Gaussian prior."""
        n_features = X.shape[1]
        
        # Prior: w ~ N(0, alpha^-1 * I)
        # Posterior: w|D ~ N(m_N, S_N)
        
        # Compute posterior parameters
        S_N_inv = self.alpha * np.eye(n_features) + self.beta * X.T @ X
        self.cov = np.linalg.inv(S_N_inv)
        self.mean = self.beta * self.cov @ X.T @ y
        
        return self
    
    def predict(self, X, return_std=False):
        """Predict with uncertainty quantification."""
        y_pred = X @ self.mean
        
        if return_std:
            # Predictive variance
            y_var = 1/self.beta + np.diag(X @ self.cov @ X.T)
            y_std = np.sqrt(y_var)
            return y_pred, y_std
        
        return y_pred

# Generate data
np.random.seed(42)
X = np.random.randn(20, 1)
y = 2 * X.flatten() + 1 + np.random.randn(20) * 0.5

# Fit Bayesian model
blr = BayesianLinearRegression(alpha=1.0, beta=2.0)
blr.fit(X, y)

# Predict with uncertainty
X_test = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_pred, y_std = blr.predict(X_test, return_std=True)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='black', label='Data', zorder=3)
plt.plot(X_test, y_pred, 'b-', label='Mean prediction', linewidth=2)
plt.fill_between(X_test.flatten(), 
                 y_pred - 2*y_std, 
                 y_pred + 2*y_std,
                 alpha=0.3, color='blue', label='95% confidence')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Bayesian Linear Regression with Uncertainty')
plt.grid(True, alpha=0.3)
plt.show()
```

## Conceptual Summary / Diagram

### Bayesian Learning Process

```
┌─────────────┐
│ Prior P(h)  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Data D     │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ Bayes' Theorem  │
│ P(h|D) ∝        │
│ P(D|h) · P(h)   │
└──────┬──────────┘
       │
       ▼
┌─────────────┐
│ Posterior   │
│ P(h|D)      │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Prediction  │
│ (MAP/Mean)  │
└─────────────┘
```

### MAP vs ML vs Bayes Optimal

| Method | Formula | Characteristics |
|--------|---------|-----------------|
| **ML** | $\arg\max P(D\|h)$ | Single best hypothesis, ignores prior |
| **MAP** | $\arg\max P(D\|h)P(h)$ | Single best, incorporates prior |
| **Bayes Optimal** | $\sum_h P(v\|h)P(h\|D)$ | Uses all hypotheses, optimal |

## Common Misconceptions / Pitfalls

### Misconception 1: "Bayesian Methods Are Always Better"

**Reality**: Bayesian methods have advantages (uncertainty, priors) but also:
- Computational complexity
- Need for good priors
- May be overkill for simple problems

!!! warning "When NOT to Use Bayesian Methods"
    Bayesian methods are not always the best choice:
    - **Large datasets**: Frequentist methods may be sufficient and faster
    - **No prior knowledge**: If you have no domain expertise, uniform priors = ML
    - **Computational constraints**: Bayesian inference can be expensive
    - **Simple problems**: Overkill for straightforward classification/regression
    
    **Rule of thumb**: Use Bayesian methods when you have prior knowledge or need uncertainty estimates.

### Misconception 2: "Prior Doesn't Matter with Enough Data"

**Reality**: While data dominates with large samples, priors are crucial when:
- Data is limited
- We have strong domain knowledge
- We want to regularize

### Misconception 3: "Naive Bayes Assumption Is Too Strong"

**Reality**: Despite independence assumption, Naive Bayes often works well because:
- It's the class-conditional independence that matters
- Classification only needs correct class ordering
- Robust to violations in practice

### Pitfall: Improper Priors

Using improper priors (non-integrable) can lead to improper posteriors. Always verify priors are proper.

### Pitfall: Ignoring the Evidence Term

When comparing hypotheses, $P(D)$ cancels out. But for model comparison, it's crucial (Bayes factor).

## Practice Exercises

### Exercise 1: MAP Derivation

Derive the MAP estimate for a Gaussian likelihood $P(D|\mu) = \mathcal{N}(\mu, \sigma^2)$ with Gaussian prior $P(\mu) = \mathcal{N}(\mu_0, \tau^2)$.

Show that the posterior is also Gaussian and find its parameters.

### Exercise 2: Bayes Optimal Classifier

Given:
- $\mathcal{H} = \{h_1, h_2, h_3\}$
- $P(h_1|D) = 0.3$, $P(h_2|D) = 0.5$, $P(h_3|D) = 0.2$
- $P(+|h_1) = 0.9$, $P(+|h_2) = 0.6$, $P(+|h_3) = 0.3$

What is the Bayes optimal prediction for class $+$?

### Exercise 3: Naive Bayes Implementation

Implement Naive Bayes for continuous features using Gaussian distributions:
$$P(x_i|v_j) = \mathcal{N}(\mu_{ij}, \sigma_{ij}^2)$$

### Exercise 4: MDL Analysis

For a hypothesis space with hypotheses of different complexities, show how MDL naturally prefers simpler hypotheses that still explain the data well.

## Summary & Key Takeaways

### Key Concepts

1. **Bayes' Theorem**: Foundation for updating beliefs with evidence
2. **MAP Estimation**: Balances likelihood and prior
3. **Bayes Optimal Classifier**: Theoretically optimal, uses all hypotheses
4. **MDL Principle**: Connects learning to information theory
5. **Naive Bayes**: Simple, effective classifier despite independence assumption

### Important Principles

- **Prior Knowledge**: Bayesian methods naturally incorporate domain knowledge
- **Uncertainty Quantification**: Maintains probability distributions, not just point estimates
- **Occam's Razor**: Simpler hypotheses preferred (via priors or MDL)
- **Optimality**: Bayes optimal classifier is theoretically best possible

### Next Steps

Understanding Bayesian learning prepares you for:
- Regularization in linear models (Bayesian interpretation)
- Probabilistic graphical models
- Gaussian processes

## References / Further Reading

### Primary References

1. **Mitchell, T. M.** (1997). *Machine Learning*.
   - Chapter 6: Bayesian Learning

2. **Bishop, C. M.** (2006). *Pattern Recognition & Machine Learning*.
   - Chapter 1: Introduction (Bayesian perspective)
   - Chapter 2: Probability Distributions
   - Chapter 3: Linear Models for Regression (Bayesian)

### Research Papers

3. **Rissanen, J.** (1978). Modeling by shortest data description. *Automatica*, 14(5), 465-471. (MDL Principle)
   - [DOI: 10.1016/0005-1098(78)90005-5](https://doi.org/10.1016/0005-1098(78)90005-5)
   - [PDF](https://www.sciencedirect.com/science/article/abs/pii/0005109878900055)

4. **Domingos, P., & Pazzani, M.** (1997). On the optimality of the simple Bayesian classifier under zero-one loss. *Machine Learning*, 29(2-3), 103-130.
   - [DOI: 10.1023/A:1007413511361](https://doi.org/10.1023/A:1007413511361)
   - [PDF](https://link.springer.com/article/10.1023/A:1007413511361)

---

## Recommended Reads

<details>
<summary><strong>📚 Official Documentation</strong></summary>

- **Scikit-learn Naive Bayes** - [Naive Bayes classifiers](https://scikit-learn.org/stable/modules/naive_bayes.html)
- **PyMC3 Documentation** - [Probabilistic programming](https://www.pymc.io/projects/docs/en/stable/)
- **Stan Documentation** - [Bayesian inference](https://mc-stan.org/users/documentation/)

</details>

<details>
<summary><strong>📖 Essential Articles</strong></summary>

- **Bayesian Inference Tutorial** - [Bayesian methods guide](https://towardsdatascience.com/bayesian-inference-introduction-90c9e2c3aaeb)
- **Naive Bayes Explained** - [Naive Bayes tutorial](https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/)
- **MAP vs ML** - [Understanding MAP estimation](https://towardsdatascience.com/maximum-likelihood-estimation-vs-maximum-a-posteriori-estimation-b3af2bdf9a5)

</details>

<details>
<summary><strong>🎓 Learning Resources</strong></summary>

- **Bayesian Methods** - [Bayesian statistics course](https://www.coursera.org/learn/bayesian-statistics)
- **Probabilistic Programming** - [PyMC tutorials](https://www.pymc.io/projects/examples/en/latest/)
- **Bayesian ML** - [Bayesian machine learning guide](https://www.cs.ubc.ca/~murphyk/MLbook/)

</details>

<details>
<summary><strong>💡 Best Practices</strong></summary>

- **Prior Selection** - [Choosing priors](https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/prior_choice_guidelines.html)
- **Bayesian Model Comparison** - [Model comparison techniques](https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/model_comparison.html)
- **MCMC Diagnostics** - [MCMC convergence diagnostics](https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/Diagnosing_biased_Inference_with_Divergences.html)

</details>

<details>
<summary><strong>🔬 Research Papers</strong></summary>

- **MDL Principle** - [Rissanen (1978)](https://doi.org/10.1016/0005-1098(78)90005-5) - Minimum description length
- **Naive Bayes Optimality** - [Domingos & Pazzani (1997)](https://doi.org/10.1023/A:1007413511361) - Optimality of Naive Bayes

</details>

---

**Previous Chapter**: [Chapter 3: Model Selection](03_model_selection.md) | **Next Chapter**: [Chapter 5: Linear Models](05_linear_models.md)

