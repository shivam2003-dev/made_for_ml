# Chapter 3: Model Selection and Evaluation

## Overview & Motivation

Selecting the right model and evaluating its performance are fundamental challenges in machine learning. This chapter covers the bias-variance decomposition, cross-validation, evaluation metrics, and the crucial trade-offs involved in model selection. Understanding these concepts is essential for building models that generalize well to unseen data.

!!! important "Why Model Selection Matters"
    Model selection is arguably the most critical skill in machine learning:
    - **Prevents overfitting**: Choose models that generalize, not just memorize
    - **Saves resources**: Avoid training overly complex models unnecessarily
    - **Enables comparison**: Fairly compare different algorithms
    - **Builds intuition**: Understand when and why models fail
    
    A poorly selected model will fail in production, no matter how sophisticated the algorithm.

The central problem: How do we choose among different models, and how do we know if our model will perform well on new data?

## Core Theory & Intuitive Explanation

### The Bias-Variance Trade-off

**Intuition**: Every model makes errors. These errors come from two sources:
- **Bias**: How far off our model's assumptions are from reality (underfitting)
- **Variance**: How much our model's predictions vary with different training sets (overfitting)

A good model balances both sources of error.

!!! tip "Understanding Bias and Variance"
    Think of bias and variance like shooting at a target:
    - **High Bias, Low Variance**: Always miss in the same direction (consistent but wrong)
      - Example: Linear model trying to fit non-linear data
    - **Low Bias, High Variance**: Hits different spots each time (unpredictable)
      - Example: Complex model that memorizes training data
    - **Low Bias, Low Variance**: Hits the bullseye consistently (ideal)
      - Example: Well-tuned model that captures true patterns
    
    The goal is to find the sweet spot between these extremes.

### Model Selection

**Intuition**: We want a model that:
1. Fits the training data well (low training error)
2. Generalizes to new data (low test error)

These goals often conflict - a model that fits training data too well may not generalize.

## Mathematical Foundations

### Bias-Variance Decomposition

For regression with squared loss, the expected prediction error can be decomposed as:

$$E[(y - \hat{f}(x))^2] = \text{Bias}^2(\hat{f}(x)) + \text{Var}(\hat{f}(x)) + \sigma^2$$

where:

- **Bias**: $\text{Bias}(\hat{f}(x)) = E[\hat{f}(x)] - f(x)$
  - Measures how much the average prediction differs from the true value

- **Variance**: $\text{Var}(\hat{f}(x)) = E[(\hat{f}(x) - E[\hat{f}(x)])^2]$
  - Measures how much predictions vary across different training sets

- **Irreducible Error**: $\sigma^2 = E[(y - f(x))^2]$
  - Noise in the data that cannot be reduced
  - Represents the fundamental uncertainty in the problem

!!! important "The Irreducible Error"
    No matter how good your model is, you can never reduce error below $\sigma^2$. This represents:
    - Measurement noise in your data
    - Unobservable factors affecting the outcome
    - Fundamental randomness in the process
    
    **Key Insight**: If your model's error is close to $\sigma^2$, you've done about as well as possible!

### Derivation

Let $y = f(x) + \epsilon$ where $\epsilon \sim \mathcal{N}(0, \sigma^2)$.

The expected squared error at point $x$ is:

$$E[(y - \hat{f}(x))^2] = E[(f(x) + \epsilon - \hat{f}(x))^2]$$

Expanding:

$$= E[(f(x) - \hat{f}(x))^2] + E[\epsilon^2] + 2E[\epsilon(f(x) - \hat{f}(x))]$$

Since $\epsilon$ is independent and $E[\epsilon] = 0$:

$$= E[(f(x) - \hat{f}(x))^2] + \sigma^2$$

Now, let $\bar{f}(x) = E[\hat{f}(x)]$. Then:

$$E[(f(x) - \hat{f}(x))^2] = E[(f(x) - \bar{f}(x) + \bar{f}(x) - \hat{f}(x))^2]$$

$$= (f(x) - \bar{f}(x))^2 + E[(\bar{f}(x) - \hat{f}(x))^2]$$

$$= \text{Bias}^2 + \text{Variance}$$

### Cross-Validation

**K-Fold Cross-Validation**:

1. Split data into $k$ folds
2. For each fold $i$:
   - Train on all folds except $i$
   - Evaluate on fold $i$
3. Average the $k$ evaluation scores

The cross-validation error is:

$$CV_{(k)} = \frac{1}{k} \sum_{i=1}^k L(\hat{f}^{-i}, D_i)$$

where $\hat{f}^{-i}$ is the model trained on all data except fold $i$, and $D_i$ is fold $i$.

!!! note "Choosing k in K-Fold CV"
    Common choices for $k$:
    - **k=5 or k=10**: Standard choice, good balance
    - **k=m (Leave-One-Out)**: Maximum data usage, but computationally expensive
    - **k=3**: Faster but less reliable estimates
    
    **Rule of thumb**: Use $k=5$ or $k=10$ unless you have very small datasets (< 100 samples).

### Evaluation Metrics

**Classification Metrics**:

- **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
- **Precision**: $\frac{TP}{TP + FP}$
- **Recall**: $\frac{TP}{TP + FN}$
- **F1-Score**: $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$

**Regression Metrics**:

- **MSE**: $\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$
- **RMSE**: $\sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}$
- **MAE**: $\frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$
- **R²**: $1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$

## Visual/Graphical Illustrations

### Bias-Variance Trade-off Visualization

```
Error
  ↑
  │     ┌─────────────┐
  │    ╱             ╲
  │   ╱               ╲
  │  ╱                 ╲ Total Error
  │ ╱                   ╲
  │╱                     ╲
  │───────────────────────→ Model Complexity
  │        ╱│╲
  │       ╱ │ ╲ Variance
  │      ╱  │  ╲
  │     ╱   │   ╲
  │    ╱    │    ╲
  │   ╱     │     ╲
  │  ╱      │      ╲
  │ ╱       │       ╲
  │╱        │        ╲
  └─────────┼─────────┘
            │
         Bias²
```

### K-Fold Cross-Validation

```
Data: [████████████████████]

Fold 1: [Test][Train][Train][Train][Train]
Fold 2: [Train][Test][Train][Train][Train]
Fold 3: [Train][Train][Test][Train][Train]
Fold 4: [Train][Train][Train][Test][Train]
Fold 5: [Train][Train][Train][Train][Test]
```

## Worked Examples

### Example 1: Polynomial Regression and Overfitting

**Problem**: Fit a curve to noisy data points.

**Data**: $y = \sin(2\pi x) + \epsilon$ where $\epsilon \sim \mathcal{N}(0, 0.1)$

**Models**: Polynomials of degree 1, 3, 9

**Observation**: 
- Degree 1: High bias, low variance (underfitting)
- Degree 3: Balanced (good fit)
- Degree 9: Low bias, high variance (overfitting)

### Example 2: Model Selection via Cross-Validation

**Problem**: Choose regularization parameter $\lambda$ for Ridge Regression.

**Approach**:
1. Try $\lambda \in \{0.001, 0.01, 0.1, 1, 10, 100\}$
2. For each $\lambda$, compute 5-fold CV error
3. Select $\lambda$ with minimum CV error

## Code Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

# Generate synthetic data
np.random.seed(42)
X = np.linspace(0, 1, 30).reshape(-1, 1)
y_true = np.sin(2 * np.pi * X.flatten())
y = y_true + np.random.normal(0, 0.1, size=X.shape[0])

# Test different polynomial degrees
degrees = [1, 3, 9]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, degree in enumerate(degrees):
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    
    # Fit model
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Predict
    X_test = np.linspace(0, 1, 100).reshape(-1, 1)
    X_test_poly = poly.transform(X_test)
    y_pred = model.predict(X_test_poly)
    
    # Plot
    axes[idx].scatter(X.flatten(), y, alpha=0.6, label='Data')
    axes[idx].plot(X_test.flatten(), np.sin(2 * np.pi * X_test.flatten()), 
                   'g-', label='True function', linewidth=2)
    axes[idx].plot(X_test.flatten(), y_pred, 'r-', label='Fitted', linewidth=2)
    axes[idx].set_title(f'Degree {degree}')
    axes[idx].legend()
    axes[idx].set_xlabel('x')
    axes[idx].set_ylabel('y')

plt.tight_layout()
plt.show()

# Cross-validation for model selection
def evaluate_model_degree(X, y, degree, cv_folds=5):
    """Evaluate polynomial model using cross-validation."""
    poly = PolynomialFeatures(degree=degree)
    model = Pipeline([
        ('poly', poly),
        ('linear', LinearRegression())
    ])
    
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kfold, 
                            scoring='neg_mean_squared_error')
    return -scores.mean(), scores.std()

# Test degrees 1-10
degrees_to_test = range(1, 11)
cv_errors = []
cv_stds = []

for degree in degrees_to_test:
    mean_error, std_error = evaluate_model_degree(X, y, degree)
    cv_errors.append(mean_error)
    cv_stds.append(std_error)

# Plot CV error vs degree
plt.figure(figsize=(10, 6))
plt.errorbar(degrees_to_test, cv_errors, yerr=cv_stds, 
             marker='o', capsize=5)
plt.xlabel('Polynomial Degree')
plt.ylabel('Cross-Validation MSE')
plt.title('Model Selection via Cross-Validation')
plt.grid(True, alpha=0.3)
plt.show()

# Find best degree
best_degree = degrees_to_test[np.argmin(cv_errors)]
print(f"Best degree: {best_degree} (CV MSE: {min(cv_errors):.4f})")
```

### Regularization Parameter Selection

```python
from sklearn.linear_model import RidgeCV
from sklearn.datasets import make_regression

# Generate data
X, y = make_regression(n_samples=100, n_features=20, 
                       n_informative=10, noise=10, random_state=42)

# Use RidgeCV for automatic lambda selection
alphas = np.logspace(-4, 2, 50)  # Lambda values to try
ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring='neg_mean_squared_error')
ridge_cv.fit(X, y)

print(f"Best alpha (lambda): {ridge_cv.alpha_:.4f}")
print(f"CV MSE: {-ridge_cv.best_score_:.2f}")

# Compare with unregularized
lr = LinearRegression()
lr.fit(X, y)
lr_mse = mean_squared_error(y, lr.predict(X))
print(f"Unregularized MSE: {lr_mse:.2f}")
```

## Conceptual Summary / Diagram

### Model Selection Process

```
┌──────────────┐
│ Training Data│
└──────┬───────┘
       │
       ▼
┌─────────────────────┐
│ Candidate Models    │
│ (different params)  │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Cross-Validation    │
│ (evaluate each)     │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Select Best Model   │
│ (lowest CV error)   │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Final Evaluation    │
│ (on test set)       │
└─────────────────────┘
```

### Bias-Variance Relationship

| Model Complexity | Bias | Variance | Total Error |
|-----------------|------|----------|-------------|
| Low (simple)    | High | Low      | High        |
| Medium          | Low  | Medium   | **Low**     |
| High (complex)  | Low  | High     | High        |

## Common Misconceptions / Pitfalls

### Misconception 1: "Lower Training Error = Better Model"

**Reality**: Lower training error can indicate overfitting. Always evaluate on held-out test data.

### Misconception 2: "Cross-Validation Eliminates Need for Test Set"

**Reality**: CV is for model selection. You still need a separate test set for final evaluation to avoid overfitting to the validation procedure.

### Misconception 3: "More Complex Models Are Always Better"

**Reality**: Complexity must match the problem. Simple models often generalize better (Occam's Razor).

### Pitfall: Data Leakage in Cross-Validation

If preprocessing (e.g., scaling) is done before splitting, information leaks from validation to training folds. Always fit preprocessing on training fold only.

!!! danger "Data Leakage Example"
    **WRONG** (data leakage):
    ```python
    # Compute mean/std on ALL data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Uses test data!
    
    # Then split
    X_train, X_test = train_test_split(X_scaled, ...)
    ```
    
    **CORRECT** (no leakage):
    ```python
    # Split first
    X_train, X_test = train_test_split(X, ...)
    
    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Use same scaling
    ```
    
    Always: **Split → Fit on train → Transform both**

### Pitfall: Using Test Set for Model Selection

Using the test set to select hyperparameters causes overfitting to the test set. Use nested CV or separate validation set.

## Practice Exercises

### Exercise 1: Bias-Variance Calculation

For a regression problem, given:
- True function: $f(x) = x^2$
- Model: $\hat{f}(x) = ax + b$
- Data: $y = x^2 + \epsilon$ where $\epsilon \sim \mathcal{N}(0, 0.1)$

Calculate the bias and variance of the linear model at $x = 1$.

### Exercise 2: Cross-Validation Implementation

Implement 5-fold cross-validation from scratch (without sklearn) for a simple linear regression model.

### Exercise 3: Model Selection

Given a dataset, compare:
1. Linear regression
2. Polynomial regression (degree 3)
3. Ridge regression
4. Lasso regression

Use cross-validation to select the best model and regularization parameter.

### Exercise 4: Analysis

A model achieves:
- Training accuracy: 99%
- Validation accuracy: 65%
- Test accuracy: 64%

Explain what's happening and propose solutions.

## Summary & Key Takeaways

### Key Concepts

1. **Bias-Variance Decomposition**: Total error = Bias² + Variance + Irreducible Error
2. **Trade-off**: Increasing model complexity decreases bias but increases variance
3. **Cross-Validation**: Robust method for model selection and hyperparameter tuning
4. **Evaluation Metrics**: Choose metrics appropriate for your problem

### Important Principles

- **Generalization over Memorization**: Prefer models that generalize
- **Validation Set**: Always reserve data for validation
- **Nested CV**: Use for unbiased hyperparameter selection
- **Multiple Metrics**: Don't rely on a single metric

### Next Steps

Understanding model selection prepares you for:
- Bayesian approaches to model selection (Chapter 4)
- Regularization in linear models (Chapter 5)
- Pruning in decision trees (Chapter 6)

## References / Further Reading

### Primary References

1. **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning*. 
   - Chapter 7: Model Assessment and Selection
   - Chapter 2: Overview of Supervised Learning (Bias-Variance)

2. **Bishop, C. M.** (2006). *Pattern Recognition & Machine Learning*.
   - Chapter 1: Introduction (Model Selection)
   - Chapter 3: Linear Models for Regression (Regularization)

### Research Papers

3. **Kohavi, R.** (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection. *IJCAI*, 14(2), 1137-1145.
   - [PDF](https://www.ijcai.org/Proceedings/95-2/Papers/016.pdf)
   - [CiteSeerX](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.48.529)

4. **Geman, S., Bienenstock, E., & Doursat, R.** (1992). Neural networks and the bias/variance dilemma. *Neural Computation*, 4(1), 1-58.
   - [DOI: 10.1162/neco.1992.4.1.1](https://doi.org/10.1162/neco.1992.4.1.1)
   - [PDF](https://direct.mit.edu/neco/article-abstract/4/1/1/5515/Neural-Networks-and-the-Bias-Variance-Dilemma)

---

**Previous Chapter**: [Chapter 2: Learning Paradigms](02_learning_paradigms.md) | **Next Chapter**: [Chapter 4: Bayesian Learning](04_bayesian_learning.md)

