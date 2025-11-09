# Chapter 5: Linear Models

## Overview & Motivation

Linear models form the foundation of many machine learning algorithms. Despite their simplicity, they are powerful, interpretable, and serve as building blocks for more complex methods. This chapter covers linear models for both regression and classification, including regularization techniques (Ridge, Lasso) and their mathematical foundations.

!!! important "Why Start with Linear Models?"
    Linear models are the perfect starting point because:
    - **Simplicity**: Easy to understand and implement
    - **Interpretability**: You can see exactly what each feature contributes
    - **Baseline**: Always try linear models first before complex ones
    - **Foundation**: Neural networks and SVMs build on linear concepts
    - **Production**: Often the best choice for real-world systems (simplicity + performance)

Linear models are particularly valuable because:
- They are computationally efficient
- They provide interpretable results
- They have strong theoretical foundations
- They serve as baselines for comparison
- They extend naturally to non-linear problems via feature engineering

## Core Theory & Intuitive Explanation

### Linear Regression

**Intuition**: Find a line (or hyperplane) that best fits the data by minimizing the distance between predictions and actual values.

**Goal**: Learn a function $h(x) = w^T x + b$ that predicts $y$ from features $x$.

### Linear Classification

**Intuition**: Find a line (or hyperplane) that separates different classes.

**Goal**: Learn a decision boundary $w^T x + b = 0$ that classifies examples.

### Regularization

**Intuition**: Prevent the model from becoming too complex by penalizing large weights. This helps generalization.

**Types**:
- **L2 (Ridge)**: Penalizes sum of squared weights
- **L1 (Lasso)**: Penalizes sum of absolute weights (can zero out features)

## Mathematical Foundations

### Linear Regression

Given training data $\{(x_i, y_i)\}_{i=1}^m$ where $x_i \in \mathbb{R}^d$, learn:

$$h(x) = w^T x + b = \sum_{j=1}^d w_j x_j + b$$

**Objective**: Minimize squared error:

$$L(w, b) = \frac{1}{2m} \sum_{i=1}^m (h(x_i) - y_i)^2 = \frac{1}{2m} \sum_{i=1}^m (w^T x_i + b - y_i)^2$$

**Closed-form solution** (Normal Equation):

In matrix form, with $X \in \mathbb{R}^{m \times d}$ and $y \in \mathbb{R}^m$:

$$w^* = (X^T X)^{-1} X^T y$$

!!! warning "Normal Equation Limitations"
    The normal equation requires $(X^T X)^{-1}$ to exist, which fails when:
    - **More features than samples**: $d > m$ (underdetermined system)
    - **Multicollinearity**: Features are linearly dependent
    - **Singular matrix**: $X^T X$ is not invertible
    
    **Solution**: Use gradient descent or add regularization (Ridge) to make it invertible.

**Gradient Descent**:

Update rule:
$$w_j := w_j - \alpha \frac{\partial L}{\partial w_j}$$

where:
$$\frac{\partial L}{\partial w_j} = \frac{1}{m} \sum_{i=1}^m (h(x_i) - y_i) x_{ij}$$

### Ridge Regression (L2 Regularization)

Add penalty on weights:

$$L_{ridge}(w) = \frac{1}{2m} \sum_{i=1}^m (w^T x_i + b - y_i)^2 + \frac{\lambda}{2} \|w\|_2^2$$

where $\|w\|_2^2 = \sum_{j=1}^d w_j^2$ and $\lambda > 0$ is the regularization parameter.

**Closed-form solution**:

$$w^* = (X^T X + \lambda I)^{-1} X^T y$$

**Effect**: Shrinks weights toward zero, prevents overfitting.

### Lasso Regression (L1 Regularization)

$$L_{lasso}(w) = \frac{1}{2m} \sum_{i=1}^m (w^T x_i + b - y_i)^2 + \lambda \|w\|_1$$

where $\|w\|_1 = \sum_{j=1}^d |w_j|$.

**Effect**: Can drive weights to exactly zero (feature selection).

### Logistic Regression (Linear Classification)

For binary classification, use the logistic function:

$$h(x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$$

where $\sigma(z) = \frac{1}{1+e^{-z}}$ is the sigmoid function.

**Objective**: Maximize log-likelihood (or minimize cross-entropy):

$$L(w, b) = -\frac{1}{m} \sum_{i=1}^m [y_i \log(h(x_i)) + (1-y_i) \log(1-h(x_i))]$$

**Gradient**:

$$\frac{\partial L}{\partial w_j} = \frac{1}{m} \sum_{i=1}^m (h(x_i) - y_i) x_{ij}$$

### Perceptron Algorithm

Simple linear classifier:

$$h(x) = \text{sign}(w^T x + b) = \begin{cases} +1 & \text{if } w^T x + b \geq 0 \\ -1 & \text{otherwise} \end{cases}$$

**Update rule** (if misclassified):

$$w := w + \alpha \cdot y_i \cdot x_i$$

where $\alpha$ is the learning rate.

## Visual/Graphical Illustrations

### Linear Regression

```
y
│     ●
│    ●
│   ●
│  ●
│ ●
│●
└─────────── x
    h(x) = wx + b
```

### Linear Classification

```
x2
│
│    ●  ●  ●
│   ●   ●  ●
│  ●    ●
│─────────────────── Decision Boundary
│        ○  ○  ○
│       ○   ○  ○
│      ○    ○
└─────────────────── x1
```

### Regularization Effect

```
Weights
  ↑
  │     Lasso (L1)
  │    ╱│╲
  │   ╱ │ ╲
  │  ╱  │  ╲
  │ ╱   │   ╲ Ridge (L2)
  │╱    │    ╲
  └─────┼─────┘
        │
     λ increases
```

## Worked Examples

### Example 1: Simple Linear Regression

**Data**: House prices vs. size
- Size: [50, 60, 70, 80, 90] m²
- Price: [100, 120, 140, 160, 180] (in thousands)

**Model**: $h(x) = w \cdot x + b$

**Solution**: Using normal equation or gradient descent, find $w \approx 2$, $b \approx 0$.

### Example 2: Logistic Regression for Classification

**Problem**: Classify emails as spam (1) or not spam (0).

**Features**: Word frequencies

**Model**: $P(y=1|x) = \sigma(w^T x + b)$

**Decision**: Predict spam if $P(y=1|x) > 0.5$, i.e., $w^T x + b > 0$.

## Code Implementation

### Linear Regression from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        """Train using gradient descent."""
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        
        for _ in range(self.n_iterations):
            # Predictions
            y_pred = X @ self.weights + self.bias
            
            # Gradients
            dw = (1/m) * X.T @ (y_pred - y)
            db = (1/m) * np.sum(y_pred - y)
            
            # Update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        return X @ self.weights + self.bias

# Generate data
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
lr = LinearRegression(learning_rate=0.01, n_iterations=1000)
lr.fit(X_train, y_train)

# Evaluate
y_pred = lr.predict(X_test)
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.3f}")

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, alpha=0.6, label='Training data')
plt.scatter(X_test, y_test, alpha=0.6, label='Test data')
plt.plot(X_test, y_pred, 'r-', linewidth=2, label='Prediction')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression')
plt.grid(True, alpha=0.3)
plt.show()
```

### Ridge and Lasso Regression

```python
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Generate data with many features
X, y = make_regression(n_samples=50, n_features=20, noise=10, random_state=42)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Test different lambda values
alphas = np.logspace(-4, 2, 50)
ridge_coefs = []
lasso_coefs = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    lasso = Lasso(alpha=alpha)
    ridge.fit(X_scaled, y)
    lasso.fit(X_scaled, y)
    ridge_coefs.append(ridge.coef_)
    lasso_coefs.append(lasso.coef_)

ridge_coefs = np.array(ridge_coefs)
lasso_coefs = np.array(lasso_coefs)

# Plot regularization paths
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
for i in range(min(10, ridge_coefs.shape[1])):
    plt.plot(alphas, ridge_coefs[:, i])
plt.xscale('log')
plt.xlabel('Lambda (α)')
plt.ylabel('Coefficient Value')
plt.title('Ridge Regression Regularization Path')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
for i in range(min(10, lasso_coefs.shape[1])):
    plt.plot(alphas, lasso_coefs[:, i])
plt.xscale('log')
plt.xlabel('Lambda (α)')
plt.ylabel('Coefficient Value')
plt.title('Lasso Regression Regularization Path')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Count features selected by Lasso
lasso_final = Lasso(alpha=1.0)
lasso_final.fit(X_scaled, y)
n_selected = np.sum(np.abs(lasso_final.coef_) > 1e-5)
print(f"Lasso selected {n_selected} out of {X.shape[1]} features")
```

### Logistic Regression

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Generate classification data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                          n_classes=2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train logistic regression
lr = LogisticRegression(max_iter=1000, C=1.0)
lr.fit(X_train, y_train)

# Predict
y_pred = lr.predict(X_test)
y_pred_proba = lr.predict_proba(X_test)[:, 1]

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot decision boundary (for 2D case)
if X.shape[1] == 2:
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', alpha=0.6)
    
    # Create mesh for decision boundary
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = lr.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Logistic Regression Decision Boundary')
    plt.colorbar(scatter)
    plt.show()
```

## Conceptual Summary / Diagram

### Linear Model Family

```
Linear Models
    │
    ├── Regression
    │   ├── Linear Regression
    │   ├── Ridge (L2)
    │   └── Lasso (L1)
    │
    └── Classification
        ├── Logistic Regression
        ├── Perceptron
        └── Linear SVM
```

### Regularization Comparison

| Method | Penalty | Effect | Use Case |
|--------|---------|--------|----------|
| **None** | - | No regularization | Simple problems, large data |
| **Ridge** | $\lambda \|w\|_2^2$ | Shrinks weights | Multicollinearity, many features |
| **Lasso** | $\lambda \|w\|_1$ | Feature selection | Sparse solutions, interpretability |

## Common Misconceptions / Pitfalls

### Misconception 1: "Linear Models Are Too Simple"

**Reality**: Linear models can be very powerful with:
- Feature engineering (polynomial, interactions)
- Kernel methods (implicit non-linear mapping)
- Ensemble methods

### Misconception 2: "Lasso Always Selects Features"

**Reality**: Lasso selects features only if $\lambda$ is large enough. Too small $\lambda$ gives dense solutions.

### Misconception 3: "Ridge and Lasso Are Interchangeable"

**Reality**: They have different effects:
- Ridge: Smooth shrinkage, all features retained
- Lasso: Sparse solutions, feature selection
- Elastic Net: Combines both

### Pitfall: Not Scaling Features

Gradient descent and regularization are sensitive to feature scales. Always standardize features before applying Ridge/Lasso.

!!! danger "Feature Scaling Critical"
    **Without scaling**: Features with larger scales dominate the model
    ```python
    # BAD: Features on different scales
    X = [[1000, 0.5], [2000, 0.3], ...]  # Size vs Price ratio
    # Size feature dominates!
    ```
    
    **With scaling**: All features contribute equally
    ```python
    # GOOD: Standardized features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Now all features on same scale
    ```
    
    **Always scale** before Ridge/Lasso - regularization assumes features are comparable!

### Pitfall: Ignoring Multicollinearity

Highly correlated features can cause instability in linear regression. Ridge regression helps, but feature selection may be needed.

## Practice Exercises

### Exercise 1: Normal Equation Derivation

Derive the normal equation $w^* = (X^T X)^{-1} X^T y$ by setting the gradient of the squared error to zero.

### Exercise 2: Ridge Regression Derivation

Show that the Ridge regression solution $w^* = (X^T X + \lambda I)^{-1} X^T y$ can be derived by adding the regularization term to the normal equation.

### Exercise 3: Logistic Regression Gradient

Derive the gradient of the cross-entropy loss for logistic regression:
$$\frac{\partial L}{\partial w_j} = \frac{1}{m} \sum_{i=1}^m (h(x_i) - y_i) x_{ij}$$

### Exercise 4: Implementation

Implement Ridge regression from scratch using both:
1. Closed-form solution
2. Gradient descent

Compare the results and computational efficiency.

## Summary & Key Takeaways

### Key Concepts

1. **Linear Regression**: Minimize squared error to fit a line/hyperplane
2. **Regularization**: Prevent overfitting by penalizing complexity
3. **Ridge (L2)**: Shrinks weights, handles multicollinearity
4. **Lasso (L1)**: Feature selection, sparse solutions
5. **Logistic Regression**: Linear classifier using sigmoid function

### Important Principles

- **Feature Scaling**: Critical for gradient descent and regularization
- **Bias-Variance Trade-off**: Regularization controls this balance
- **Interpretability**: Linear models provide clear feature importance
- **Extensibility**: Foundation for kernel methods and neural networks

### Next Steps

Understanding linear models prepares you for:
- Non-linear extensions (kernel methods, Chapter 8)
- Tree-based methods (Chapter 6)
- Neural networks (linear layers, Chapter 9)

## References / Further Reading

??? note "📚 Primary References"

    1. **Bishop, C. M.** (2006). *Pattern Recognition & Machine Learning*.
       - Chapter 3: Linear Models for Regression
       - Chapter 4: Linear Models for Classification

    2. **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning*.
       - Chapter 3: Linear Methods for Regression
       - Chapter 4: Linear Methods for Classification

??? note "🔬 Research Papers"

    3. **Tibshirani, R.** (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society*, 58(1), 267-288.
       - [DOI: 10.1111/j.2517-6161.1996.tb02080.x](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x)
       - [PDF](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.2517-6161.1996.tb02080.x)

    4. **Hoerl, A. E., & Kennard, R. W.** (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55-67.
       - [DOI: 10.1080/00401706.1970.10488634](https://doi.org/10.1080/00401706.1970.10488634)
       - [PDF](https://www.tandfonline.com/doi/abs/10.1080/00401706.1970.10488634)

    ---

## Recommended Reads

??? note "📚 Official Documentation"

    - **Linear Models** - [Scikit-learn linear models](https://scikit-learn.org/stable/modules/linear_model.html)

    - **Logistic Regression** - [Logistic regression guide](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)

    - **Regularization** - [Ridge and Lasso documentation](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification)

??? note "📖 Essential Articles"

    - **Linear Regression Tutorial** - [Complete linear regression guide](https://machinelearningmastery.com/linear-regression-for-machine-learning/)

    - **Ridge vs Lasso** - [Understanding regularization](https://towardsdatascience.com/ridge-and-lasso-regression-a-complete-guide-with-python-scikit-learn-e20e34bcbf0b)

    - **Logistic Regression Explained** - [Logistic regression tutorial](https://machinelearningmastery.com/logistic-regression-for-machine-learning/)

??? note "🎓 Learning Resources"

    - **Gradient Descent** - [Understanding gradient descent](https://towardsdatascience.com/gradient-descent-algorithm-and-its-variants-10f652806a3)

    - **Feature Scaling** - [Why and how to scale features](https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling)

    - **Multicollinearity** - [Handling multicollinearity](https://towardsdatascience.com/multicollinearity-in-regression-fe7a2b4d4d0)

??? note "💡 Best Practices"

    - **Feature Scaling** - [Scaling best practices](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler)

    - **Regularization Tuning** - [Hyperparameter tuning for regularization](https://scikit-learn.org/stable/modules/grid_search.html)

    - **Linear Model Diagnostics** - [Model diagnostics](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)

??? note "🔬 Research Papers"

    - **Lasso Regression** - [Tibshirani (1996)](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x) - Regression shrinkage and selection via lasso

    - **Ridge Regression** - [Hoerl & Kennard (1970)](https://doi.org/10.1080/00401706.1970.10488634) - Biased estimation for nonorthogonal problems

    ---

    **Previous Chapter**: [Chapter 4: Bayesian Learning](04_bayesian_learning.md) | **Next Chapter**: [Chapter 6: Decision Trees](06_decision_trees.md)

