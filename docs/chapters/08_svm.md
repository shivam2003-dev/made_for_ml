# Chapter 8: Support Vector Machines

## Overview & Motivation

Support Vector Machines (SVM) are powerful classifiers that find the optimal separating hyperplane by maximizing the margin between classes. This chapter covers the mathematical foundations of SVMs, including hard and soft margins, kernel methods, and the connection to VC dimension theory.

!!! important "Why SVMs?"
    SVMs have strong theoretical foundations and practical advantages:
    - **Maximum Margin**: Finds the "safest" decision boundary
    - **Kernel Trick**: Handles non-linear problems elegantly
    - **Sparse**: Only support vectors matter (memory efficient)
    - **Theoretical Guarantees**: VC dimension theory provides generalization bounds
    - **Versatile**: Works for classification, regression, and novelty detection

SVMs are particularly valuable because:
- Strong theoretical foundations (statistical learning theory)
- Effective in high-dimensional spaces
- Memory efficient (only stores support vectors)
- Versatile (can handle non-linear problems via kernels)
- Robust to overfitting (maximizing margin provides regularization)

## Core Theory & Intuitive Explanation

### Maximum Margin Intuition

**Intuition**: Instead of finding any separating hyperplane, find the one with the largest margin (distance to nearest points). This provides the best generalization.

**Support Vectors**: The training examples closest to the decision boundary. They "support" the margin.

### Hard Margin SVM

**Goal**: Find hyperplane $w^T x + b = 0$ that:
1. Correctly classifies all training examples
2. Maximizes the margin

**Margin**: Distance from hyperplane to nearest point.

### Soft Margin SVM

**Problem**: Data may not be linearly separable.

**Solution**: Allow some misclassifications but penalize them. Balance between margin size and classification errors.

### Kernel Trick

**Intuition**: Map data to higher-dimensional space where it becomes linearly separable, without explicitly computing the transformation.

**Kernel Function**: $K(x, x') = \phi(x)^T \phi(x')$ computes dot product in feature space without computing $\phi(x)$.

## Mathematical Foundations

### Hard Margin SVM

Given training data $\{(x_i, y_i)\}_{i=1}^m$ where $y_i \in \{-1, +1\}$.

**Hyperplane**: $w^T x + b = 0$

**Distance from point to hyperplane**:

$$d = \frac{|w^T x + b|}{\|w\|}$$

**Margin**: Distance to nearest point:

$$\gamma = \min_i \frac{y_i(w^T x_i + b)}{\|w\|}$$

**Optimization Problem**:

Maximize margin $\gamma$ subject to all points correctly classified:

$$\max_{w,b} \gamma \quad \text{s.t.} \quad y_i(w^T x_i + b) \geq \gamma \|w\|, \forall i$$

This is equivalent to:

$$\min_{w,b} \frac{1}{2}\|w\|^2 \quad \text{s.t.} \quad y_i(w^T x_i + b) \geq 1, \forall i$$

### Lagrangian Formulation

Introduce Lagrange multipliers $\alpha_i \geq 0$:

$$L(w, b, \alpha) = \frac{1}{2}\|w\|^2 - \sum_{i=1}^m \alpha_i [y_i(w^T x_i + b) - 1]$$

**Dual Problem** (maximize w.r.t. $\alpha$, minimize w.r.t. $w, b$):

$$\max_\alpha \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y_i y_j x_i^T x_j$$

subject to:
- $\sum_{i=1}^m \alpha_i y_i = 0$
- $\alpha_i \geq 0, \forall i$

**Solution**:

$$w = \sum_{i=1}^m \alpha_i y_i x_i$$

$$b = y_k - w^T x_k \quad \text{for any support vector } x_k$$

**Support Vectors**: Points with $\alpha_i > 0$ (on or inside margin).

### Soft Margin SVM

Introduce slack variables $\xi_i \geq 0$:

$$\min_{w,b,\xi} \frac{1}{2}\|w\|^2 + C \sum_{i=1}^m \xi_i$$

subject to:
- $y_i(w^T x_i + b) \geq 1 - \xi_i, \forall i$
- $\xi_i \geq 0, \forall i$

**Parameter $C$**: Controls trade-off between margin size and classification errors.
- Large $C$: Hard margin (fewer errors allowed)
- Small $C$: Soft margin (more errors allowed)

**Dual Problem** (similar, but with upper bound on $\alpha_i$):

$$\max_\alpha \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y_i y_j x_i^T x_j$$

subject to:
- $\sum_{i=1}^m \alpha_i y_i = 0$
- $0 \leq \alpha_i \leq C, \forall i$

### Kernel Methods

**Feature Mapping**: $\phi: \mathcal{X} \rightarrow \mathcal{F}$ (maps to higher-dimensional space)

**Kernel Function**: $K(x, x') = \phi(x)^T \phi(x')$

**Kernelized Decision Function**:

$$f(x) = \sum_{i=1}^m \alpha_i y_i K(x_i, x) + b$$

**Common Kernels**:

1. **Linear**: $K(x, x') = x^T x'$

2. **Polynomial**: $K(x, x') = (x^T x' + c)^d$

3. **RBF (Gaussian)**: $K(x, x') = \exp(-\gamma \|x - x'\|^2)$

4. **Sigmoid**: $K(x, x') = \tanh(\kappa x^T x' + \theta)$

### VC Dimension and Generalization

**VC Dimension** of a hypothesis class $\mathcal{H}$: Largest number of points that can be shattered by $\mathcal{H}$.

**Shattering**: For $n$ points, if $\mathcal{H}$ can realize all $2^n$ possible labelings, the points are shattered.

**Bound on VC Dimension of SVMs**:

For margin $\gamma$ and data in ball of radius $R$:

$$VC \leq \min\left(\left\lceil\frac{R^2}{\gamma^2}\right\rceil, d\right) + 1$$

where $d$ is the dimensionality.

**Generalization Bound**:

With probability $1-\delta$:

$$R(h) \leq R_{emp}(h) + \sqrt{\frac{VC \log(2m/VC) + \log(4/\delta)}{m}}$$

## Visual/Graphical Illustrations

### Maximum Margin Hyperplane

```
x2
│
│  ●  ●  ●
│   ●   ●
│    ╱───╲  ← Margin
│   ╱     ╲
│  ╱       ╲  ← Decision Boundary
│ ╱         ╲
│╱           ╲
│─────────────── x1
│           ╱
│          ╱
│    ○  ○ ╱
│   ○   ○
│  ○  ○  ○
```

### Support Vectors

Support vectors are the points on or inside the margin boundaries.

### Kernel Trick Visualization

```
Original Space        Feature Space
    │                      │
  ● │ ○                  ● │ ○
    │                      │
  ○ │ ●                  ● │ ○
────┼────              ────┼────
    │                      │
(Not separable)      (Linearly separable)
```

## Worked Examples

### Example 1: Hard Margin SVM

**Data**: Linearly separable 2D points
- Class +1: (1,1), (2,2), (2,0)
- Class -1: (0,0), (1,0), (0,1)

**Solution**: Find $w$ and $b$ such that margin is maximized.

For this simple case, the optimal hyperplane might be $x_1 + x_2 - 1.5 = 0$.

### Example 2: RBF Kernel

**Problem**: Non-linearly separable data (concentric circles).

**Solution**: Use RBF kernel $K(x, x') = \exp(-\gamma \|x - x'\|^2)$.

This implicitly maps to infinite-dimensional space where data becomes separable.

## Code Implementation

### SVM from Scratch (Simplified)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

class SimpleSVM:
    """Simplified SVM implementation using quadratic programming."""
    def __init__(self, C=1.0, kernel='linear', gamma='scale'):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.alpha = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.b = None
        self.X_train = None
        self.y_train = None
    
    def _linear_kernel(self, X1, X2):
        """Linear kernel."""
        return X1 @ X2.T
    
    def _rbf_kernel(self, X1, X2):
        """RBF (Gaussian) kernel."""
        if self.gamma == 'scale':
            gamma = 1.0 / (X1.shape[1] * X1.var())
        else:
            gamma = self.gamma
        
        pairwise_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + \
                        np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
        return np.exp(-gamma * pairwise_dists)
    
    def _kernel_matrix(self, X1, X2):
        """Compute kernel matrix."""
        if self.kernel == 'linear':
            return self._linear_kernel(X1, X2)
        elif self.kernel == 'rbf':
            return self._rbf_kernel(X1, X2)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def fit(self, X, y):
        """Train SVM using simplified SMO-like algorithm."""
        self.X_train = X
        self.y_train = y
        n_samples = X.shape[0]
        
        # Initialize
        self.alpha = np.zeros(n_samples)
        self.b = 0
        
        # Simplified training (for demonstration)
        # In practice, use proper QP solver or SMO algorithm
        K = self._kernel_matrix(X, X)
        
        # Simplified: Use sklearn's SVC for proper implementation
        # This is a placeholder showing the structure
        print("Note: For production, use sklearn.svm.SVC")
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        K = self._kernel_matrix(self.X_train, X)
        predictions = np.sign((self.alpha * self.y_train) @ K + self.b)
        return predictions

# Use sklearn's SVM for proper implementation
def demonstrate_svm():
    # Linearly separable data
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                              n_informative=2, n_clusters_per_class=1,
                              random_state=42)
    y = 2 * y - 1  # Convert to {-1, +1}
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Linear SVM
    svm_linear = SVC(kernel='linear', C=1.0)
    svm_linear.fit(X_train, y_train)
    y_pred_linear = svm_linear.predict(X_test)
    print(f"Linear SVM Accuracy: {accuracy_score(y_test, y_pred_linear):.3f}")
    
    # RBF SVM
    svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm_rbf.fit(X_train, y_train)
    y_pred_rbf = svm_rbf.predict(X_test)
    print(f"RBF SVM Accuracy: {accuracy_score(y_test, y_pred_rbf):.3f}")
    
    # Visualize decision boundaries
    def plot_decision_boundary(X, y, model, title):
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black')
        
        # Mark support vectors
        if hasattr(model, 'support_vectors_'):
            plt.scatter(model.support_vectors_[:, 0],
                       model.support_vectors_[:, 1],
                       s=200, facecolors='none', edgecolors='black',
                       linewidths=2, label='Support Vectors')
        
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(title)
        plt.legend()
        plt.show()
    
    plot_decision_boundary(X_train, y_train, svm_linear, 'Linear SVM')
    plot_decision_boundary(X_train, y_train, svm_rbf, 'RBF Kernel SVM')

demonstrate_svm()
```

### Kernel Comparison

```python
# Compare different kernels on non-linearly separable data
X, y = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)
y = 2 * y - 1  # Convert to {-1, +1}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for idx, kernel in enumerate(kernels):
    svm = SVC(kernel=kernel, C=1.0, gamma='scale')
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # Plot decision boundary
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    axes[idx].contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    axes[idx].scatter(X_train[:, 0], X_train[:, 1], c=y_train, 
                     cmap='RdYlBu', edgecolors='black', alpha=0.6)
    axes[idx].set_title(f'{kernel.upper()} Kernel (Acc: {acc:.3f})')
    axes[idx].set_xlabel('Feature 1')
    axes[idx].set_ylabel('Feature 2')

plt.tight_layout()
plt.show()
```

### Effect of C Parameter

```python
# Show effect of C on margin and support vectors
C_values = [0.1, 1.0, 10.0, 100.0]
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                          n_informative=2, n_clusters_per_class=1,
                          random_state=42)
y = 2 * y - 1

for idx, C in enumerate(C_values):
    svm = SVC(kernel='linear', C=C)
    svm.fit(X, y)
    
    # Plot
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    axes[idx].contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    axes[idx].scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black')
    axes[idx].scatter(svm.support_vectors_[:, 0],
                     svm.support_vectors_[:, 1],
                     s=200, facecolors='none', edgecolors='red',
                     linewidths=2)
    axes[idx].set_title(f'C = {C} (Support Vectors: {len(svm.support_vectors_)})')
    axes[idx].set_xlabel('Feature 1')
    axes[idx].set_ylabel('Feature 2')

plt.tight_layout()
plt.show()
```

## Conceptual Summary / Diagram

### SVM Learning Process

```
Training Data
    │
    ▼
[Formulate Optimization Problem]
    │
    ├── Hard Margin (if separable)
    └── Soft Margin (with slack)
    │
    ▼
[Solve Dual Problem]
    │
    ▼
[Find Support Vectors]
    │
    ▼
[Construct Decision Function]
    │
    ▼
Classifier
```

### Kernel Methods

```
Input Space          Feature Space
    │                    │
    │  φ(x)               │
    ├─────────────────────┤
    │                    │
  ● │ ○                ● │ ○
    │                    │
  ○ │ ●                ● │ ○
────┼────            ────┼────
    │                    │
(Complex)          (Linear)
```

## Common Misconceptions / Pitfalls

### Misconception 1: "SVM Always Finds Global Optimum"

**Reality**: The dual problem is convex, so global optimum exists. However, practical solvers (SMO) may have numerical issues.

### Misconception 2: "More Support Vectors Means Better Model"

**Reality**: Fewer support vectors often indicate better generalization (larger margin). Many support vectors may indicate overfitting.

### Misconception 3: "RBF Kernel Always Better Than Linear"

**Reality**: 
- Linear: Fast, interpretable, works if data is linearly separable
- RBF: Powerful but can overfit, requires careful tuning of $\gamma$

### Pitfall: Poor Kernel Choice

Choosing wrong kernel or parameters leads to poor performance. Always:
- Try linear first
- Use cross-validation for hyperparameter tuning
- Understand your data's structure

### Pitfall: Scaling Sensitivity

SVMs are sensitive to feature scaling. Always standardize features before training.

## Practice Exercises

### Exercise 1: Margin Calculation

For a hyperplane $w^T x + b = 0$ and a point $x_i$ with label $y_i$, derive the formula for the margin:
$$\gamma = \min_i \frac{y_i(w^T x_i + b)}{\|w\|}$$

### Exercise 2: Dual Formulation

Starting from the primal SVM problem, derive the dual formulation using Lagrange multipliers.

### Exercise 3: Kernel Properties

Show that the RBF kernel $K(x, x') = \exp(-\gamma \|x - x'\|^2)$ is a valid kernel (positive definite).

### Exercise 4: VC Dimension

For a 2D dataset with margin $\gamma$ and data in a circle of radius $R$, calculate an upper bound on the VC dimension.

## Summary & Key Takeaways

### Key Concepts

1. **Maximum Margin**: SVM finds hyperplane with largest margin
2. **Support Vectors**: Training examples that define the margin
3. **Soft Margin**: Allows misclassifications with penalty $C$
4. **Kernel Trick**: Implicit mapping to higher dimensions
5. **VC Dimension**: Theoretical foundation for generalization

### Important Principles

- **Margin Maximization**: Provides regularization and good generalization
- **Dual Formulation**: Enables kernel methods
- **Sparsity**: Only support vectors matter (memory efficient)
- **Kernel Selection**: Critical for non-linear problems

### Next Steps

Understanding SVMs prepares you for:
- Kernel methods in general
- Statistical learning theory
- Neural networks (similar optimization concepts)

## References / Further Reading

### Primary References

1. **Burges, C. J. C.** (1998). A Tutorial on Support Vector Machines for Pattern Recognition. *Data Mining and Knowledge Discovery*, 2(2), 121-167.

2. **Bishop, C. M.** (2006). *Pattern Recognition & Machine Learning*.
   - Chapter 7: Sparse Kernel Machines

### Research Papers

3. **Cortes, C., & Vapnik, V.** (1995). Support-vector networks. *Machine Learning*, 20(3), 273-297.

4. **Vapnik, V. N.** (1998). *Statistical Learning Theory*. Wiley.

### VC Dimension

5. **Vapnik, V., & Chervonenkis, A.** (1971). On the uniform convergence of relative frequencies of events to their probabilities. *Theory of Probability and its Applications*, 16(2), 264-280.

---

**Previous Chapter**: [Chapter 7: Instance-Based Learning](07_instance_based.md) | **Next Chapter**: [Chapter 9: Neural Networks](09_neural_networks.md)

