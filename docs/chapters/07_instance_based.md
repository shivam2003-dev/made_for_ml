# Chapter 7: Instance-Based Learning

## Overview & Motivation

Instance-based learning (also called lazy learning) differs from eager learning methods by deferring generalization until prediction time. Instead of building an explicit model, these methods store training examples and make predictions based on similarity to stored instances. This chapter covers K-Nearest Neighbors (KNN), Case-Based Reasoning (CBR), and distance metrics.

!!! tip "Lazy vs Eager Learning"
    **Eager Learning** (Decision Trees, Neural Networks):
    - Build model during training
    - Fast prediction, slower training
    - Model is compact
    
    **Lazy Learning** (KNN, CBR):
    - Store all training data
    - Fast training, slower prediction
    - Model = the data itself
    
    Choose based on your priorities: prediction speed vs training speed.

Instance-based learning is valuable because:
- No explicit model training phase
- Can adapt to local patterns in data
- Naturally handles multi-modal distributions
- Simple to understand and implement
- Effective for non-linear problems

## Core Theory & Intuitive Explanation

### Lazy vs Eager Learning

**Eager Learning** (e.g., decision trees, neural networks):
- Build model during training
- Discard training data after model is built
- Fast prediction, slower training

**Lazy Learning** (e.g., KNN):
- Store all training examples
- Build model during prediction
- Fast training, slower prediction

### K-Nearest Neighbors (KNN)

**Intuition**: "Tell me who your neighbors are, and I'll tell you who you are."

For a new example:
1. Find the $k$ nearest training examples
2. For classification: Majority vote of their labels
3. For regression: Average of their values

### Case-Based Reasoning (CBR)

**Intuition**: Solve new problems by adapting solutions from similar past cases.

**Process**:
1. **Retrieve**: Find similar past cases
2. **Reuse**: Adapt solution from similar case
3. **Revise**: Adjust solution if needed
4. **Retain**: Store new case for future use

## Mathematical Foundations

### Distance Metrics

**Euclidean Distance**:

$$d(x, x') = \sqrt{\sum_{i=1}^d (x_i - x'_i)^2} = \|x - x'\|_2$$

**Manhattan Distance**:

$$d(x, x') = \sum_{i=1}^d |x_i - x'_i| = \|x - x'\|_1$$

**Minkowski Distance** (generalization):

$$d_p(x, x') = \left(\sum_{i=1}^d |x_i - x'_i|^p\right)^{1/p}$$

- $p=1$: Manhattan
- $p=2$: Euclidean
- $p \to \infty$: Chebyshev ($\max_i |x_i - x'_i|$)

**Cosine Similarity**:

$$\text{cos}(\theta) = \frac{x \cdot x'}{\|x\| \|x'\|} = \frac{\sum_{i=1}^d x_i x'_i}{\sqrt{\sum_{i=1}^d x_i^2} \sqrt{\sum_{i=1}^d x'_i^2}}$$

### KNN Classification

For query point $x_q$, find $k$ nearest neighbors $N_k(x_q)$.

**Prediction**:

$$\hat{y} = \text{mode}(\{y_i : x_i \in N_k(x_q)\})$$

**Weighted KNN** (distance-weighted):

$$\hat{y} = \text{arg}\max_{v \in V} \sum_{x_i \in N_k(x_q)} w_i \cdot \mathbb{1}[y_i = v]$$

where weights $w_i = \frac{1}{d(x_q, x_i)}$ (inverse distance).

### KNN Regression

**Prediction**:

$$\hat{y} = \frac{1}{k} \sum_{x_i \in N_k(x_q)} y_i$$

**Weighted version**:

$$\hat{y} = \frac{\sum_{x_i \in N_k(x_q)} w_i y_i}{\sum_{x_i \in N_k(x_q)} w_i}$$

### Curse of Dimensionality

As dimensionality $d$ increases:
- All points become approximately equidistant
- Distance metrics become less meaningful
- Need exponentially more data

**Volume of $d$-dimensional sphere**:

$$V_d(r) = \frac{\pi^{d/2} r^d}{\Gamma(d/2 + 1)}$$

Most volume is near the surface in high dimensions.

## Visual/Graphical Illustrations

### KNN Classification (k=3)

```
x2
│
│  ●  ●
│    ●
│    ○  ← Query point
│  ●  ●    3 nearest neighbors
│    ●     (2 of class ●, 1 of class ○)
│  ○  ○    Prediction: ●
│
└─────────────────── x1
```

### Distance Metrics Comparison

```
Euclidean:    Manhattan:    Chebyshev:
    ╱│╲           │              │
   ╱ │ ╲          │              │
  ╱  │  ╲         │              │
 ╱   │   ╲        │              │
╱    │    ╲       │              │
─────┼─────        ───────────────
     │
```

## Worked Examples

### Example 1: KNN Classification

**Data**: 2D points with classes
- Class A: (1,1), (1,2), (2,1)
- Class B: (5,5), (6,5), (5,6)

**Query**: (3,3)

**Distances** (Euclidean):
- To (1,1): $\sqrt{(3-1)^2 + (3-1)^2} = \sqrt{8} \approx 2.83$
- To (1,2): $\sqrt{(3-1)^2 + (3-2)^2} = \sqrt{5} \approx 2.24$
- To (2,1): $\sqrt{(3-2)^2 + (3-1)^2} = \sqrt{5} \approx 2.24$
- To (5,5): $\sqrt{(3-5)^2 + (3-5)^2} = \sqrt{8} \approx 2.83$
- To (6,5): $\sqrt{(3-6)^2 + (3-5)^2} = \sqrt{13} \approx 3.61$
- To (5,6): $\sqrt{(3-5)^2 + (3-6)^2} = \sqrt{13} \approx 3.61$

**For k=3**: Nearest are (1,2), (2,1), (1,1) → All Class A → Predict A

**For k=5**: Add (5,5) → 4 A, 1 B → Predict A

### Example 2: Weighted KNN

Same data, but use inverse distance weights:
- $w(1,2) = 1/2.24 \approx 0.45$
- $w(2,1) = 1/2.24 \approx 0.45$
- $w(1,1) = 1/2.83 \approx 0.35$

Weighted vote favors closer neighbors more.

## Code Implementation

### KNN from Scratch

```python
import numpy as np
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class KNN:
    def __init__(self, k=3, distance_metric='euclidean', weights='uniform'):
        self.k = k
        self.distance_metric = distance_metric
        self.weights = weights
        self.X_train = None
        self.y_train = None
    
    def _euclidean_distance(self, x1, x2):
        """Calculate Euclidean distance."""
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def _manhattan_distance(self, x1, x2):
        """Calculate Manhattan distance."""
        return np.sum(np.abs(x1 - x2))
    
    def _distance(self, x1, x2):
        """Calculate distance based on metric."""
        if self.distance_metric == 'euclidean':
            return self._euclidean_distance(x1, x2)
        elif self.distance_metric == 'manhattan':
            return self._manhattan_distance(x1, x2)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    def fit(self, X, y):
        """Store training data (lazy learning)."""
        self.X_train = X
        self.y_train = y
        return self
    
    def _get_neighbors(self, x):
        """Find k nearest neighbors."""
        distances = []
        for i, x_train in enumerate(self.X_train):
            dist = self._distance(x, x_train)
            distances.append((dist, i))
        
        # Sort by distance and get k nearest
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:self.k]
        return neighbors
    
    def predict(self, X):
        """Predict for multiple samples."""
        predictions = []
        for x in X:
            neighbors = self._get_neighbors(x)
            
            if self.weights == 'uniform':
                # Simple majority vote
                neighbor_labels = [self.y_train[idx] for _, idx in neighbors]
                prediction = Counter(neighbor_labels).most_common(1)[0][0]
            else:  # distance-weighted
                # Weighted vote
                class_weights = {}
                for dist, idx in neighbors:
                    label = self.y_train[idx]
                    weight = 1 / (dist + 1e-10)  # Avoid division by zero
                    class_weights[label] = class_weights.get(label, 0) + weight
                prediction = max(class_weights, key=class_weights.get)
            
            predictions.append(prediction)
        
        return np.array(predictions)

# Generate data
X, y = make_classification(n_samples=200, n_features=2, n_informative=2,
                          n_redundant=0, n_classes=2, n_clusters_per_class=1,
                          random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Test different k values
k_values = [1, 3, 5, 7, 9, 11]
accuracies = []

for k in k_values:
    knn = KNN(k=k, distance_metric='euclidean', weights='uniform')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"k={k}: Accuracy = {acc:.3f}")

# Plot k vs accuracy
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o')
plt.xlabel('k (number of neighbors)')
plt.ylabel('Accuracy')
plt.title('KNN: Effect of k on Accuracy')
plt.grid(True, alpha=0.3)
plt.show()

# Visualize decision boundary
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
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.show()

knn_vis = KNN(k=5)
knn_vis.fit(X_train, y_train)
plot_decision_boundary(X_train, y_train, knn_vis, 'KNN Decision Boundary (k=5)')
```

### KNN Regression

```python
class KNNRegressor:
    def __init__(self, k=3, distance_metric='euclidean', weights='uniform'):
        self.k = k
        self.distance_metric = distance_metric
        self.weights = weights
        self.X_train = None
        self.y_train = None
    
    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def _distance(self, x1, x2):
        return self._euclidean_distance(x1, x2)
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        return self
    
    def _get_neighbors(self, x):
        distances = []
        for i, x_train in enumerate(self.X_train):
            dist = self._distance(x, x_train)
            distances.append((dist, i))
        distances.sort(key=lambda x: x[0])
        return distances[:self.k]
    
    def predict(self, X):
        predictions = []
        for x in X:
            neighbors = self._get_neighbors(x)
            
            if self.weights == 'uniform':
                neighbor_values = [self.y_train[idx] for _, idx in neighbors]
                prediction = np.mean(neighbor_values)
            else:  # distance-weighted
                weights = []
                values = []
                for dist, idx in neighbors:
                    weight = 1 / (dist + 1e-10)
                    weights.append(weight)
                    values.append(self.y_train[idx])
                prediction = np.average(values, weights=weights)
            
            predictions.append(prediction)
        
        return np.array(predictions)

# Test on regression problem
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

X_reg, y_reg = make_regression(n_samples=200, n_features=1, noise=10, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

knn_reg = KNNRegressor(k=5, weights='distance')
knn_reg.fit(X_train_reg, y_train_reg)
y_pred_reg = knn_reg.predict(X_test_reg)

print(f"KNN Regression MSE: {mean_squared_error(y_test_reg, y_pred_reg):.2f}")

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X_train_reg, y_train_reg, alpha=0.6, label='Training data')
plt.scatter(X_test_reg, y_test_reg, alpha=0.6, label='Test data')
X_plot = np.linspace(X_reg.min(), X_reg.max(), 100).reshape(-1, 1)
y_plot = knn_reg.predict(X_plot)
plt.plot(X_plot, y_plot, 'r-', linewidth=2, label='KNN prediction')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('KNN Regression')
plt.grid(True, alpha=0.3)
plt.show()
```

### Case-Based Reasoning Example

```python
class SimpleCBR:
    """Simple Case-Based Reasoning system."""
    def __init__(self, similarity_threshold=0.7):
        self.cases = []
        self.similarity_threshold = similarity_threshold
    
    def add_case(self, problem, solution):
        """Store a case."""
        self.cases.append({'problem': problem, 'solution': solution})
    
    def _similarity(self, problem1, problem2):
        """Calculate similarity between problems."""
        # Simple cosine similarity
        dot_product = np.dot(problem1, problem2)
        norm1 = np.linalg.norm(problem1)
        norm2 = np.linalg.norm(problem2)
        if norm1 == 0 or norm2 == 0:
            return 0
        return dot_product / (norm1 * norm2)
    
    def retrieve(self, query_problem):
        """Retrieve similar cases."""
        similar_cases = []
        for case in self.cases:
            sim = self._similarity(query_problem, case['problem'])
            if sim >= self.similarity_threshold:
                similar_cases.append((sim, case))
        
        # Sort by similarity
        similar_cases.sort(key=lambda x: x[0], reverse=True)
        return similar_cases
    
    def solve(self, query_problem):
        """Solve using CBR."""
        similar_cases = self.retrieve(query_problem)
        if not similar_cases:
            return None, 0.0
        
        # Reuse solution from most similar case
        best_sim, best_case = similar_cases[0]
        return best_case['solution'], best_sim

# Example usage
cbr = SimpleCBR(similarity_threshold=0.5)

# Add some cases
cbr.add_case(np.array([1, 2, 3]), 'Solution A')
cbr.add_case(np.array([4, 5, 6]), 'Solution B')
cbr.add_case(np.array([7, 8, 9]), 'Solution C')

# Query
query = np.array([1.1, 2.1, 3.1])
solution, similarity = cbr.solve(query)
print(f"Query: {query}")
print(f"Solution: {solution}")
print(f"Similarity: {similarity:.3f}")
```

## Conceptual Summary / Diagram

### KNN Algorithm Flow

```
Query Point x_q
    │
    ▼
[Calculate distances to all training points]
    │
    ▼
[Sort by distance]
    │
    ▼
[Select k nearest]
    │
    ▼
[Vote (classification) or Average (regression)]
    │
    ▼
Prediction
```

### Lazy vs Eager Learning

| Aspect | Eager Learning | Lazy Learning |
|--------|---------------|---------------|
| **Training** | Build model | Store data |
| **Time** | Slow training, fast prediction | Fast training, slow prediction |
| **Memory** | Model parameters | All training data |
| **Examples** | Decision trees, Neural nets | KNN, CBR |

## Common Misconceptions / Pitfalls

### Misconception 1: "KNN Works Well in High Dimensions"

**Reality**: KNN suffers from the curse of dimensionality. In high dimensions, all points are approximately equidistant, making distance metrics meaningless.

### Misconception 2: "Larger k Always Better"

**Reality**: 
- Small $k$: Sensitive to noise, overfitting
- Large $k$: Smooths out local patterns, underfitting
- Optimal $k$ depends on data (use cross-validation)

### Misconception 3: "KNN Doesn't Need Feature Scaling"

**Reality**: Distance metrics are sensitive to feature scales. Always normalize features before applying KNN.

### Pitfall: Computational Cost

KNN requires computing distances to all training examples for each prediction. For large datasets, this is expensive. Solutions:
- Use approximate nearest neighbors (LSH, KD-trees)
- Reduce dataset size (prototype selection)

### Pitfall: Irrelevant Features

KNN treats all features equally. Irrelevant features hurt performance. Use feature selection or weighted distances.

## Practice Exercises

### Exercise 1: Distance Metrics

Implement and compare Euclidean, Manhattan, and Chebyshev distances. Show that for 2D points, they give different nearest neighbors.

### Exercise 2: Optimal k Selection

Use cross-validation to find the optimal $k$ for KNN on a classification dataset. Plot validation accuracy vs $k$.

### Exercise 3: Weighted KNN

Implement distance-weighted KNN where closer neighbors have more influence. Compare with uniform KNN.

### Exercise 4: Curse of Dimensionality

Generate data in increasing dimensions (1D, 2D, 5D, 10D, 20D) and show how KNN accuracy degrades. Visualize how distances become more uniform.

## Summary & Key Takeaways

### Key Concepts

1. **Lazy Learning**: Defer generalization until prediction
2. **KNN**: Predict based on $k$ nearest neighbors
3. **Distance Metrics**: Euclidean, Manhattan, Cosine, etc.
4. **CBR**: Solve problems by adapting similar past cases
5. **Curse of Dimensionality**: High dimensions hurt distance-based methods

### Important Principles

- **Local vs Global**: KNN captures local patterns
- **No Training**: Fast to "train" (just store data)
- **Distance Matters**: Choice of metric is crucial
- **Dimensionality**: Performance degrades in high dimensions

### Next Steps

Understanding instance-based learning prepares you for:
- Kernel methods (similarity-based)
- Support Vector Machines (Chapter 8)
- Clustering algorithms (similar distance concepts)

## References / Further Reading

??? note "📚 Primary References"

    1. **Mitchell, T. M.** (1997). *Machine Learning*.
       - Chapter 8: Instance-Based Learning

    2. **Aha, D. W., Kibler, D., & Albert, M. K.** (1991). Instance-based learning algorithms. *Machine Learning*, 6(1), 37-66.
       - [DOI: 10.1023/A:1022689900470](https://doi.org/10.1023/A:1022689900470)
       - [PDF](https://link.springer.com/article/10.1023/A:1022689900470)

??? note "📖 Case-Based Reasoning"

    3. **Kolodner, J.** (1993). *Case-Based Reasoning*. Morgan Kaufmann.
       - [Amazon](https://www.amazon.com/Case-Based-Reasoning-Janet-Kolodner/dp/1558602372)

    4. **Aamodt, A., & Plaza, E.** (1994). Case-based reasoning: Foundational issues, methodological variations, and system approaches. *AI Communications*, 7(1), 39-59.
       - [PDF](https://www.researchgate.net/publication/220420529_Case-Based_Reasoning_Foundational_Issues_Methodological_Variations_and_System_Approaches)

??? note "🔬 Research Papers"

    5. **Cover, T., & Hart, P.** (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory*, 13(1), 21-27.
       - [DOI: 10.1109/TIT.1967.1053964](https://doi.org/10.1109/TIT.1967.1053964)
       - [IEEE Xplore](https://ieeexplore.ieee.org/document/1053964)

    ---

## Recommended Reads

??? note "📚 Official Documentation"

    - **K-Nearest Neighbors** - [Scikit-learn KNN](https://scikit-learn.org/stable/modules/neighbors.html)

    - **Nearest Neighbors** - [Neighbor algorithms](https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-classification)

    - **Distance Metrics** - [Distance metrics reference](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html)

??? note "📖 Essential Articles"

    - **KNN Tutorial** - [Complete KNN guide](https://machinelearningmastery.com/k-nearest-neighbors-for-machine-learning/)

    - **Choosing k in KNN** - [Selecting optimal k](https://towardsdatascience.com/how-to-find-the-optimal-value-of-k-in-knn-35d936e554eb)

    - **Distance Metrics** - [Understanding distance metrics](https://towardsdatascience.com/9-distance-measures-in-data-science-918109d069fa)

??? note "🎓 Learning Resources"

    - **Curse of Dimensionality** - [Understanding high-dimensional problems](https://towardsdatascience.com/the-curse-of-dimensionality-50a6b05cedbf)

    - **KNN Optimization** - [Optimizing KNN performance](https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-algorithms)

    - **Weighted KNN** - [Distance-weighted KNN](https://scikit-learn.org/stable/modules/neighbors.html#classification)

??? note "💡 Best Practices"

    - **Feature Scaling** - [Scaling for KNN](https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling)

    - **KNN Best Practices** - [KNN best practices](https://scikit-learn.org/stable/modules/neighbors.html#tips-on-using-knn)

    - **Handling High Dimensions** - [Dimensionality reduction for KNN](https://scikit-learn.org/stable/modules/decomposition.html)

??? note "🔬 Research Papers"

    - **Nearest Neighbor Classification** - [Cover & Hart (1967)](https://doi.org/10.1109/TIT.1967.1053964) - Foundational KNN paper

    - **Instance-Based Learning** - [Aha et al. (1991)](https://doi.org/10.1023/A:1022689900470) - Instance-based learning algorithms

    ---

    **Previous Chapter**: [Chapter 6: Decision Trees](06_decision_trees.md) | **Next Chapter**: [Chapter 8: Support Vector Machines](08_svm.md)

