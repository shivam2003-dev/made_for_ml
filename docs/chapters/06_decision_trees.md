# Chapter 6: Decision Trees

## Overview & Motivation

Decision trees are intuitive, interpretable models that make decisions by asking a series of questions. They partition the feature space into regions and assign predictions based on these partitions. This chapter covers tree construction algorithms (ID3, C4.5, CART), splitting criteria, pruning, and ensemble methods.

!!! important "Why Decision Trees?"
    Decision trees are among the most interpretable ML models:
    - **Human-readable**: Can be explained to non-technical stakeholders
    - **No preprocessing**: Work with raw data (missing values, mixed types)
    - **Feature importance**: Automatically identify important features
    - **Non-linear**: Capture complex interactions without feature engineering
    - **Foundation**: Basis for Random Forest, XGBoost, LightGBM (state-of-the-art)

Decision trees are valuable because:
- Highly interpretable (can be visualized)
- Handle both numerical and categorical features
- Require little data preprocessing
- Can model non-linear relationships
- Form the basis for powerful ensemble methods (Random Forest, XGBoost)

## Core Theory & Intuitive Explanation

### Basic Concept

**Intuition**: A decision tree asks questions about features, branching based on answers, until reaching a leaf node with a prediction.

**Structure**:
- **Root node**: Top of the tree
- **Internal nodes**: Decision points (test on features)
- **Leaf nodes**: Predictions (class labels or values)

### Tree Construction

**Algorithm** (recursive):
1. Start with all training examples at root
2. Select best feature to split on
3. Partition data based on feature values
4. Recursively build subtrees for each partition
5. Stop when stopping criterion met (pure node, max depth, etc.)

### Splitting Criteria

**Goal**: Choose splits that best separate classes or reduce prediction error.

**Measures**:
- **Information Gain** (ID3): Based on entropy reduction
- **Gini Impurity** (CART): Measure of class mixing
- **Variance Reduction** (Regression): For continuous targets

## Mathematical Foundations

### Entropy

Measure of impurity/uncertainty:

$$H(S) = -\sum_{i=1}^c p_i \log_2 p_i$$

where $p_i$ is the proportion of class $i$ in set $S$, and $c$ is the number of classes.

**Properties**:
- $H(S) = 0$ when $S$ is pure (one class)
- $H(S)$ is maximum when classes are equally likely

### Information Gain

For a split on feature $A$:

$$\text{IG}(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)$$

where $S_v$ is the subset of $S$ where $A = v$.

**Intuition**: How much uncertainty is reduced by splitting on $A$.

### Gini Impurity

Alternative measure of impurity:

$$\text{Gini}(S) = 1 - \sum_{i=1}^c p_i^2$$

**Gini Gain** (for split on $A$):

$$\text{GiniGain}(S, A) = \text{Gini}(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \text{Gini}(S_v)$$

### Variance Reduction (Regression)

For regression trees, minimize variance:

$$\text{Var}(S) = \frac{1}{|S|} \sum_{x_i \in S} (y_i - \bar{y})^2$$

**Variance Reduction**:

$$\text{VarRed}(S, A) = \text{Var}(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \text{Var}(S_v)$$

### ID3 Algorithm

**Input**: Training examples $S$, features $F$, target attribute

**Output**: Decision tree

```
1. If all examples in S have same class, return leaf with that class
2. If F is empty, return leaf with majority class in S
3. Select feature A in F that maximizes IG(S, A)
4. For each value v of A:
   a. Create branch for A = v
   b. Let S_v = subset of S where A = v
   c. If S_v is empty, add leaf with majority class
      Else recursively call ID3(S_v, F - {A})
5. Return tree
```

### C4.5 Improvements

1. **Handles continuous features**: Find optimal threshold
2. **Handles missing values**: Use probability distributions
3. **Uses Gain Ratio** instead of Information Gain:

$$\text{GainRatio}(S, A) = \frac{\text{IG}(S, A)}{\text{SplitInfo}(S, A)}$$

where:

$$\text{SplitInfo}(S, A) = -\sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \log_2 \frac{|S_v|}{|S|}$$

This penalizes features with many values.

### Pruning

**Problem**: Full trees overfit training data.

**Solution**: Remove branches that don't improve generalization.

**Pre-pruning**: Stop early (max depth, min samples per leaf)

**Post-pruning**: Build full tree, then remove branches

**Cost-Complexity Pruning**:

Minimize: $\text{Error}(T) + \alpha \cdot |T|$

where $|T|$ is the number of leaves and $\alpha$ controls complexity.

## Visual/Graphical Illustrations

### Decision Tree Structure

```
            [Root: Feature A?]
                 /    \
            A<5 /      \ A≥5
               /        \
        [Feature B?]  [Leaf: Class 1]
         /      \
    B<3 /        \ B≥3
       /          \
[Leaf: Class 0] [Leaf: Class 1]
```

### Feature Space Partitioning

```
x2
│
│  ┌───┐  ┌───┐
│  │ 0 │  │ 1 │
│  ├───┼──┼───┤
│  │ 1 │  │ 0 │
│  └───┴──┴───┘
└─────────────────── x1
```

Each rectangle represents a leaf node's region.

## Worked Examples

### Example 1: Simple Classification Tree

**Data**: Weather prediction
- Features: Outlook (Sunny/Rainy/Overcast), Humidity (High/Low)
- Target: Play (Yes/No)

**Construction**:
1. Calculate entropy of root: $H(S) = -(\frac{9}{14}\log_2\frac{9}{14} + \frac{5}{14}\log_2\frac{5}{14}) \approx 0.940$

2. For Outlook:
   - Sunny: 5 examples (2 Yes, 3 No) → $H = 0.971$
   - Rainy: 5 examples (3 Yes, 2 No) → $H = 0.971$
   - Overcast: 4 examples (4 Yes, 0 No) → $H = 0$
   - Weighted average: $\frac{5}{14} \cdot 0.971 + \frac{5}{14} \cdot 0.971 + \frac{4}{14} \cdot 0 = 0.694$
   - Information Gain: $0.940 - 0.694 = 0.246$

3. Similar calculation for Humidity → IG = 0.152

4. Choose Outlook (higher IG) as root split

### Example 2: Regression Tree

**Problem**: Predict house prices from size and location.

**Splitting**: Use variance reduction instead of information gain.

## Code Implementation

### Decision Tree Classifier from Scratch

```python
import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # For leaf nodes
    
    def is_leaf(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None
    
    def _entropy(self, y):
        """Calculate entropy."""
        counts = np.bincount(y)
        proportions = counts / len(y)
        return -np.sum([p * np.log2(p) for p in proportions if p > 0])
    
    def _gini(self, y):
        """Calculate Gini impurity."""
        counts = np.bincount(y)
        proportions = counts / len(y)
        return 1 - np.sum([p**2 for p in proportions])
    
    def _information_gain(self, y_parent, y_left, y_right):
        """Calculate information gain for a split."""
        if self.criterion == 'entropy':
            parent_impurity = self._entropy(y_parent)
            left_impurity = self._entropy(y_left)
            right_impurity = self._entropy(y_right)
        else:  # gini
            parent_impurity = self._gini(y_parent)
            left_impurity = self._gini(y_left)
            right_impurity = self._gini(y_right)
        
        n = len(y_parent)
        n_left, n_right = len(y_left), len(y_right)
        
        child_impurity = (n_left / n) * left_impurity + (n_right / n) * right_impurity
        return parent_impurity - child_impurity
    
    def _best_split(self, X, y):
        """Find the best feature and threshold to split on."""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                gain = self._information_gain(y, y[left_mask], y[right_mask])
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree."""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_classes == 1 or 
            n_samples < self.min_samples_split):
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)
        
        # Find best split
        feature, threshold = self._best_split(X, y)
        
        if feature is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)
        
        # Split data
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        # Recursively build children
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return Node(feature=feature, threshold=threshold, 
                   left=left_child, right=right_child)
    
    def fit(self, X, y):
        """Train the decision tree."""
        self.root = self._build_tree(X, y)
        return self
    
    def _predict_sample(self, x, node):
        """Predict for a single sample."""
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict(self, X):
        """Make predictions."""
        return np.array([self._predict_sample(x, self.root) for x in X])

# Test on synthetic data
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1000, n_features=4, n_informative=2,
                          n_classes=2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
tree = DecisionTree(max_depth=5, min_samples_split=10, criterion='gini')
tree.fit(X_train, y_train)

# Predict
y_pred = tree.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
```

### Using sklearn Decision Trees

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
import matplotlib.pyplot as plt

# Train sklearn tree
sklearn_tree = DecisionTreeClassifier(max_depth=3, criterion='gini', random_state=42)
sklearn_tree.fit(X_train, y_train)

# Visualize tree
plt.figure(figsize=(20, 10))
plot_tree(sklearn_tree, filled=True, feature_names=[f'Feature {i}' for i in range(X.shape[1])],
          class_names=['Class 0', 'Class 1'], fontsize=10)
plt.title('Decision Tree Visualization')
plt.show()

# Text representation
tree_rules = export_text(sklearn_tree, feature_names=[f'Feature {i}' for i in range(X.shape[1])])
print(tree_rules)

# Feature importance
importances = sklearn_tree.feature_importances_
print("\nFeature Importances:")
for i, imp in enumerate(importances):
    print(f"Feature {i}: {imp:.3f}")
```

## Conceptual Summary / Diagram

### Decision Tree Learning Process

```
Training Data
    │
    ▼
[Calculate Impurity]
    │
    ▼
[Find Best Split]
    │
    ▼
[Partition Data]
    │
    ├─── Left Subtree ────┐
    │                      │
    └─── Right Subtree ────┘
            │
            ▼
    [Recurse or Stop]
```

### Splitting Criteria Comparison

| Criterion | Formula | Characteristics |
|-----------|---------|-----------------|
| **Entropy** | $-\sum p_i \log p_i$ | Information-theoretic, used in ID3 |
| **Gini** | $1 - \sum p_i^2$ | Faster to compute, used in CART |
| **Variance** | $\frac{1}{n}\sum(y_i - \bar{y})^2$ | For regression |

## Common Misconceptions / Pitfalls

### Misconception 1: "Decision Trees Don't Need Preprocessing"

**Reality**: While trees handle mixed data types, preprocessing can help:
- Feature scaling (for continuous splits)
- Handling missing values
- Encoding categorical variables

### Misconception 2: "Deeper Trees Are Always Better"

**Reality**: Deep trees overfit. Pruning is essential for generalization.

### Misconception 3: "Information Gain Always Prefers Many-Valued Features"

**Reality**: Information Gain does favor many-valued features, which is why C4.5 uses Gain Ratio to normalize.

### Pitfall: Overfitting

Decision trees can perfectly fit training data, leading to poor generalization. Always use:
- Pruning
- Cross-validation for hyperparameter tuning
- Ensemble methods

### Pitfall: Instability

Small changes in data can lead to very different trees. This is why ensemble methods (Random Forest) are preferred.

## Practice Exercises

### Exercise 1: Entropy Calculation

Given a dataset with 8 examples of class A and 2 examples of class B, calculate:
1. Entropy
2. Gini impurity

### Exercise 2: Information Gain

Given a split that creates:
- Left child: 5 A, 1 B
- Right child: 3 A, 1 B

Calculate the information gain from the parent (8 A, 2 B).

### Exercise 3: Tree Construction

Manually construct a decision tree for the following data:

| Outlook | Temperature | Humidity | Wind | Play |
|---------|------------|----------|------|------|
| Sunny   | Hot        | High     | Weak | No   |
| Sunny   | Hot        | High     | Strong | No |
| Overcast| Hot        | High     | Weak | Yes  |
| Rainy   | Mild       | High     | Weak | Yes  |
| Rainy   | Cool       | Normal   | Weak | Yes  |
| Rainy   | Cool       | Normal   | Strong | No |
| Overcast| Cool       | Normal   | Strong | Yes |
| Sunny   | Mild       | High     | Weak | No   |
| Sunny   | Cool       | Normal   | Weak | Yes  |
| Rainy   | Mild       | Normal   | Weak | Yes  |
| Sunny   | Mild       | Normal   | Strong | Yes |
| Overcast| Mild       | High     | Strong | Yes |
| Overcast| Hot        | Normal   | Weak | Yes  |
| Rainy   | Mild       | High     | Strong | No |

### Exercise 4: Pruning Implementation

Implement cost-complexity pruning for a decision tree. Test it on a dataset and show how different $\alpha$ values affect tree size and accuracy.

## Summary & Key Takeaways

### Key Concepts

1. **Decision Trees**: Hierarchical partitioning of feature space
2. **Splitting Criteria**: Information Gain, Gini, Variance Reduction
3. **Tree Construction**: Recursive partitioning until stopping criteria
4. **Pruning**: Essential to prevent overfitting
5. **Interpretability**: Trees are highly interpretable

### Important Principles

- **Greedy Construction**: Trees are built greedily (locally optimal splits)
- **Overfitting Risk**: Full trees overfit; pruning is crucial
- **Feature Selection**: Trees naturally perform feature selection
- **Non-parametric**: No assumptions about data distribution

### Next Steps

Understanding decision trees prepares you for:
- Ensemble methods (Random Forest, Boosting)
- Instance-based learning (similar partitioning ideas)
- Rule-based systems

## References / Further Reading

### Primary References

1. **Mitchell, T. M.** (1997). *Machine Learning*.
   - Chapter 3: Decision Tree Learning

2. **Quinlan, J. R.** (1986). Induction of decision trees. *Machine Learning*, 1(1), 81-106. (ID3)
   - [DOI: 10.1007/BF00116251](https://doi.org/10.1007/BF00116251)
   - [PDF](https://link.springer.com/article/10.1007/BF00116251)

3. **Quinlan, J. R.** (1993). *C4.5: Programs for Machine Learning*. Morgan Kaufmann.
   - [Amazon](https://www.amazon.com/C4-5-Programs-Machine-Learning/dp/1558602380)
   - [Morgan Kaufmann](https://www.elsevier.com/books/c45-programs-for-machine-learning/quinlan/978-1-55860-238-0)

### Research Papers

4. **Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A.** (1984). *Classification and Regression Trees*. Wadsworth.
   - [Amazon](https://www.amazon.com/Classification-Regression-Wadsworth-Statistics-Probability/dp/0412048418)
   - [Taylor & Francis](https://www.taylorfrancis.com/books/mono/10.1201/9781315139470/classification-regression-trees-leo-breiman)

5. **Breiman, L.** (2001). Random forests. *Machine Learning*, 45(1), 5-32.
   - [DOI: 10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324)
   - [PDF](https://link.springer.com/article/10.1023/A:1010933404324)

---

**Previous Chapter**: [Chapter 5: Linear Models](05_linear_models.md) | **Next Chapter**: [Chapter 7: Instance-Based Learning](07_instance_based.md)

