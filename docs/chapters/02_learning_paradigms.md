# Chapter 2: Learning Paradigms

## Overview & Motivation

Machine learning algorithms can be categorized into different paradigms based on the nature of the learning signal available. Understanding these paradigms is crucial for selecting the appropriate approach for a given problem. This chapter explores supervised, unsupervised, and reinforcement learning, along with their mathematical foundations and applications.

!!! important "Why Learning Paradigms Matter"
    The choice of learning paradigm is one of the first and most critical decisions in any ML project:
    - **Determines data requirements**: Do you need labels? Can you interact with the environment?
    - **Shapes algorithm selection**: Different paradigms use fundamentally different approaches
    - **Affects evaluation**: How do you measure success without labels or with delayed rewards?
    - **Impacts project timeline**: Labeling data can be expensive and time-consuming

The choice of learning paradigm fundamentally affects:
- What data is required
- How the learning algorithm works
- What types of problems can be solved
- How performance is evaluated

## Core Theory & Intuitive Explanation

### Supervised Learning

**Intuition**: Learning with a "teacher" - we have examples with correct answers.

The learner receives input-output pairs $(x_i, y_i)$ and learns to predict $y$ for new inputs $x$.

!!! tip "The Teacher Analogy"
    Think of supervised learning like a student learning from a teacher:
    - **Teacher provides examples**: "This is a cat" (labeled image)
    - **Student learns patterns**: Recognizes features that distinguish cats
    - **Student practices**: Makes predictions on new examples
    - **Teacher evaluates**: Checks if predictions are correct
    
    The "supervision" comes from having the correct answers (labels) during training.

**Key Characteristics**:
- Training data includes labels (ground truth)
- Goal: Learn mapping $f: X \rightarrow Y$
- Evaluation: Compare predictions to true labels
- Most common and well-studied paradigm

**Types**:
- **Classification**: $Y$ is discrete (e.g., spam/not spam, image categories)
  - Binary: Two classes (spam/not spam)
  - Multi-class: Multiple classes (cat/dog/bird)
  - Multi-label: Multiple labels per example (tags)
  
- **Regression**: $Y$ is continuous (e.g., house prices, temperature)
  - Predict real-valued outputs
  - Often involves minimizing squared or absolute error

### Unsupervised Learning

**Intuition**: Learning without a "teacher" - we only have inputs, no labels.

The learner discovers patterns, structure, or representations in the data.

!!! note "Why Unsupervised Learning?"
    Unsupervised learning is valuable because:
    - **Labels are expensive**: Annotating data requires human experts and time
    - **Discover hidden patterns**: Find structure you didn't know existed
    - **Data exploration**: Understand your data before building models
    - **Feature learning**: Learn useful representations automatically
    - **Most data is unlabeled**: In many domains, labeled data is rare

**Key Characteristics**:
- Training data: Only inputs $\{x_i\}$ (no labels)
- Goal: Discover hidden structure
- Evaluation: More subjective (coherence, interpretability)
- Often used for exploratory analysis

**Types**:
- **Clustering**: Group similar examples
  - Partition data into groups (clusters)
  - Examples: Customer segmentation, image segmentation
  
- **Dimensionality Reduction**: Find lower-dimensional representations
  - Reduce number of features while preserving information
  - Examples: PCA, t-SNE, autoencoders
  
- **Density Estimation**: Model the data distribution
  - Learn $p(x)$ from samples
  - Examples: Anomaly detection, generative models

### Reinforcement Learning

**Intuition**: Learning through interaction and feedback - like training a pet with rewards.

The learner (agent) takes actions in an environment and receives rewards/penalties.

!!! example "RL in Action"
    Consider training a robot to walk:
    - **State**: Current position, joint angles, balance
    - **Action**: Move leg forward, adjust balance
    - **Reward**: +1 for forward progress, -10 for falling
    - **Learning**: Agent learns which actions lead to rewards
    
    The agent explores different actions, learns from consequences, and improves over time.

**Key Characteristics**:
- Training: Agent-environment interaction
- Goal: Maximize cumulative reward
- Evaluation: Total reward over episodes
- Sequential decision making
- Delayed rewards (credit assignment problem)

!!! important "Key RL Challenges"
    1. **Exploration vs Exploitation**: Try new actions or stick with known good ones?
    2. **Credit Assignment**: Which actions led to the reward?
    3. **Delayed Rewards**: Rewards may come many steps later
    4. **Non-stationary**: Environment may change over time

## Mathematical Foundations

### Supervised Learning Formalism

Given training data $D = \{(x_i, y_i)\}_{i=1}^m$ where:
- $x_i \in \mathcal{X}$ (input space)
- $y_i \in \mathcal{Y}$ (output space)

Learn $h: \mathcal{X} \rightarrow \mathcal{Y}$ to minimize:

$$L(h) = \frac{1}{m} \sum_{i=1}^m \ell(h(x_i), y_i) + \lambda \Omega(h)$$

where:
- $\ell$: loss function
- $\Omega$: regularization term
- $\lambda$: regularization parameter

**Classification**: $\mathcal{Y} = \{1, 2, ..., k\}$ (discrete)

**Regression**: $\mathcal{Y} = \mathbb{R}$ (continuous)

### Unsupervised Learning Formalism

Given data $D = \{x_i\}_{i=1}^m$ (no labels), learn:

1. **Clustering**: Partition into $k$ clusters
   - Find $C_1, C_2, ..., C_k$ such that $\bigcup_i C_i = D$ and $C_i \cap C_j = \emptyset$

2. **Dimensionality Reduction**: Find $f: \mathbb{R}^d \rightarrow \mathbb{R}^k$ where $k < d$
   - Preserve important information while reducing dimensions

3. **Density Estimation**: Estimate $p(x)$ from samples
   - Maximum Likelihood: $\hat{p} = \arg\max_p \prod_{i=1}^m p(x_i)$

### Reinforcement Learning Formalism

**Markov Decision Process (MDP)**:
- States: $s \in \mathcal{S}$
- Actions: $a \in \mathcal{A}$
- Transition: $P(s'|s, a)$
- Reward: $R(s, a, s')$

**Objective**: Find policy $\pi: \mathcal{S} \rightarrow \mathcal{A}$ maximizing:

$$V^\pi(s) = \mathbb{E}\left[\sum_{t=0}^\infty \gamma^t R(s_t, a_t, s_{t+1}) \middle| s_0 = s, a_t \sim \pi(s_t)\right]$$

where $\gamma \in [0, 1]$ is the discount factor.

## Visual/Graphical Illustrations

### Supervised Learning Flow

```
Training Data (x, y)
        ↓
   Learning Algorithm
        ↓
   Learned Model h
        ↓
   New Input x'
        ↓
   Prediction h(x')
```

### Unsupervised Learning: Clustering Example

```
Before Clustering:          After Clustering:
    •  •                        ○  ○
  •      •                    ○      ○
    •  •                        ○  ○
    
  •      •                    ●      ●
    •  •                        ●  ●
```

### Reinforcement Learning Loop

```
┌─────────┐
│  Agent  │──action──>┌─────────────┐
│         │<--reward--│ Environment │
└─────────┘           └─────────────┘
     ↑                      │
     └───────state──────────┘
```

## Worked Examples

### Example 1: Supervised Learning - Email Classification

**Problem**: Classify emails as spam or legitimate.

**Data**: 
- Features: Word frequencies, sender info, email structure
- Labels: {spam, legitimate}

**Algorithm**: Naive Bayes, Logistic Regression, or SVM

**Evaluation**: Accuracy, precision, recall on test set

### Example 2: Unsupervised Learning - Customer Segmentation

**Problem**: Group customers by purchasing behavior.

**Data**: 
- Features: Purchase history, demographics, browsing behavior
- Labels: None

**Algorithm**: K-means clustering, hierarchical clustering

**Evaluation**: Cluster coherence, business interpretability

### Example 3: Reinforcement Learning - Game Playing

**Problem**: Learn to play chess.

**Data**: 
- States: Board positions
- Actions: Legal moves
- Rewards: Win (+1), Loss (-1), Draw (0)

**Algorithm**: Q-learning, Policy Gradient

**Evaluation**: Win rate against opponents

## Code Implementation

### Supervised Learning: Classification

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Generate synthetic classification data
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=15,
    n_redundant=5, n_classes=2, random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train supervised model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred))
```

### Unsupervised Learning: Clustering

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Generate synthetic data with clusters
X, y_true = make_blobs(
    n_samples=300, centers=4, n_features=2,
    random_state=42, cluster_std=0.60
)

# Apply K-means clustering (unsupervised)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X)

# Visualize
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis')
plt.title('True Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], 
            kmeans.cluster_centers_[:, 1],
            c='red', marker='x', s=200, linewidths=3)
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

print(f"Inertia (within-cluster sum of squares): {kmeans.inertia_:.2f}")
```

### Reinforcement Learning: Simple Q-Learning

```python
import numpy as np

class QLearning:
    def __init__(self, n_states, n_actions, learning_rate=0.1, 
                 discount=0.95, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, n_actions))
    
    def choose_action(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])
    
    def update(self, state, action, reward, next_state):
        current_q = self.Q[state, action]
        max_next_q = np.max(self.Q[next_state])
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.Q[state, action] = new_q

# Example: Simple grid world
# States: 0-3 (4 positions)
# Actions: 0=left, 1=right
# Goal: Reach state 3 (reward +10)

agent = QLearning(n_states=4, n_actions=2)

# Training
for episode in range(1000):
    state = 0  # Start state
    while state != 3:  # Goal state
        action = agent.choose_action(state)
        if action == 0:  # Left
            next_state = max(0, state - 1)
        else:  # Right
            next_state = min(3, state + 1)
        
        reward = 10 if next_state == 3 else -0.1
        agent.update(state, action, reward, next_state)
        state = next_state

print("Learned Q-values:")
print(agent.Q)
```

## Conceptual Summary / Diagram

### Learning Paradigm Comparison

| Aspect | Supervised | Unsupervised | Reinforcement |
|--------|-----------|--------------|---------------|
| **Data** | $(x, y)$ pairs | $x$ only | $(s, a, r, s')$ tuples |
| **Goal** | Predict $y$ from $x$ | Discover structure | Maximize reward |
| **Evaluation** | Compare to labels | Subjective/coherence | Cumulative reward |
| **Examples** | Classification, Regression | Clustering, PCA | Game playing, Robotics |

### When to Use Each Paradigm

**Supervised Learning**:
- ✅ You have labeled data
- ✅ You want to predict specific outputs
- ✅ You can measure performance objectively

**Unsupervised Learning**:
- ✅ You have unlabeled data
- ✅ You want to discover patterns
- ✅ Exploratory data analysis

**Reinforcement Learning**:
- ✅ Sequential decision making
- ✅ Delayed rewards
- ✅ Interactive learning environment

## Common Misconceptions / Pitfalls

### Misconception 1: "Unsupervised Learning is Easier"

**Reality**: Unsupervised learning is often harder because:
- No clear objective function
- Evaluation is subjective
- Requires domain expertise to interpret results

!!! warning "The Unsupervised Learning Challenge"
    Many students assume unsupervised learning is easier because "no labels needed." However:
    - **No ground truth**: How do you know if clustering is "correct"?
    - **Multiple valid solutions**: Different clusterings may be equally valid
    - **Interpretation required**: You need domain knowledge to make sense of results
    - **Parameter tuning**: Choosing number of clusters, distance metrics, etc. is non-trivial

### Misconception 2: "All Problems Need Labels"

**Reality**: Many problems can be reformulated:
- Semi-supervised learning uses both labeled and unlabeled data
- Self-supervised learning creates labels from data structure
- Transfer learning uses pre-trained models

### Misconception 3: "Reinforcement Learning Only for Games"

**Reality**: RL applies to:
- Robotics and control
- Recommendation systems
- Resource allocation
- Autonomous systems

### Pitfall: Confusing Problem Types

Using classification algorithms for regression problems (or vice versa) leads to poor performance. Always match the algorithm to the problem type.

!!! danger "Common Mistake"
    **Don't do this**:
    - Using logistic regression for continuous outputs (it outputs probabilities, not continuous values)
    - Using regression for classification (rounding predictions doesn't work well)
    - Treating ordinal data as nominal (loses ordering information)
    
    **Always**: Check your output space $Y$ first, then choose appropriate algorithms.

## Practice Exercises

### Exercise 1: Paradigm Identification

For each problem, identify the appropriate learning paradigm:

1. Predicting stock prices from historical data
2. Grouping news articles by topic
3. Training a robot to navigate a maze
4. Detecting anomalies in network traffic
5. Recommending movies to users

### Exercise 2: Mathematical Derivation

For supervised learning with squared loss, show that the optimal hypothesis for regression is:
$$h^*(x) = \mathbb{E}[Y | X = x]$$

### Exercise 3: Implementation

Implement K-means clustering from scratch. Test it on the `make_blobs` dataset and compare with sklearn's implementation.

### Exercise 4: Analysis

Consider a reinforcement learning problem where:
- States: Grid positions (5×5 grid)
- Actions: Up, Down, Left, Right
- Reward: +100 at goal, -1 per step

Design a reward structure that encourages:
1. Reaching the goal quickly
2. Avoiding obstacles
3. Exploring the environment

## Summary & Key Takeaways

### Key Concepts

1. **Supervised Learning**: Learning with labeled examples
   - Classification: Discrete outputs
   - Regression: Continuous outputs

2. **Unsupervised Learning**: Discovering structure without labels
   - Clustering, dimensionality reduction, density estimation

3. **Reinforcement Learning**: Learning through interaction and rewards
   - Sequential decision making in dynamic environments

### Important Principles

- **Problem-appropriate paradigm**: Match the learning paradigm to your problem
- **Data availability**: Determines which paradigms are feasible
- **Evaluation methods**: Differ across paradigms

### Next Steps

Understanding learning paradigms prepares you for:
- Model selection strategies (Chapter 3)
- Specific algorithms within each paradigm (Chapters 4-10)

## References / Further Reading

??? note "📚 Primary References"

    1. **Mitchell, T. M.** (1997). *Machine Learning*. 
       - Chapter 1: Introduction (Supervised Learning)
       - Chapter 12: Learning from Observations (Unsupervised Learning)

    2. **Bishop, C. M.** (2006). *Pattern Recognition & Machine Learning*.
       - Chapter 1: Introduction
       - Chapter 9: Mixture Models and EM (Unsupervised)

??? note "📖 Reinforcement Learning"

    3. **Sutton, R. S., & Barto, A. G.** (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
       - [MIT Press](https://mitpress.mit.edu/9780262039246/reinforcement-learning-second-edition/)
       - [Free Online Book](http://incompleteideas.net/book/the-book-2nd.html)
       - [Amazon](https://www.amazon.com/Reinforcement-Learning-Introduction-Adaptive-Computation/dp/0262039249)

    4. **Szepesvári, C.** (2010). *Algorithms for Reinforcement Learning*. Morgan & Claypool.
       - [Morgan & Claypool](https://www.morganclaypool.com/doi/abs/10.2200/S00268ED1V01Y201005AIM009)
       - [Free PDF](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf)
       - [Amazon](https://www.amazon.com/Algorithms-Reinforcement-Learning-Synthesis-Artificial/dp/1608454924)

??? note "🔬 Research Papers"

    5. **Jordan, M. I., & Mitchell, T. M.** (2015). Machine learning: Trends, perspectives, and prospects. *Science*, 349(6245), 255-260.
       - [DOI: 10.1126/science.aaa8415](https://doi.org/10.1126/science.aaa8415)
       - [Science](https://www.science.org/doi/10.1126/science.aaa8415)
       - [PDF](https://www.science.org/doi/pdf/10.1126/science.aaa8415)

    ---

## Recommended Reads

??? note "📚 Official Documentation"

    - **Scikit-learn Supervised Learning** - [Classification and regression](https://scikit-learn.org/stable/supervised_learning.html)

    - **Scikit-learn Unsupervised Learning** - [Clustering and dimensionality reduction](https://scikit-learn.org/stable/unsupervised_learning.html)

    - **OpenAI Gym** - [Reinforcement learning environments](https://www.gymlibrary.dev/)

??? note "📖 Essential Articles"

    - **Supervised vs Unsupervised** - [Towards Data Science guide](https://towardsdatascience.com/supervised-vs-unsupervised-learning-14f68e32ea8d)

    - **Reinforcement Learning Tutorial** - [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/)

    - **Semi-supervised Learning** - [Introduction to semi-supervised learning](https://scikit-learn.org/stable/modules/semi_supervised.html)

??? note "🎓 Learning Resources"

    - **RL Course** - [David Silver's RL Course](https://www.davidsilver.uk/teaching/)

    - **Unsupervised Learning** - [Coursera Unsupervised Learning](https://www.coursera.org/learn/unsupervised-learning)

    - **Clustering Tutorial** - [Scikit-learn clustering tutorial](https://scikit-learn.org/stable/modules/clustering.html)

??? note "💡 Best Practices"

    - **Choosing Learning Paradigm** - [When to use supervised/unsupervised/RL](https://machinelearningmastery.com/what-is-machine-learning/)

    - **Data Labeling** - [Labeling strategies](https://www.cloudfactory.com/data-labeling-guide)

    - **RL Best Practices** - [OpenAI RL best practices](https://spinningup.openai.com/en/latest/user/algorithms.html)

??? note "🔬 Research Papers"

    - **ML Trends** - [Jordan & Mitchell (2015)](https://doi.org/10.1126/science.aaa8415) - Machine learning perspectives

    - **RL Introduction** - [Sutton & Barto (2018)](http://incompleteideas.net/book/the-book-2nd.html) - Reinforcement learning textbook

    ---

    **Previous Chapter**: [Chapter 1: Introduction](01_introduction.md) | **Next Chapter**: [Chapter 3: Model Selection and Evaluation](03_model_selection.md)

