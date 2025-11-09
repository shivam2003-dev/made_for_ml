# Chapter 9: Neural Networks

## Overview & Motivation

Neural networks are computational models inspired by biological neural systems. They consist of interconnected nodes (neurons) organized in layers, capable of learning complex non-linear patterns. This chapter covers perceptrons, multi-layer perceptrons, the backpropagation algorithm, and foundations of deep learning.

!!! important "The Neural Network Revolution"
    Neural networks have revolutionized ML:
    - **Universal Approximation**: Can approximate any continuous function
    - **Automatic Feature Learning**: No manual feature engineering needed
    - **Hierarchical Representations**: Learn features at multiple levels
    - **State-of-the-Art**: Dominant in vision, NLP, speech, games
    - **Foundation for Deep Learning**: CNNs, RNNs, Transformers all build on these concepts

!!! warning "When to Use Neural Networks"
    Neural networks are powerful but not always the best choice:
    - **Large datasets**: Need sufficient data to learn
    - **Computational resources**: Training can be expensive
    - **Interpretability**: Harder to explain than linear models or trees
    - **Overkill for simple problems**: Linear models may suffice
    
    **Rule**: Try simpler models first, use neural networks when you need their power.

Neural networks are powerful because:
- Universal function approximators (can approximate any continuous function)
- Learn hierarchical representations automatically
- Handle high-dimensional, complex data
- Foundation for modern deep learning
- State-of-the-art performance in many domains

## Core Theory & Intuitive Explanation

### Biological Inspiration

**Neuron**: Receives inputs, processes them, and produces an output.

**Artificial Neuron**: Mathematical model:
1. Weighted sum of inputs
2. Apply activation function
3. Output result

### Perceptron

**Intuition**: Simplest neural network - a single neuron that can learn linear decision boundaries.

**Learning**: Adjust weights to minimize classification errors.

### Multi-Layer Perceptron (MLP)

**Intuition**: Stack multiple layers of neurons to learn complex, non-linear functions.

**Architecture**:
- **Input Layer**: Receives features
- **Hidden Layers**: Process information
- **Output Layer**: Produces predictions

### Backpropagation

**Intuition**: Efficiently compute gradients by propagating errors backward through the network.

**Key Insight**: Chain rule of calculus allows us to compute gradients for all parameters in one forward and one backward pass.

## Mathematical Foundations

### Single Neuron (Perceptron)

**Input**: $x = (x_1, x_2, ..., x_d)$

**Weighted Sum**: $z = \sum_{i=1}^d w_i x_i + b = w^T x + b$

**Activation**: $a = \sigma(z)$

where $\sigma$ is the activation function.

**Common Activation Functions**:

1. **Sigmoid**: $\sigma(z) = \frac{1}{1 + e^{-z}}$

2. **Tanh**: $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$

3. **ReLU**: $\text{ReLU}(z) = \max(0, z)$

4. **Linear**: $\sigma(z) = z$

### Perceptron Learning Algorithm

**Update Rule** (for misclassified example $(x, y)$):

$$w_i := w_i + \alpha \cdot y \cdot x_i$$

$$b := b + \alpha \cdot y$$

where $\alpha$ is the learning rate.

**Convergence**: If data is linearly separable, perceptron converges in finite steps.

### Multi-Layer Network

**Forward Propagation**:

For layer $l$:

$$z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$$

$$a^{[l]} = \sigma^{[l]}(z^{[l]})$$

where:
- $W^{[l]}$: Weight matrix for layer $l$
- $b^{[l]}$: Bias vector
- $a^{[0]} = x$: Input

**Output**: $a^{[L]} = \hat{y}$ (final layer)

### Loss Function

**For Regression** (Mean Squared Error):

$$L = \frac{1}{2m} \sum_{i=1}^m (y_i - \hat{y}_i)^2$$

**For Classification** (Cross-Entropy):

$$L = -\frac{1}{m} \sum_{i=1}^m [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

### Backpropagation Algorithm

**Goal**: Compute $\frac{\partial L}{\partial W^{[l]}}$ and $\frac{\partial L}{\partial b^{[l]}}$ for all layers.

**Output Layer Error**:

For regression with MSE:
$$\delta^{[L]} = \frac{\partial L}{\partial a^{[L]}} \odot \sigma'^{[L]}(z^{[L]}) = (a^{[L]} - y) \odot \sigma'^{[L]}(z^{[L]})$$

For binary classification with cross-entropy and sigmoid:
$$\delta^{[L]} = a^{[L]} - y$$

**Backpropagate Error**:

For layer $l$:
$$\delta^{[l]} = (W^{[l+1]})^T \delta^{[l+1]} \odot \sigma'^{[l]}(z^{[l]})$$

**Gradients**:

$$\frac{\partial L}{\partial W^{[l]}} = \delta^{[l]} (a^{[l-1]})^T$$

$$\frac{\partial L}{\partial b^{[l]}} = \delta^{[l]}$$

**Update Rule** (Gradient Descent):

$$W^{[l]} := W^{[l]} - \alpha \frac{\partial L}{\partial W^{[l]}}$$

$$b^{[l]} := b^{[l]} - \alpha \frac{\partial L}{\partial b^{[l]}}$$

### Derivation of Backpropagation

**Chain Rule**: For composite function $f(g(x))$:

$$\frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$$

**For Neural Network**:

$$L = L(a^{[L]}) = L(\sigma^{[L]}(z^{[L]})) = L(\sigma^{[L]}(W^{[L]} a^{[L-1]} + b^{[L]}))$$

$$\frac{\partial L}{\partial W^{[L]}} = \frac{\partial L}{\partial a^{[L]}} \cdot \frac{\partial a^{[L]}}{\partial z^{[L]}} \cdot \frac{\partial z^{[L]}}{\partial W^{[L]}}$$

$$= (a^{[L]} - y) \cdot \sigma'^{[L]}(z^{[L]}) \cdot a^{[L-1]}$$

### Universal Approximation Theorem

**Theorem**: A feedforward network with a single hidden layer containing a finite number of neurons can approximate any continuous function on a compact subset of $\mathbb{R}^n$ to arbitrary accuracy, given appropriate activation functions and weights.

**Implication**: Neural networks are theoretically powerful, but:
- May require exponentially many neurons
- Doesn't guarantee learnability
- Doesn't specify architecture

## Visual/Graphical Illustrations

### Single Neuron

```
Inputs          Weights        Activation
x₁ ──w₁──┐
x₂ ──w₂──┤
x₃ ──w₃──┼── Σ ── σ ── Output
  ...    │
xₙ ──wₙ──┘
      +b
```

### Multi-Layer Network

```
Input Layer    Hidden Layer 1    Hidden Layer 2    Output Layer
    x₁            ○                 ○                 ○
    x₂      →     ○            →    ○            →   ○  (ŷ)
    x₃            ○                 ○
```

### Backpropagation Flow

```
Forward:  x → z¹ → a¹ → z² → a² → ... → ŷ
          ↑                           ↓
Backward: ←── δ² ←── δ¹ ←── δ⁰ ←── L
```

## Worked Examples

### Example 1: Perceptron Learning

**Data**: AND function
- (0,0) → 0
- (0,1) → 0
- (1,0) → 0
- (1,1) → 1

**Initial**: $w = [0, 0]$, $b = 0$, $\alpha = 0.1$

**Update 1**: Misclassify (1,1) as 0
- $w := [0,0] + 0.1 \cdot 1 \cdot [1,1] = [0.1, 0.1]$
- $b := 0 + 0.1 \cdot 1 = 0.1$

Continue until convergence.

### Example 2: XOR Problem

**Problem**: XOR is not linearly separable.

**Solution**: Need hidden layer (multi-layer perceptron).

**Architecture**: 2 inputs → 2 hidden neurons → 1 output

## Code Implementation

### Perceptron from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, learning_rate=0.1, n_iterations=100):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.errors = []
    
    def fit(self, X, y):
        """Train the perceptron."""
        n_samples, n_features = X.shape
        
        # Initialize weights
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Convert labels to {-1, +1}
        y_ = np.where(y == 0, -1, 1)
        
        for _ in range(self.n_iterations):
            errors = 0
            for idx, x_i in enumerate(X):
                # Prediction
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = np.where(linear_output >= 0, 1, -1)
                
                # Update if misclassified
                if y_pred != y_[idx]:
                    self.weights += self.learning_rate * y_[idx] * x_i
                    self.bias += self.learning_rate * y_[idx]
                    errors += 1
            
            self.errors.append(errors)
            if errors == 0:
                break
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)

# Test on AND function
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

perceptron = Perceptron(learning_rate=0.1, n_iterations=100)
perceptron.fit(X_and, y_and)

print("Perceptron predictions for AND:")
for x, y_true in zip(X_and, y_and):
    y_pred = perceptron.predict(x.reshape(1, -1))[0]
    print(f"Input: {x}, True: {y_true}, Predicted: {y_pred}")

# Visualize learning
plt.figure(figsize=(10, 6))
plt.plot(perceptron.errors)
plt.xlabel('Iteration')
plt.ylabel('Number of Errors')
plt.title('Perceptron Learning: Error vs Iteration')
plt.grid(True, alpha=0.3)
plt.show()
```

### Multi-Layer Perceptron with Backpropagation

```python
class MLP:
    def __init__(self, layers, learning_rate=0.01):
        self.layers = layers  # e.g., [2, 4, 1] for 2 inputs, 4 hidden, 1 output
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for i in range(len(self.layers) - 1):
            w = np.random.randn(self.layers[i+1], self.layers[i]) * np.sqrt(2.0 / self.layers[i])
            b = np.zeros((self.layers[i+1], 1))
            self.weights.append(w)
            self.biases.append(b)
    
    def _sigmoid(self, z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))  # Clip to avoid overflow
    
    def _sigmoid_derivative(self, z):
        """Derivative of sigmoid."""
        s = self._sigmoid(z)
        return s * (1 - s)
    
    def _forward(self, X):
        """Forward propagation."""
        activations = [X.T]
        z_values = []
        
        for i in range(len(self.weights)):
            z = self.weights[i] @ activations[-1] + self.biases[i]
            z_values.append(z)
            a = self._sigmoid(z)
            activations.append(a)
        
        return activations, z_values
    
    def _backward(self, activations, z_values, y):
        """Backpropagation."""
        m = y.shape[0]
        
        # Output layer error
        delta = activations[-1] - y.T
        
        # Store gradients
        dW = []
        db = []
        
        # Backpropagate
        for i in range(len(self.weights) - 1, -1, -1):
            dW_i = (1/m) * delta @ activations[i].T
            db_i = (1/m) * np.sum(delta, axis=1, keepdims=True)
            
            dW.insert(0, dW_i)
            db.insert(0, db_i)
            
            if i > 0:
                # Propagate error to previous layer
                delta = (self.weights[i].T @ delta) * self._sigmoid_derivative(z_values[i-1])
        
        return dW, db
    
    def fit(self, X, y, epochs=1000, verbose=True):
        """Train the network."""
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            activations, z_values = self._forward(X)
            
            # Compute loss
            loss = (1/(2*len(y))) * np.sum((activations[-1].T - y)**2)
            losses.append(loss)
            
            # Backward pass
            dW, db = self._backward(activations, z_values, y)
            
            # Update weights
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * dW[i]
                self.biases[i] -= self.learning_rate * db[i]
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return losses
    
    def predict(self, X):
        """Make predictions."""
        activations, _ = self._forward(X)
        return activations[-1].T

# Test on XOR problem (requires hidden layer)
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_xor = np.array([[0], [1], [1], [0]], dtype=float)

# Create MLP: 2 inputs, 4 hidden neurons, 1 output
mlp = MLP(layers=[2, 4, 1], learning_rate=0.5)
losses = mlp.fit(X_xor, y_xor, epochs=2000, verbose=True)

# Test
print("\nXOR Predictions:")
for x, y_true in zip(X_xor, y_xor):
    y_pred = mlp.predict(x.reshape(1, -1))[0, 0]
    print(f"Input: {x}, True: {y_true[0]}, Predicted: {y_pred:.3f}")

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('MLP Learning Curve (XOR Problem)')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.show()
```

### Using Keras/TensorFlow

```python
try:
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    
    # Create MLP using Keras
    model = Sequential([
        Dense(4, activation='sigmoid', input_shape=(2,)),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train
    history = model.fit(X_xor, y_xor, epochs=1000, verbose=0)
    
    # Evaluate
    predictions = model.predict(X_xor)
    print("\nKeras MLP Predictions:")
    for x, y_true, pred in zip(X_xor, y_xor, predictions):
        print(f"Input: {x}, True: {y_true[0]}, Predicted: {pred[0]:.3f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.show()
except ImportError:
    print("TensorFlow/Keras not installed. Install with: pip install tensorflow")
```

## Conceptual Summary / Diagram

### Neural Network Architecture

```
Input → Hidden Layer 1 → Hidden Layer 2 → Output
  x₁        ○                ○               ŷ
  x₂   →    ○           →    ○
  x₃        ○                ○
```

### Learning Process

```
1. Forward Pass: Compute predictions
2. Compute Loss: Compare predictions to targets
3. Backward Pass: Compute gradients
4. Update Weights: Gradient descent
5. Repeat
```

### Activation Functions Comparison

| Function | Range | Derivative | Use Case |
|----------|-------|------------|----------|
| **Sigmoid** | (0,1) | $\sigma(1-\sigma)$ | Output layer (binary) |
| **Tanh** | (-1,1) | $1-\tanh^2$ | Hidden layers |
| **ReLU** | $[0,\infty)$ | 0 or 1 | Hidden layers (modern) |
| **Linear** | $(-\infty,\infty)$ | 1 | Regression output |

## Common Misconceptions / Pitfalls

### Misconception 1: "More Layers Always Better"

**Reality**: Deeper networks can overfit and are harder to train. Need sufficient data and proper regularization.

### Misconception 2: "Backpropagation Is the Learning Algorithm"

**Reality**: Backpropagation computes gradients. The learning algorithm is gradient descent (or variants).

### Misconception 3: "Neural Networks Always Outperform Other Methods"

**Reality**: Neural networks excel on large, complex data but may be overkill for simple problems. Consider:
- Data size
- Problem complexity
- Interpretability needs
- Computational resources

### Pitfall: Vanishing/Exploding Gradients

In deep networks, gradients can vanish (→0) or explode (→∞) during backpropagation.

**Solutions**:
- Proper weight initialization (Xavier, He)
- Batch normalization
- Residual connections
- Gradient clipping

### Pitfall: Overfitting

Neural networks can memorize training data.

**Solutions**:
- Dropout
- L2 regularization
- Early stopping
- Data augmentation

## Practice Exercises

### Exercise 1: Perceptron Convergence

Prove that if data is linearly separable, the perceptron algorithm converges in finite steps.

### Exercise 2: Backpropagation Derivation

Derive the backpropagation equations for a 3-layer network (input, hidden, output) with sigmoid activations and MSE loss.

### Exercise 3: XOR Implementation

Implement a 2-layer network (2 inputs, 2 hidden, 1 output) that learns the XOR function. Visualize the decision boundary.

### Exercise 4: Activation Functions

Compare sigmoid, tanh, and ReLU activations:
1. Plot their functions and derivatives
2. Train networks with each on the same dataset
3. Compare convergence speed and final accuracy

## Summary & Key Takeaways

### Key Concepts

1. **Perceptron**: Single neuron, learns linear boundaries
2. **MLP**: Multiple layers, learns non-linear functions
3. **Backpropagation**: Efficient gradient computation
4. **Activation Functions**: Introduce non-linearity
5. **Universal Approximation**: Theoretical power of neural networks

### Important Principles

- **Non-linearity**: Essential for learning complex patterns
- **Gradient-Based Learning**: Backpropagation enables training
- **Architecture Matters**: Network structure affects learning
- **Regularization**: Critical for generalization

### Next Steps

Understanding neural networks prepares you for:
- Deep learning architectures (CNNs, RNNs)
- Modern optimization (Adam, etc.)
- Advanced techniques (dropout, batch norm)

## References / Further Reading

??? note "📚 Primary References"

    1. **Mitchell, T. M.** (1997). *Machine Learning*.
       - Chapter 4: Artificial Neural Networks

    2. **Bishop, C. M.** (2006). *Pattern Recognition & Machine Learning*.
       - Chapter 5: Neural Networks

??? note "📖 Deep Learning"

    3. **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). *Deep Learning*. MIT Press.
       - [MIT Press](https://mitpress.mit.edu/9780262035613/deep-learning/)
       - [Free Online Book](https://www.deeplearningbook.org/)
       - [Amazon](https://www.amazon.com/Deep-Learning-Adaptive-Computation-Machine/dp/0262035618)

    4. **Nielsen, M.** (2015). *Neural Networks and Deep Learning*. Determination Press.
       - [Free Online Book](http://neuralnetworksanddeeplearning.com/)
       - [GitHub](https://github.com/mnielsen/neural-networks-and-deep-learning)

??? note "🔬 Research Papers"

    5. **Rumelhart, D. E., Hinton, G. E., & Williams, R. J.** (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533-536.
       - [DOI: 10.1038/323533a0](https://doi.org/10.1038/323533a0)
       - [Nature](https://www.nature.com/articles/323533a0)
       - [PDF](https://www.nature.com/articles/323533a0.pdf)

    6. **Hornik, K., Stinchcombe, M., & White, H.** (1989). Multilayer feedforward networks are universal approximators. *Neural Networks*, 2(5), 359-366.
       - [DOI: 10.1016/0893-6080(89)90020-8](https://doi.org/10.1016/0893-6080(89)90020-8)
       - [PDF](https://www.sciencedirect.com/science/article/pii/pii/0893608089900208)

    ---

## Recommended Reads

??? note "📚 Official Documentation"

    - **Neural Networks** - [PyTorch neural networks](https://pytorch.org/docs/stable/nn.html)

    - **TensorFlow Keras** - [Keras API reference](https://www.tensorflow.org/api_docs/python/tf/keras)

    - **Activation Functions** - [Activation functions guide](https://pytorch.org/docs/stable/nn.html#activation-functions)

??? note "📖 Essential Articles"

    - **Neural Networks Tutorial** - [Complete NN guide](https://machinelearningmastery.com/neural-networks-crash-course/)

    - **Backpropagation Explained** - [Understanding backpropagation](https://towardsdatascience.com/understanding-backpropagation-algorithm-7bb3aa2f95fd)

    - **Activation Functions** - [Choosing activation functions](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)

??? note "🎓 Learning Resources"

    - **Neural Networks Course** - [Andrew Ng's Deep Learning](https://www.coursera.org/specializations/deep-learning)

    - **Backpropagation Tutorial** - [Step-by-step backpropagation](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)

    - **Neural Network Playground** - [Interactive NN visualization](https://playground.tensorflow.org/)

??? note "💡 Best Practices"

    - **Weight Initialization** - [Initialization strategies](https://pytorch.org/docs/stable/nn.init.html)

    - **Gradient Clipping** - [Preventing gradient explosion](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)

    - **Learning Rate Scheduling** - [LR scheduling strategies](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)

??? note "🔬 Research Papers"

    - **Backpropagation** - [Rumelhart et al. (1986)](https://doi.org/10.1038/323533a0) - Learning representations by back-propagating errors

    - **Universal Approximation** - [Hornik et al. (1989)](https://doi.org/10.1016/0893-6080(89)90020-8) - Multilayer networks as universal approximators
    - **Deep Learning Book** - [Goodfellow et al. (2016)](https://www.deeplearningbook.org/) - Comprehensive deep learning reference

    ---

    **Previous Chapter**: [Chapter 8: Support Vector Machines](08_svm.md) | **Next Chapter**: [Chapter 10: Genetic Algorithms](10_genetic_algorithms.md)

