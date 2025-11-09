# Chapter 10: Genetic Algorithms

## Overview & Motivation

Genetic Algorithms (GAs) are evolutionary computation methods inspired by natural selection and genetics. They maintain a population of candidate solutions and evolve them over generations using selection, crossover, and mutation operators. This chapter covers the foundations of GAs, their components, and applications in machine learning.

!!! note "When to Use Genetic Algorithms"
    GAs are particularly useful for:
    - **Discrete optimization**: Feature selection, hyperparameter tuning
    - **Non-differentiable problems**: Where gradients don't exist
    - **Multi-modal landscapes**: Multiple good solutions
    - **Black-box optimization**: When you can only evaluate, not differentiate
    - **Hybrid approaches**: Combine with local search for best results

!!! warning "GA Limitations"
    GAs are not always the best choice:
    - **Slower than gradient methods**: For smooth, differentiable problems
    - **No convergence guarantee**: May not find global optimum
    - **Parameter tuning**: Many hyperparameters to tune
    - **Computational cost**: Need many function evaluations
    
    **Use when**: Problem is discrete, non-differentiable, or gradient methods fail.

Genetic Algorithms are valuable because:
- Global optimization (can escape local optima)
- No gradient required (works with discrete/non-differentiable problems)
- Robust to noise
- Naturally parallelizable
- Can handle complex, multi-modal search spaces
- Useful for hyperparameter tuning and feature selection

## Core Theory & Intuitive Explanation

### Evolutionary Metaphor

**Intuition**: Mimic natural evolution:
1. **Population**: Set of candidate solutions (individuals)
2. **Fitness**: How good each solution is
3. **Selection**: Better solutions more likely to reproduce
4. **Crossover**: Combine parts of parent solutions
5. **Mutation**: Random changes to maintain diversity
6. **Evolution**: Repeat over generations

### Key Components

**Chromosome**: Representation of a solution (e.g., binary string, real-valued vector)

**Gene**: Individual component of a chromosome

**Fitness Function**: Evaluates solution quality (objective function)

**Selection**: Choose parents for reproduction

**Crossover**: Create offspring from parents

**Mutation**: Introduce random changes

## Mathematical Foundations

### GA Algorithm Structure

```
1. Initialize population P(0) of size N
2. Evaluate fitness of each individual
3. For generation t = 1 to T:
   a. Select parents from P(t-1)
   b. Create offspring via crossover
   c. Apply mutation to offspring
   d. Evaluate fitness of offspring
   e. Select survivors for P(t)
4. Return best individual
```

### Selection Operators

**Roulette Wheel Selection**:

Probability of selecting individual $i$:

$$P(i) = \frac{f_i}{\sum_{j=1}^N f_j}$$

where $f_i$ is the fitness of individual $i$.

**Tournament Selection**:

1. Randomly select $k$ individuals
2. Choose the best one

**Rank-Based Selection**:

Sort individuals by fitness, assign selection probability based on rank:

$$P(i) = \frac{2(N - r_i + 1)}{N(N + 1)}$$

where $r_i$ is the rank of individual $i$ (1 = best).

### Crossover Operators

**Single-Point Crossover**:

```
Parent 1: [1 1 0 0 1 0]
Parent 2: [0 0 1 1 0 1]
          └───┘
         Crossover point

Offspring 1: [1 1 0 1 0 1]
Offspring 2: [0 0 1 0 1 0]
```

**Uniform Crossover**:

Each gene comes from either parent with probability 0.5.

**Arithmetic Crossover** (for real-valued):

$$\text{Offspring} = \alpha \cdot \text{Parent}_1 + (1-\alpha) \cdot \text{Parent}_2$$

where $\alpha \in [0, 1]$ is a parameter.

### Mutation Operators

**Bit-Flip Mutation** (binary):

Flip each bit with probability $p_m$:

$$x_i' = \begin{cases} 1 - x_i & \text{with probability } p_m \\ x_i & \text{otherwise} \end{cases}$$

**Gaussian Mutation** (real-valued):

$$x_i' = x_i + \mathcal{N}(0, \sigma^2)$$

with probability $p_m$.

### Schema Theorem

**Schema**: Template describing a subset of strings (e.g., $1**0*$ matches $11000$, $10101$, etc.).

**Order** $o(H)$: Number of fixed positions in schema $H$.

**Defining Length** $\delta(H)$: Distance between first and last fixed positions.

**Schema Theorem** (Holland, 1975):

$$m(H, t+1) \geq m(H, t) \cdot \frac{f(H)}{f_{avg}} \cdot \left(1 - p_c \frac{\delta(H)}{L-1}\right)(1 - p_m)^{o(H)}$$

where:
- $m(H, t)$: Number of instances of schema $H$ at generation $t$
- $f(H)$: Average fitness of schema $H$
- $f_{avg}$: Average fitness of population
- $p_c$: Crossover probability
- $p_m$: Mutation probability
- $L$: Chromosome length

**Interpretation**: Above-average, short, low-order schemas increase exponentially.

### Convergence Analysis

**Selection Pressure**: Rate at which better solutions take over population.

**Diversity**: Measure of population variety.

**Trade-off**: High selection pressure → fast convergence but premature convergence.

**Mutation Rate**: Too low → loss of diversity, too high → random search.

## Visual/Graphical Illustrations

### GA Flow

```
Generation t
    │
    ▼
[Evaluate Fitness]
    │
    ▼
[Select Parents]
    │
    ▼
[Crossover] ────┐
    │            │
    ▼            │
[Mutation]    │
    │            │
    ▼            │
[Evaluate]     │
    │            │
    ▼            │
[Select Survivors] ←──┘
    │
    ▼
Generation t+1
```

### Fitness Landscape

```
Fitness
  ↑
  │     ●
  │    ╱ ╲
  │   ╱   ╲  ●
  │  ╱     ╲╱ ╲
  │ ╱           ╲
  │╱             ╲
  └──────────────────→ Solution Space
```

GAs can explore multiple peaks simultaneously.

## Worked Examples

### Example 1: Maximizing Function

**Problem**: Maximize $f(x) = x^2$ for $x \in [0, 31]$ (integer).

**Encoding**: 5-bit binary (e.g., $x = 13$ → $01101$)

**Fitness**: $f(x) = x^2$

**Example Run**:
- Generation 0: Random population
- Generation 1: After selection, crossover, mutation
- Continue until convergence

### Example 2: Feature Selection

**Problem**: Select best subset of features for classification.

**Encoding**: Binary string where 1 = feature selected, 0 = not selected

**Fitness**: Classification accuracy (or accuracy - $\lambda \cdot$ number of features)

## Code Implementation

### Genetic Algorithm from Scratch

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

class GeneticAlgorithm:
    def __init__(self, population_size=50, n_generations=100,
                 crossover_rate=0.8, mutation_rate=0.01,
                 selection_pressure=2, elitism=True):
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.selection_pressure = selection_pressure
        self.elitism = elitism
        self.fitness_history = []
    
    def _initialize_population(self, chromosome_length):
        """Initialize random population."""
        return np.random.randint(0, 2, size=(self.population_size, chromosome_length))
    
    def _evaluate_fitness(self, population, fitness_func):
        """Evaluate fitness of each individual."""
        return np.array([fitness_func(individual) for individual in population])
    
    def _tournament_selection(self, population, fitness, tournament_size=3):
        """Select parent using tournament selection."""
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = fitness[tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx].copy()
    
    def _crossover(self, parent1, parent2):
        """Single-point crossover."""
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        crossover_point = np.random.randint(1, len(parent1))
        offspring1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        offspring2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return offspring1, offspring2
    
    def _mutation(self, individual):
        """Bit-flip mutation."""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if np.random.random() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]
        return mutated
    
    def evolve(self, chromosome_length, fitness_func):
        """Run genetic algorithm."""
        # Initialize
        population = self._initialize_population(chromosome_length)
        fitness = self._evaluate_fitness(population, fitness_func)
        
        best_fitness_history = []
        avg_fitness_history = []
        
        for generation in range(self.n_generations):
            # Track statistics
            best_fitness = np.max(fitness)
            avg_fitness = np.mean(fitness)
            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)
            
            if generation % 10 == 0:
                print(f"Generation {generation}: Best = {best_fitness:.4f}, Avg = {avg_fitness:.4f}")
            
            # Create new population
            new_population = []
            
            # Elitism: keep best individual
            if self.elitism:
                best_idx = np.argmax(fitness)
                new_population.append(population[best_idx].copy())
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._tournament_selection(population, fitness)
                parent2 = self._tournament_selection(population, fitness)
                
                # Crossover
                offspring1, offspring2 = self._crossover(parent1, parent2)
                
                # Mutation
                offspring1 = self._mutation(offspring1)
                offspring2 = self._mutation(offspring2)
                
                new_population.extend([offspring1, offspring2])
            
            # Trim to population size
            new_population = new_population[:self.population_size]
            population = np.array(new_population)
            
            # Evaluate new population
            fitness = self._evaluate_fitness(population, fitness_func)
        
        # Return best individual
        best_idx = np.argmax(fitness)
        return population[best_idx], fitness[best_idx], best_fitness_history, avg_fitness_history

# Example 1: Maximize simple function
def simple_fitness(individual):
    """Fitness: sum of bits (maximize)."""
    return np.sum(individual)

ga = GeneticAlgorithm(population_size=50, n_generations=100,
                     crossover_rate=0.8, mutation_rate=0.01)

best_solution, best_fitness, best_history, avg_history = ga.evolve(
    chromosome_length=20, fitness_func=simple_fitness
)

print(f"\nBest solution: {best_solution}")
print(f"Best fitness: {best_fitness}")

# Plot evolution
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(best_history, label='Best Fitness')
plt.plot(avg_history, label='Average Fitness')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('GA Evolution: Simple Function')
plt.legend()
plt.grid(True, alpha=0.3)

# Example 2: Feature Selection
X, y = make_classification(n_samples=500, n_features=20, n_informative=10,
                          n_redundant=10, random_state=42)

def feature_selection_fitness(individual):
    """Fitness: classification accuracy with selected features."""
    selected_features = np.where(individual == 1)[0]
    if len(selected_features) == 0:
        return 0.0
    
    X_selected = X[:, selected_features]
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    scores = cross_val_score(clf, X_selected, y, cv=3, scoring='accuracy')
    
    # Penalize for number of features (optional)
    accuracy = np.mean(scores)
    penalty = 0.01 * len(selected_features)  # Small penalty
    return accuracy - penalty

ga_feature = GeneticAlgorithm(population_size=30, n_generations=50,
                             crossover_rate=0.8, mutation_rate=0.05)

best_features, best_fitness_feature, best_hist_feature, avg_hist_feature = ga_feature.evolve(
    chromosome_length=X.shape[1], fitness_func=feature_selection_fitness
)

print(f"\nSelected features: {np.where(best_features == 1)[0]}")
print(f"Number of features: {np.sum(best_features)}")
print(f"Best fitness (accuracy): {best_fitness_feature:.4f}")

plt.subplot(1, 2, 2)
plt.plot(best_hist_feature, label='Best Fitness')
plt.plot(avg_hist_feature, label='Average Fitness')
plt.xlabel('Generation')
plt.ylabel('Fitness (Accuracy)')
plt.title('GA Evolution: Feature Selection')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Real-Valued GA

```python
class RealValuedGA:
    """Genetic Algorithm for real-valued optimization."""
    def __init__(self, population_size=50, n_generations=100,
                 crossover_rate=0.8, mutation_rate=0.1, mutation_scale=0.1):
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
    
    def _initialize_population(self, bounds):
        """Initialize population within bounds."""
        n_vars = len(bounds)
        population = []
        for _ in range(self.population_size):
            individual = [np.random.uniform(bounds[i][0], bounds[i][1]) 
                         for i in range(n_vars)]
            population.append(individual)
        return np.array(population)
    
    def _arithmetic_crossover(self, parent1, parent2):
        """Arithmetic crossover for real values."""
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        alpha = np.random.random()
        offspring1 = alpha * parent1 + (1 - alpha) * parent2
        offspring2 = (1 - alpha) * parent1 + alpha * parent2
        return offspring1, offspring2
    
    def _gaussian_mutation(self, individual, bounds):
        """Gaussian mutation."""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if np.random.random() < self.mutation_rate:
                mutated[i] += np.random.normal(0, self.mutation_scale)
                # Clip to bounds
                mutated[i] = np.clip(mutated[i], bounds[i][0], bounds[i][1])
        return mutated
    
    def optimize(self, fitness_func, bounds):
        """Optimize function."""
        population = self._initialize_population(bounds)
        fitness = np.array([fitness_func(ind) for ind in population])
        
        best_history = []
        
        for generation in range(self.n_generations):
            best_idx = np.argmax(fitness)
            best_history.append(fitness[best_idx])
            
            # Selection and reproduction
            new_population = []
            for _ in range(self.population_size // 2):
                # Tournament selection
                idx1 = np.random.choice(len(population), 3)
                idx2 = np.random.choice(len(population), 3)
                parent1 = population[idx1[np.argmax(fitness[idx1])]]
                parent2 = population[idx2[np.argmax(fitness[idx2])]]
                
                # Crossover and mutation
                off1, off2 = self._arithmetic_crossover(parent1, parent2)
                off1 = self._gaussian_mutation(off1, bounds)
                off2 = self._gaussian_mutation(off2, bounds)
                
                new_population.extend([off1, off2])
            
            population = np.array(new_population[:self.population_size])
            fitness = np.array([fitness_func(ind) for ind in population])
        
        best_idx = np.argmax(fitness)
        return population[best_idx], fitness[best_idx], best_history

# Test on optimization problem
def sphere_function(x):
    """Sphere function: f(x) = sum(x_i^2), minimize."""
    return -np.sum(x**2)  # Negative for maximization

bounds = [(-5, 5), (-5, 5), (-5, 5)]  # 3D problem
rv_ga = RealValuedGA(population_size=50, n_generations=100)
best_solution, best_fitness, history = rv_ga.optimize(sphere_function, bounds)

print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(history)
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.title('Real-Valued GA: Sphere Function')
plt.grid(True, alpha=0.3)
plt.show()
```

## Conceptual Summary / Diagram

### GA Components

```
┌─────────────┐
│ Population  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Fitness    │
│ Evaluation  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Selection   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Crossover   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Mutation    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ New Population│
└──────────────┘
```

### Selection Methods Comparison

| Method | Characteristics | Use Case |
|--------|----------------|----------|
| **Roulette Wheel** | Proportional to fitness | When fitness range is reasonable |
| **Tournament** | Simple, parallelizable | General purpose |
| **Rank-Based** | Reduces selection pressure | When fitness scaling is needed |

## Common Misconceptions / Pitfalls

### Misconception 1: "GAs Always Find Global Optimum"

**Reality**: GAs are stochastic and may converge to local optima. No guarantee of global optimum.

### Misconception 2: "GAs Are Always Better Than Gradient Methods"

**Reality**: For smooth, differentiable problems, gradient methods are typically faster and more efficient.

### Misconception 3: "Mutation Rate Should Be Very Low"

**Reality**: Mutation rate depends on problem. Too low → loss of diversity, too high → random search.

### Pitfall: Premature Convergence

Population converges too quickly to suboptimal solution.

**Solutions**:
- Increase mutation rate
- Increase population diversity
- Use niching techniques
- Adjust selection pressure

### Pitfall: Poor Encoding

Bad chromosome representation makes problem harder.

**Guidelines**:
- Use natural representation
- Ensure all valid solutions are representable
- Make similar solutions have similar encodings

## Practice Exercises

### Exercise 1: Schema Theorem

Given a schema $H = 1**0*$ with:
- Average fitness $f(H) = 10$
- Population average $f_{avg} = 8$
- Crossover rate $p_c = 0.8$
- Mutation rate $p_m = 0.01$
- Chromosome length $L = 5$

Calculate the expected number of instances in the next generation if current count is 5.

### Exercise 2: TSP with GA

Implement a GA for the Traveling Salesman Problem:
- Encoding: Permutation of cities
- Fitness: Negative of tour length
- Crossover: Order crossover or PMX
- Mutation: Swap two cities

### Exercise 3: Hyperparameter Tuning

Use GA to tune hyperparameters of a machine learning model (e.g., learning rate, regularization).

### Exercise 4: Convergence Analysis

Study the effect of:
1. Population size on convergence
2. Mutation rate on diversity
3. Crossover rate on exploration vs exploitation

## Summary & Key Takeaways

### Key Concepts

1. **Population-Based Search**: Maintains multiple candidate solutions
2. **Selection**: Favors better solutions
3. **Crossover**: Combines good solutions
4. **Mutation**: Maintains diversity
5. **Schema Theorem**: Explains why GAs work

### Important Principles

- **Exploration vs Exploitation**: Balance search and convergence
- **Diversity**: Critical for avoiding premature convergence
- **Encoding**: Representation affects performance
- **Parameter Tuning**: Mutation and crossover rates are crucial

### Next Steps

Understanding genetic algorithms prepares you for:
- Other evolutionary algorithms (Evolution Strategies, Genetic Programming)
- Hybrid methods (GA + local search)
- Multi-objective optimization (NSGA-II)

## References / Further Reading

<details>
<summary><strong>📚 Primary References</strong></summary>

1. **Mitchell, M.** (1998). *An Introduction to Genetic Algorithms*. MIT Press.
   - [MIT Press](https://mitpress.mit.edu/9780262631853/an-introduction-to-genetic-algorithms/)
   - [Amazon](https://www.amazon.com/Introduction-Genetic-Algorithms-Complex-Adaptive/dp/0262631857)

2. **Goldberg, D. E.** (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*. Addison-Wesley.
   - [Amazon](https://www.amazon.com/Genetic-Algorithms-Search-Optimization-Machine/dp/0201157675)
   - [Pearson](https://www.pearson.com/en-us/subject-catalog/p/genetic-algorithms-in-search-optimization-and-machine-learning/P200000003431)

</details>

<details>
<summary><strong>🔬 Research Papers</strong></summary>

3. **Holland, J. H.** (1975). *Adaptation in Natural and Artificial Systems*. University of Michigan Press.
   - [Amazon](https://www.amazon.com/Adaptation-Natural-Artificial-Systems-Introductory/dp/0262581116)
   - [MIT Press (Reprint)](https://mitpress.mit.edu/9780262581110/adaptation-in-natural-and-artificial-systems/)

4. **Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T.** (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182-197.
   - [DOI: 10.1109/4235.996017](https://doi.org/10.1109/4235.996017)
   - [IEEE Xplore](https://ieeexplore.ieee.org/document/996017)

</details>

<details>
<summary><strong>📖 Applications</strong></summary>

5. **Whitley, D.** (1994). A genetic algorithm tutorial. *Statistics and Computing*, 4(2), 65-85.
   - [DOI: 10.1007/BF00175354](https://doi.org/10.1007/BF00175354)
   - [PDF](https://link.springer.com/article/10.1007/BF00175354)

</details>

---

## Recommended Reads

<details>
<summary><strong>📚 Official Documentation</strong></summary>

- **DEAP Framework** - [Distributed Evolutionary Algorithms](https://deap.readthedocs.io/)

- **PyGAD** - [Python Genetic Algorithm](https://pygad.readthedocs.io/)

- **Scikit-optimize** - [Bayesian optimization](https://scikit-optimize.github.io/stable/)

</details>

<details>
<summary><strong>📖 Essential Articles</strong></summary>

- **Genetic Algorithms Tutorial** - [Complete GA guide](https://towardsdatascience.com/introduction-to-genetic-algorithms-including-example-code-e396e98d8bf3)

- **GA Implementation** - [Implementing GAs from scratch](https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/)

- **Evolutionary Algorithms** - [Evolutionary computation overview](https://towardsdatascience.com/introduction-to-evolutionary-algorithms-a8594b484ac)

</details>

<details>
<summary><strong>🎓 Learning Resources</strong></summary>

- **Genetic Algorithms Course** - [GA fundamentals](https://www.coursera.org/learn/genetic-algorithms)

- **Evolutionary Computation** - [Evolutionary algorithms tutorial](https://www.tutorialspoint.com/genetic_algorithms/index.htm)

- **Optimization with GAs** - [GA for optimization](https://machinelearningmastery.com/genetic-algorithms-for-real-world-optimization-problems/)

</details>

<details>
<summary><strong>💡 Best Practices</strong></summary>

- **Parameter Tuning** - [GA hyperparameter selection](https://towardsdatascience.com/genetic-algorithm-parameters-5e2891c7a7c4)

- **Selection Strategies** - [Selection operator comparison](https://towardsdatascience.com/genetic-algorithm-selection-methods-8b0d0c0e5c1)

- **Convergence Criteria** - [Stopping conditions](https://towardsdatascience.com/genetic-algorithm-convergence-criteria-8b0d0c0e5c1)

</details>

<details>
<summary><strong>🔬 Research Papers</strong></summary>

- **Genetic Algorithms** - [Holland (1975)](https://mitpress.mit.edu/9780262581110/adaptation-in-natural-and-artificial-systems/) - Adaptation in natural and artificial systems

- **NSGA-II** - [Deb et al. (2002)](https://doi.org/10.1109/4235.996017) - Fast and elitist multiobjective genetic algorithm

- **GA Tutorial** - [Whitley (1994)](https://doi.org/10.1007/BF00175354) - Comprehensive GA tutorial

</details>

---

**Previous Chapter**: [Chapter 9: Neural Networks](09_neural_networks.md)

**Back to**: [Course Home](index.md)

