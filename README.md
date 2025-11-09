# Machine Learning Course - Postgraduate Level

A comprehensive, research-level Machine Learning course designed for M.Tech and PhD students. This course provides detailed mathematical foundations, intuitive explanations, and practical implementations of core ML algorithms.

## Course Overview

This course covers fundamental and advanced topics in Machine Learning, including:

- Introduction to Machine Learning
- Learning Paradigms (Supervised, Unsupervised, Reinforcement)
- Model Selection and Evaluation
- Bayesian Learning
- Linear Models (Regression and Classification)
- Decision Trees
- Instance-Based Learning (KNN, CBR)
- Support Vector Machines
- Neural Networks (Perceptron, Backpropagation)
- Genetic Algorithms

## Course Structure

The course is organized into 10 comprehensive chapters, each following a structured format:

1. **Overview & Motivation** - Why this topic matters
2. **Core Theory & Intuitive Explanation** - Conceptual understanding
3. **Mathematical Foundations** - Rigorous derivations and formulas
4. **Visual/Graphical Illustrations** - Diagrams and visualizations
5. **Worked Examples** - Step-by-step problem solving
6. **Code Implementation** - Python implementations from scratch
7. **Conceptual Summary** - Key concepts and relationships
8. **Common Misconceptions** - Pitfalls to avoid
9. **Practice Exercises** - Progressive difficulty problems
10. **Summary & Key Takeaways** - Review and connections
11. **References** - Further reading and research papers

## Getting Started

### Prerequisites

- Strong background in linear algebra, calculus, and probability theory
- Programming experience in Python
- Familiarity with basic statistics and optimization
- Understanding of data structures and algorithms

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/made_for_ml.git
cd made_for_ml
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Building the Documentation

This course uses [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) for documentation.

**To build and serve locally**:
```bash
mkdocs serve
```

Then open `http://127.0.0.1:8000` in your browser.

**To build static site**:
```bash
mkdocs build
```

The site will be generated in the `site/` directory.

**To build static site**:
```bash
mkdocs build
```

The site will be generated in the `site/` directory.

**Automatic Deployment**: The course is automatically deployed to GitHub Pages via GitHub Actions whenever you push to the `main` branch. No manual deployment needed!

**Manual deployment** (if needed):
```bash
mkdocs gh-deploy
```

## Course Content

### Chapters

1. **[Introduction to Machine Learning](docs/chapters/01_introduction.md)** - Foundations, terminology, and motivation
2. **[Learning Paradigms](docs/chapters/02_learning_paradigms.md)** - Supervised, unsupervised, and reinforcement learning
3. **[Model Selection and Evaluation](docs/chapters/03_model_selection.md)** - Bias-variance, cross-validation, metrics
4. **[Bayesian Learning](docs/chapters/04_bayesian_learning.md)** - MAP, MDL, Bayes optimal classifier, Naive Bayes
5. **[Linear Models](docs/chapters/05_linear_models.md)** - Regression, classification, regularization (Ridge, Lasso)
6. **[Decision Trees](docs/chapters/06_decision_trees.md)** - ID3, C4.5, CART, pruning
7. **[Instance-Based Learning](docs/chapters/07_instance_based.md)** - KNN, CBR, distance metrics
8. **[Support Vector Machines](docs/chapters/08_svm.md)** - Maximum margin, kernels, VC dimension
9. **[Neural Networks](docs/chapters/09_neural_networks.md)** - Perceptron, MLP, backpropagation
10. **[Genetic Algorithms](docs/chapters/10_genetic_algorithms.md)** - Evolutionary computation, selection, crossover, mutation

## Learning Outcomes

Upon completion of this course, students will be able to:

- **LO1**: Demonstrate a strong understanding of the foundations of Machine Learning algorithms
- **LO2**: Solve Machine Learning problems using appropriate learning techniques
- **LO3**: Evaluate machine learning solutions to problems
- **LO4**: Identify appropriate tools to implement solutions to machine learning problems

## Recommended Textbooks

- **T1**: Tom M. Mitchell, "Machine Learning," The McGraw-Hill Companies, Inc. Indian Edition, 1997
- **R1**: Christopher M. Bishop, "Pattern Recognition & Machine Learning," Springer, 2006
- **R2**: PANG-NING TAN, MICHAEL STEINBACH, VIPIN KUMAR, "Introduction To Data Mining," Pearson, 2nd Edition

See [References](docs/references.md) for complete bibliography.

## Features

- **Mathematical Rigor**: Every algorithm derived from first principles
- **Intuitive Explanations**: Complex concepts explained through analogies
- **Practical Implementation**: Code examples from scratch
- **Visual Learning**: Diagrams and illustrations throughout
- **Progressive Exercises**: From basic to advanced problems
- **Research-Level Content**: Suitable for postgraduate studies

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This course material is provided for educational purposes.

## Acknowledgments

This course is based on standard machine learning curricula and textbooks, particularly:
- Tom M. Mitchell's "Machine Learning"
- Christopher M. Bishop's "Pattern Recognition & Machine Learning"
- Various research papers and online resources

---

**Last Updated**: {{ git_revision_date_localized }}

For questions or issues, please open an issue on GitHub.
