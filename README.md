# Cookiecutter Deep Learning

A modern, fast, and structured cookiecutter template for Deep Learning projects.

## Features

- **Frameworks:** Keras 3 (Multi-backend), PyTorch, or TensorFlow.
- **Configuration:** Hydra for robust hyperparameter management.
- **Experiment Tracking:** WandB or TensorBoard integration.
- **Package Management:** Native support for `uv` (lightning-fast) or standard `pip`.
- **Formatting:** `ruff` pre-configured for speed.

## Quickstart

To generate a new project, simply run:

```bash
# If using standard pip
pip install cookiecutter
cookiecutter https://github.com/your-username/cookiecutter-deep-learning

# If using uv
uvx cookiecutter https://github.com/your-username/cookiecutter-deep-learning
```

## Directory Structure

When you generate a project, it will look like this:

```
├── LICENSE
├── Makefile           <- Convenience commands
├── README.md
├── data
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── configs            <- Hydra configuration files
│   ├── model          <- Model configurations
│   ├── data           <- Data configurations
│   └── train.yaml     <- Main training configuration
│
├── checkpoints        <- Trained and serialized models or checkpoints
│
├── logs               <- Logs from experiment trackers like TensorBoard or WandB
│
├── notebooks          <- Jupyter notebooks for exploration
│
├── pyproject.toml     <- Project configuration file
│
├── train.py           <- Main training entry point
│
└── {{ module_name }}  <- Source code for use in this project.
    │
    ├── __init__.py
    ├── data           <- Data loaders and dataset definitions
    ├── models         <- Neural network architecture definitions
    └── utils          <- Utilities for metrics, logging, etc.
```
