# Parallelising Matrix-Products in Isotropic Multilayer Perceptrons

**George Bird, 2026**

Code accompanying the paper *Parallelising Matrix-Products in Isotropic Multilayer Perceptrons*, presented at the **18th Annual Workshop on Optimization for Machine Learning (OPT2026)**.

## Overview

Isotropic activation functions permit a factorisation of multilayer perceptrons (MLPs) in which matrix multiplications can be precomputed and parallelised, while nonlinear scalar operations retain their sequential dependence.

This repository implements and compares four MLP formulations:

1. **Standard MLP:** Conventional sequential network using elementwise tanh activations.
2. **Ordinary isotropic MLP:** Sequential network using isotropic tanh activations.
3. **Factorised isotropic MLP:** Algebraically equivalent formulation with precomputable weight chains (Eq. 12).
4. **Reparameterised isotropic MLP:** Generalised architecture with independently trainable weight and bias parameters (Sec. 3.1).

The experiments examine both classification performance and computational efficiency.

## Experiments

Three Jupyter notebooks correspond to the empirical results reported in the paper.

| Notebook                                       | Description                                                                                                      | Paper correspondence            |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| **Experiment 1 — ModelAccuracyAcrossDatasets** | Compares classification performance across MNIST, CIFAR-10 and Caltech256.                                       | Figure 1                        |
| **Experiment 2 — CIFARWidthTimings**           | Measures inference and training times with a fixed input dimension of 3072 and varying hidden widths and depths. | Appendix Tables 4–5             |
| **Experiment 3 — ConstantWidthTimings**        | Measures inference and training times with equal input and hidden widths across varying network sizes.           | Table 1 and Appendix Tables 2–3 |

Each notebook contains the experimental configuration, model selection, evaluation procedure and corresponding result-generation code.

## Implementation

The `Dependency/` directory contains the shared PyTorch implementations:

* **`Models.py`** — Standard, factorised and reparameterised MLP architectures, including alternative implementations developed during optimisation.
* **`RewrittenLayers.py`** — Isotropic activation functions.
* **`TrainingModules.py`** — Training and evaluation utilities.
* **`GPU.py`** — Device selection and GPU information.

The factorised and reparameterised implementations differ in their treatment of weight-chain contractions and bias contributions. Several implementation variants are retained for comparison.

## Requirements

The experiments use Python, PyTorch, torchvision, NumPy, pandas, Matplotlib and Jupyter.

The timing experiments additionally use `torch.compile` and were originally evaluated on an NVIDIA RTX 3080 (10 GB). Timings may vary across hardware and software configurations.

## Reproduction

Run the relevant Jupyter notebook from the repository root.

Experiment 1 downloads the image-classification datasets and uses the supplied per-pixel normalisation statistics. Experiments 2 and 3 use randomly generated inputs and labels, so no dataset downloads are required.

The experiments use fixed random seeds and report results across multiple repeats.

**Note:** The supplied notebooks contain cleaned implementations of the experiments. Numerical results may differ slightly from those reported in the paper due to implementation revisions, compilation behaviour and hardware differences.

## Citation

If you use this code, please cite:

George Bird (2026). *Parallelising Matrix-Products in Isotropic Multilayer Perceptrons*. 18th Annual Workshop on Optimization for Machine Learning (OPT2026).

@inproceedings{
  bird2026parallelising,
  title={Parallelising Matrix-Products in Isotropic Multilayer Perceptrons},
  author={George Bird},
  booktitle={OPT 2026: Optimization for Machine Learning (NeurIPS 2026 Workshop)},
  year={2026},
  url={https://openreview.net/forum?id=z3N0ccKOi0}
}


# Model Implementation Variants

Several implementations of the factorised and reparameterised isotropic MLPs are included, reflecting different approaches to weight-chain computation, nonlinear scalar accumulation and bias handling. These variants implement the same underlying factorisation within their respective architectural families, but differ in computational efficiency and parameter storage.

### Factorised Isotropic MLPs

These implementations preserve the original sequential weight parameters and are mathematically equivalent to ordinary isotropic MLPs.

* **`FactorisedParallelLinear_IndependentChains`**
  Direct implementation that independently computes cumulative weight chains and weight-bias contractions. Repeated matrix multiplications introduce substantial computational overhead, particularly during training.

* **`FactorisedParallelLinear_RecursiveChains`**
  Constructs cumulative weight chains recursively, reusing intermediate matrix products rather than recomputing each chain. Also introduces recursive accumulation of nonlinear scalar and bias contributions.

* **`FactorisedParallelLinear_GroupedScalars`**
  Builds upon recursive weight-chain computation by grouping the initial weight and bias contributions within the nonlinear scalar recurrence. This reduces redundant scalar operations and simplifies the forward computation.

* **`FactorisedParallelLinear_FinalOptimised`**
  Refined implementation using recursive weight-chain construction and grouped scalar accumulation. Includes a specialised inference path that precomputes and stores contracted weights and biases, allowing parallel evaluation of weight-input products.

### Reparameterised Isotropic MLPs

These implementations replace cumulative weight chains and weight-bias contractions with independently trainable parameters, expanding the functional class relative to ordinary isotropic MLPs.

* **`ReparameterisedParallelLinear_VectorBiases`**
  Stores independently trainable weight matrices in a single concatenated parameter matrix, while retaining individual bias vectors. Nonlinear scalar products and bias contributions are computed explicitly.

* **`ReparameterisedParallelLinear_RecursiveBiases`**
  Retains the concatenated weight matrix and individual bias vectors, but introduces recursive accumulation of scalar coefficients and bias contributions. This reduces redundant scalar computations compared with the original implementation.

* **`ReparameterisedParallelLinear_MatrixBiases`**
  Stores each layer's independent bias contributions as a matrix rather than a collection of vectors. Combines nonlinear coefficients with these matrices using matrix multiplication, reducing the number of separate scalar-vector operations.

* **`ReparameterisedParallelLinear_GroupedScalars`**
  Retains individual bias vectors but groups the initial weight and bias contributions within the nonlinear scalar recurrence. This provides an alternative recursive formulation without requiring matrix-stored biases.

* **`ReparameterisedParallelLinear_CoefficientMatrix`**
  Stores bias contributions as matrices and maintains an evolving collection of nonlinear scalar coefficients throughout the forward pass. These coefficients are multiplied directly with the corresponding bias matrices, avoiding repeated reconstruction of scalar products.

### Additional Implementation

* **`ContractedParallelLinear`**
  An earlier formulation that stores cumulative weight transforms as independent parameters, rather than constructing them from sequential weight matrices. Unlike the factorised implementations, this changes the optimisation dynamics and is conceptually related to the reparameterised architecture.

### Usage

The primary experimental notebooks use `NormalLinear` as the standard baseline, `FactorisedParallelLinear_FinalOptimised` for the factorised architecture, and selected reparameterised implementations (`ReparameterisedParallelLinear_GroupedScalars` or `ReparameterisedParallelLinear_MatrixBiases`).

The remaining implementations are retained to document alternative computational strategies and facilitate further benchmarking.
