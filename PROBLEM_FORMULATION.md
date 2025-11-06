# Problem Formulation

## Problem Overview

![Problem Structure Flow Diagram](images/flow.png)

The figure above illustrates the conceptual overview of the optimization problem structure and its components, showing the relationships between decision variables, objective components, and constraints.

## Problem Statement

Minimize end-to-end processing latency in Brain-Computer Interface (BCI) systems while preserving classification accuracy and maintaining computational efficiency within hardware constraints.

## Decision Variables

The optimization problem involves three types of decision variables:

1. **Model Parameters** ($\boldsymbol{\theta} \in \mathbb{R}^{n_\theta}$)
   - Neural network weights
   - Classifier coefficients
   - Feature transformation parameters
   - Dimension: $n_\theta = 50$ (default)

2. **Design Variables** ($\boldsymbol{\alpha} \in \mathbb{R}^{n_\alpha}$)
   - Feature dimension: $\alpha_0 \in [\alpha_{\min}, \alpha_{\max}]$
   - Sampling rate: $\alpha_1 \in [100, 500]$ Hz
   - Filter bandwidth: $\alpha_2 \in [5, 30]$ Hz
   - Dimension: $n_\alpha = 3$

3. **Regularization Parameter** ($\lambda \in \mathbb{R}_+$)
   - Sparsity control coefficient
   - Range: $\lambda \in [0, 1]$
   - Dimension: $n_\lambda = 1$

**Total Decision Vector**: $x = [\boldsymbol{\theta}^T, \boldsymbol{\alpha}^T, \lambda]^T \in \mathbb{R}^{n}$ where $n = n_\theta + n_\alpha + n_\lambda = 54$

## Objective Function

### Scalarized Multi-Objective Function

\[
J(\boldsymbol{\theta}, \boldsymbol{\alpha}, \lambda) = w_L \frac{L(\boldsymbol{\theta}, \boldsymbol{\alpha})}{L_{\max}} + w_E E(\boldsymbol{\theta}, \boldsymbol{\alpha}) + w_C \frac{C(\boldsymbol{\theta}, \lambda)}{C_{\max}}
\]

**Weight Configuration**:
- $w_L = 0.4$ (Latency weight)
- $w_E = 0.4$ (Error weight)
- $w_C = 0.2$ (Complexity weight)
- Constraint: $w_L + w_E + w_C = 1$

**Normalization Constants**:
- $L_{\max} = 50.0$ ms
- $C_{\max} = 100.0$

## Component Functions

### 1. Latency Component

\[
L(\boldsymbol{\theta}, \boldsymbol{\alpha}) = \beta_0 + \sum_{b=1}^{B} \beta_b\, \phi_b(\boldsymbol{\alpha})
\]

**Implementation**:
\[
L(\boldsymbol{\theta}, \boldsymbol{\alpha}) = \beta_0 + \beta_1 \frac{\alpha_0}{64} + \beta_2 \frac{\alpha_1}{500} + \beta_3 \frac{\|\boldsymbol{\theta}\|}{10}
\]

**Parameters**:
- $\beta_0 = 2.0$ (base latency in ms)
- $\boldsymbol{\beta} = [0.15, 0.08, 0.03]$ (latency coefficients)
- Lower bound: $L(\boldsymbol{\theta}, \boldsymbol{\alpha}) \ge 0.1$ ms

**Physical Interpretation**:
- Feature dimension ($\alpha_0$) affects preprocessing time
- Sampling rate ($\alpha_1$) affects data acquisition time
- Model complexity ($\|\boldsymbol{\theta}\|$) affects inference time

### 2. Classification Error Component

\[
E(\boldsymbol{\theta}, \boldsymbol{\alpha}) = \text{base\_error} + \text{complexity\_penalty} - \text{feature\_benefit}
\]

**Implementation**:
\[
E(\boldsymbol{\theta}, \boldsymbol{\alpha}) = 0.3 + 0.1 \frac{\|\boldsymbol{\theta}\|}{\sqrt{n_\theta}} - 0.15 \frac{1}{1 + \alpha_0/32}
\]

**Components**:
- Base error: $0.3$ (baseline classification loss)
- Complexity penalty: Increases with model size
- Feature benefit: Decreases error with more features
- Lower bound: $E(\boldsymbol{\theta}, \boldsymbol{\alpha}) \ge 0.05$

**Physical Interpretation**:
- Larger models ($\|\boldsymbol{\theta}\|$) may overfit, increasing error
- More features ($\alpha_0$) improve classification accuracy

### 3. Model Complexity Component

\[
C(\boldsymbol{\theta}, \lambda) = \|\boldsymbol{\theta}\|_2^2 + \lambda \sum_{j=1}^{n_\theta} \log(\varepsilon + |\theta_j|)
\]

**Components**:
- L2 regularization: $\|\boldsymbol{\theta}\|_2^2$ (prevents large weights)
- Log penalty: $\lambda \sum_{j=1}^{n_\theta} \log(\varepsilon + |\theta_j|)$ (promotes sparsity)
- Numerical stability: $\varepsilon = 10^{-8}$

**Physical Interpretation**:
- L2 term: Prevents overfitting and improves generalization
- Log term: Encourages sparse models (fewer active parameters)
- Regularization parameter $\lambda$: Controls sparsity-accuracy trade-off

## Constraints

### 1. Latency Constraint

\[
L(\boldsymbol{\theta}, \boldsymbol{\alpha}) \leq \tau_{\max}
\]

**Test Case Values**:
- Real-Time Mode: $\tau_{\max} = 15$ ms
- Balanced Mode: $\tau_{\max} = 25$ ms
- High-Accuracy Mode: $\tau_{\max} = 40$ ms

**Physical Meaning**: Total processing delay must not exceed the maximum permissible latency for real-time operation.

### 2. Accuracy Constraint

\[
E(\boldsymbol{\theta}, \boldsymbol{\alpha}) \leq \ell_{\max}
\]

**Test Case Values**:
- Real-Time Mode: $\ell_{\max} = 0.40$ (60% accuracy)
- Balanced Mode: $\ell_{\max} = 0.35$ (65% accuracy)
- High-Accuracy Mode: $\ell_{\max} = 0.25$ (75% accuracy)

**Physical Meaning**: Classification error must remain below threshold to ensure reliable signal interpretation.

### 3. Design Variable Bounds

\[
\boldsymbol{\alpha}_{\min} \leq \boldsymbol{\alpha} \leq \boldsymbol{\alpha}_{\max}
\]

**Component-wise Bounds**:
- Feature dimension: $\alpha_0 \in [\alpha_{\min}, \alpha_{\max}]$
  - Real-Time: $\alpha_0 \in [20, 40]$
  - Balanced: $\alpha_0 \in [40, 60]$
  - High-Accuracy: $\alpha_0 \in [60, 80]$
- Sampling rate: $\alpha_1 \in [100, 500]$ Hz
- Filter bandwidth: $\alpha_2 \in [5, 30]$ Hz

**Physical Meaning**: Design variables must remain within hardware-feasible ranges for embedded deployment.

### 4. Regularization Constraint

\[
\lambda \ge 0
\]

**Physical Meaning**: Regularization parameter must be non-negative to ensure proper penalty behavior.

## Constraint Reformulation

### Standard Form

All constraints are converted to inequality form $c_j(x) \leq 0$:

\[
\begin{align}
c_1(x) &= L(\boldsymbol{\theta}, \boldsymbol{\alpha}) - \tau_{\max} \leq 0 \\
c_2(x) &= E(\boldsymbol{\theta}, \boldsymbol{\alpha}) - \ell_{\max} \leq 0 \\
c_3(x) &= \alpha_{\min} - \alpha_0 \leq 0 \\
c_4(x) &= \alpha_0 - \alpha_{\max} \leq 0 \\
c_5(x) &= -\lambda \leq 0
\end{align}
\]

### Penalty Function

The constrained problem is reformulated using quadratic penalty:

\[
\mathcal{L}_P(x; \boldsymbol{\rho}) = J(x) + \sum_{j=1}^{5} \rho_j [c_j^+(x)]^2
\]

where:
- $c_j^+(x) = \max(0, c_j(x))$: Positive part of constraint violation
- $\rho_j > 0$: Penalty coefficients (default: $\rho_j = 10$)

### Adaptive Penalty Strategy

Penalty coefficients are adapted during optimization:

\[
\rho_j^{(k)} = \rho_0 \left(1 + 0.1 \cdot k\right)
\]

where $k$ is the iteration number and $\rho_0 = 10$ is the base penalty.

## Feasibility

### Exact Feasibility

Methods like SLSQP and trust-constr achieve exact constraint satisfaction:

\[
c_j(x^*) \leq \varepsilon_{\text{feas}} \quad \forall j
\]

where $\varepsilon_{\text{feas}} = 10^{-4}$ is the feasibility tolerance.

### Practical Feasibility

Penalty-based methods (GD, L-BFGS, NCG) achieve practical feasibility:

\[
\sum_{j=1}^{5} \max(0, c_j(x^*)) \leq \varepsilon_{\text{prac}}
\]

where $\varepsilon_{\text{prac}} = 0.1$ is the practical feasibility threshold.

## Initialization

### Default Initialization

\[
x_0 = [\boldsymbol{\theta}_0^T, \boldsymbol{\alpha}_0^T, \lambda_0]^T
\]

where:
- $\boldsymbol{\theta}_0 \sim \mathcal{N}(0, 0.1^2)$: Random initialization
- $\boldsymbol{\alpha}_0 = [(\alpha_{\min} + \alpha_{\max})/2, 250, 20]^T$: Mid-range values
- $\lambda_0 = 0.1$: Small regularization

### Bounds

\[
\begin{align}
\boldsymbol{\theta} &\in [-1, 1]^{n_\theta} \\
\alpha_0 &\in [\alpha_{\min}, \alpha_{\max}] \\
\alpha_1 &\in [100, 500] \\
\alpha_2 &\in [5, 30] \\
\lambda &\in [0, 1]
\end{align}
\]

## Optimization Problem Summary

**Problem Type**: Constrained nonlinear optimization

**Objective**: Minimize $J(\boldsymbol{\theta}, \boldsymbol{\alpha}, \lambda)$

**Subject to**:
- $L(\boldsymbol{\theta}, \boldsymbol{\alpha}) \leq \tau_{\max}$
- $E(\boldsymbol{\theta}, \boldsymbol{\alpha}) \leq \ell_{\max}$
- $\boldsymbol{\alpha}_{\min} \leq \boldsymbol{\alpha} \leq \boldsymbol{\alpha}_{\max}$
- $\lambda \ge 0$

**Reformulation**: Unconstrained minimization of $\mathcal{L}_P(x; \boldsymbol{\rho})$

**Solution Methods**: GD, Newton, BFGS, L-BFGS, NCG, SLSQP, trust-constr

