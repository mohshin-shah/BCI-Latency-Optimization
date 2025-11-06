# Theoretical Background

## Introduction

Brain-Computer Interfaces (BCIs) establish direct communication pathways between the human brain and external systems by translating neural activity into actionable commands. The performance of BCI systems depends on three interlinked attributes:

1. **Latency**: Total processing delay between neural signal acquisition and control actuation
2. **Classification Accuracy**: Reliability and consistency of decoded intentions
3. **Computational Complexity**: Resource requirements for real-time processing

## Problem Motivation

### Real-Time Requirements

Contemporary EEG-based BCIs operate at sampling rates of 256-512 Hz with analysis windows ranging from one to four seconds. Key performance requirements include:

- **Latency Target**: Total delay must remain below 30 ms for seamless control
- **Accuracy Target**: At least 80% classification accuracy for reliable signal interpretation
- **Hardware Constraints**: Limited computational resources in embedded platforms

### Performance Trade-offs

- Classical algorithms (CSP, LDA) provide interpretable results but exhibit 15-25 ms delays per window
- Modern deep architectures achieve 8-12 ms latency but require higher computational resources
- Achieving high accuracy typically requires deeper models or larger feature spaces, increasing computation time

## Mathematical Framework

![Problem Structure Flow Diagram](images/flow.png)

The figure above shows the conceptual overview of the optimization problem structure, illustrating how decision variables, objective components, and constraints interact in the BCI latency optimization framework.

### Objective Function Structure

The optimization objective integrates three components into a scalar cost function:

$$
J(\boldsymbol{\theta}, \boldsymbol{\alpha}, \lambda) = w_L \frac{L(\boldsymbol{\theta}, \boldsymbol{\alpha})}{L_{\max}} + w_E E(\boldsymbol{\theta}, \boldsymbol{\alpha}) + w_C \frac{C(\boldsymbol{\theta}, \lambda)}{C_{\max}}
$$

where:
- $\boldsymbol{\theta}$: Model parameters (network weights, classifier coefficients)
- $\boldsymbol{\alpha}$: Design variables (feature length, sampling rate, filter bandwidth)
- $\lambda$: Regularization coefficient controlling model sparsity
- $w_L, w_E, w_C$: Positive weights satisfying $w_L + w_E + w_C = 1$
- $L_{\max}, C_{\max}$: Normalization constants

### Component Definitions

#### Latency Component

$$
L(\boldsymbol{\theta}, \boldsymbol{\alpha}) = \beta_0 + \sum_{b=1}^{B} \beta_b\, \phi_b(\boldsymbol{\alpha})
$$

where:
- $\beta_0$: Base latency
- $\beta_b$: Latency coefficient of computational block $b$
- $\phi_b(\boldsymbol{\alpha})$: Processing cost of block $b$

For typical EEG-based BCIs, $L(\boldsymbol{\theta},\boldsymbol{\alpha})$ ranges from 8-25 ms per decision window.

#### Classification Error Component

$$
E(\boldsymbol{\theta}, \boldsymbol{\alpha}) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} y_{ik} \log p_{ik}(\boldsymbol{\theta}, \boldsymbol{\alpha})
$$

where:
- $y_{ik}$: True class label for sample $i$ and class $k$
- $p_{ik}$: Predicted probability for class $k$ and sample $i$
- $N$: Number of samples
- $K$: Number of classes

#### Model Complexity Component

$$
C(\boldsymbol{\theta}, \lambda) = \|\boldsymbol{\theta}\|_2^2 + \lambda \sum_{j=1}^{n} \log(\varepsilon + |\theta_j|)
$$

where:
- $\|\boldsymbol{\theta}\|_2^2$: L2 regularization term
- $\lambda \sum_{j=1}^{n} \log(\varepsilon + |\theta_j|)$: Log penalty for sparsity
- $\varepsilon > 0$: Small constant preventing numerical instability
- $\lambda \ge 0$: Regularization parameter

## Constraint Formulation

The system operates under the following constraints:

$$
\begin{align}
L(\boldsymbol{\theta}, \boldsymbol{\alpha}) &\leq \tau_{\max} \quad \text{(Latency constraint)} \\
E(\boldsymbol{\theta}, \boldsymbol{\alpha}) &\leq \ell_{\max} \quad \text{(Accuracy constraint)} \\
\boldsymbol{\alpha}_{\min} \leq \boldsymbol{\alpha} &\leq \boldsymbol{\alpha}_{\max} \quad \text{(Design bounds)} \\
\lambda &\ge 0 \quad \text{(Regularization constraint)}
\end{align}
$$

### Constraint Interpretation

- **Latency Constraint**: Ensures processing delay remains below permissible limit $\tau_{\max}$
- **Accuracy Constraint**: Maintains classification loss below $\ell_{\max}$ (e.g., $\ell_{\max}=0.20$ for 80% accuracy)
- **Design Bounds**: Restricts $\boldsymbol{\alpha}$ to hardware-feasible ranges for embedded deployment
- **Regularization Constraint**: Ensures $\lambda \ge 0$ for proper penalty behavior

## Penalty-Based Reformulation

### Quadratic Penalty Method

The constrained problem is reformulated using the quadratic penalty function:

$$
\mathcal{L}_P(x; \boldsymbol{\rho}) = J(x) + \sum_{j=1}^{4} \rho_j [c_j^+(x)]^2
$$

where:
- $c_j^+(x) = \max(0, c_j(x))$: Positive part of constraint violation
- $\rho_j > 0$: Penalty coefficients (typically $\rho_j = 10$)
- $x = [\boldsymbol{\theta}^T, \boldsymbol{\alpha}^T, \lambda]^T$: Combined decision vector

### Augmented Lagrangian Method

For more precise constraint handling, the Augmented Lagrangian is:

$$
\mathcal{L}_A(x, \boldsymbol{\nu}; \mu) = J(x) + \sum_{j=1}^{5} \nu_j\, c_j^+(x) + \frac{\mu}{2}\sum_{j=1}^{5}\big(c_j^+(x)\big)^2
$$

where:
- $\boldsymbol{\nu}$: Lagrange multipliers
- $\mu > 0$: Penalty parameter

This formulation combines Lagrange multiplier precision with penalty-based smoothness.

## Optimization Algorithms

### First-Order Methods

#### Gradient Descent (GD)

$$
x_{k+1} = x_k - \eta_k \nabla \mathcal{L}_P(x_k)
$$

- **Complexity**: $O(n)$ per iteration
- **Convergence**: Linear for convex functions
- **Memory**: $O(n)$

#### Nonlinear Conjugate Gradient (NCG)

$$
\begin{align}
p_k &= -g_k + \beta_k p_{k-1} \\
\beta_k^{PR+} &= \max\left(0,\frac{g_k^T(g_k - g_{k-1})}{g_{k-1}^T g_{k-1}}\right)
\end{align}
$$

where $g_k = \nabla \mathcal{L}_P(x_k)$.

- **Complexity**: $O(n)$ per iteration
- **Convergence**: Superlinear for smooth objectives
- **Memory**: $O(n)$

### Quasi-Newton Methods

#### BFGS (Broyden-Fletcher-Goldfarb-Shanno)

Updates Hessian approximation using:

$$
B_{k+1} = B_k + \frac{y_k y_k^T}{y_k^T s_k} - \frac{B_k s_k s_k^T B_k}{s_k^T B_k s_k}
$$

where $s_k = x_{k+1} - x_k$ and $y_k = \nabla \mathcal{L}_P(x_{k+1}) - \nabla \mathcal{L}_P(x_k)$.

- **Complexity**: $O(n^2)$ per iteration
- **Convergence**: Superlinear
- **Memory**: $O(n^2)$

#### L-BFGS (Limited-Memory BFGS)

Stores only the last $m$ curvature pairs $(s_i, y_i)$ and computes descent direction using two-loop recursion.

- **Complexity**: $O(mn)$ per iteration
- **Convergence**: Superlinear
- **Memory**: $O(mn)$ where $m \ll n$

### Constrained Optimization Methods

#### SLSQP (Sequential Least Squares Programming)

Solves a sequence of quadratic subproblems:

$$
\begin{align}
\min_{d} \quad & \tfrac{1}{2} d^T B_k d + g_k^T d \\
\text{s.t.}\quad & A_k d + c_k = 0, \quad G_k d + h_k \le 0
\end{align}
$$

- **Complexity**: $O(n^2)$ to $O(n^3)$ per iteration
- **Convergence**: Fast practical convergence on smooth problems
- **Constraint Handling**: Exact feasibility

#### Trust-Region Constrained (trust-constr)

Uses trust-region framework with constraint handling:

$$
\begin{align}
\min_{p} \quad & m_k(p) = g_k^T p + \tfrac{1}{2} p^T B_k p \\
\text{s.t.}\quad & \|p\| \le \Delta_k, \quad c(x_k) + J_k p \approx 0, \quad h(x_k) + H_k p \le 0
\end{align}
$$

where $\Delta_k$ is the trust-region radius.

- **Complexity**: $O(n^2)$ to $O(n^3)$ per iteration
- **Convergence**: Robust to ill-conditioning
- **Constraint Handling**: Strong feasibility restoration

## Convergence Analysis

### Convergence Criteria

The optimization terminates when:

1. **Gradient Norm**: $\|\nabla \mathcal{L}_P(x_k)\|_2 < 10^{-5}$
2. **Relative Change**: $|J(x_{k+1}) - J(x_k)| / |J(x_k)| < 10^{-6}$
3. **Maximum Iterations**: $k \ge 500$

### Convergence Properties

- **L-BFGS**: Typically converges in 15-25 iterations
- **SLSQP**: Achieves exact feasibility with 80-120 iterations
- **trust-constr**: Robust convergence with 100-300 iterations
- **GD/NCG**: May require 200-400 iterations depending on conditioning

## Practical Considerations

### Parameter Selection

- **Weights**: $(w_L, w_E, w_C) = (0.4, 0.4, 0.2)$ for balanced optimization
- **Penalty Factors**: $\rho_j = 10$ with adaptive scaling
- **Regularization**: $\lambda \in [0, 1]$ for sparsity control

### Numerical Stability

- Small constant $\varepsilon = 10^{-8}$ prevents log singularities
- Bounds on variables stabilize search: $\boldsymbol{\theta} \in [-1, 1]$
- Normalization constants $L_{\max} = 50.0$, $C_{\max} = 100.0$

## References

1. Blankertz, B., et al. "The Berlin BCI: Progress and Perspectives." *Neural Engineering*, 2007.
2. Wolpaw, J. R., et al. "Brain-computer interfaces for communication and control." *Clinical Neurophysiology*, 2002.
3. Lotte, F., et al. "A review of classification algorithms for EEG-based brain-computer interfaces." *Journal of Neural Engineering*, 2007.
4. Thompson, D. E., et al. "Performance assessment in brain-computer interface-based augmentative and alternative communication." *Biomedical Engineering*, 2015.

