# Problem 1: Markov Chain Analysis

## Problem Statement
Given a 3x3 transition matrix P:
```python
P = [[0.6, 0.2, 0.3],
     [0.3, 0.5, 0.2],
     [0.1, 0.3, 0.5]]
```

## Analysis

### a) Transition Matrix Validation
The matrix P must be verified as a valid transition matrix by checking two key properties:

1. **Property 1**: All entries must be between 0 and 1
   - Verification: ✓ All entries are probabilities between 0 and 1

2. **Property 2**: Each column must sum to 1
   - Column 1: 0.6 + 0.3 + 0.1 = 1.0 ✓
   - Column 2: 0.2 + 0.5 + 0.3 = 1.0 ✓
   - Column 3: 0.3 + 0.2 + 0.5 = 1.0 ✓

**Conclusion**: P is a valid transition matrix as it satisfies both properties.

### b) Eigenvalue Analysis
The eigenvalues of the transition matrix are:
```
λ₁ = 1.0 + 0.0j
λ₂ = 0.3 + 0.1j
λ₃ = 0.3 - 0.1j
```

**Key Observations**:
- The largest eigenvalue is 1, which is expected for a stochastic matrix (Perron-Frobenius theorem)
- The other eigenvalues have magnitude less than 1, ensuring convergence to a stationary distribution
- The complex conjugate pair indicates some cyclic behavior in the transient states

### c) Stationary Distribution
The stationary distribution π was found to be:
```
π = [0.38, 0.34, 0.28]
```

**Interpretation**:
- State 1: Visited 38% of the time in the long run
- State 2: Visited 34% of the time in the long run
- State 3: Visited 28% of the time in the long run

This distribution satisfies πP = π, meaning it remains unchanged under the transition matrix.

### d) Convergence Analysis
Starting from initial state [1, 0, 0]:

**Short-term Evolution**:
- After 2 steps: [0.45, 0.35, 0.20]
- After 10 steps: [0.38, 0.34, 0.28]

**Convergence Properties**:
1. **Speed**: The system converges relatively quickly (approximately 10 steps)
2. **Pattern**: Monotonic convergence without oscillations
3. **Independence**: Final distribution is independent of initial state
4. **Stability**: Once reached, the stationary distribution remains stable

## Mathematical Insights
1. The existence of a unique stationary distribution is guaranteed by:
   - The matrix being stochastic
   - The chain being irreducible (all states can reach all other states)
   - The chain being aperiodic (evident from non-zero diagonal entries)

2. The convergence rate is related to the second-largest eigenvalue magnitude
   - In this case, |λ₂| ≈ 0.316, indicating relatively fast convergence

## Implementation Notes
The analysis was performed using:
- NumPy for matrix operations and eigenvalue computation
- Custom visualization tools for plotting convergence
- Iterative methods for finding the stationary distribution 