# Problem 2: Mouse in a House Markov Chain

## Problem Statement
A mouse moves between five rooms arranged in a circle. The mouse's movement follows these rules:
- 60% probability of staying in the current room
- 20% probability of moving to each adjacent room
- 0% probability of moving to non-adjacent rooms

## Mathematical Formulation

### Transition Matrix
The 5x5 transition matrix P represents movement probabilities:
```python
P = [[0.6, 0.2, 0.0, 0.0, 0.2],
     [0.2, 0.6, 0.2, 0.0, 0.0],
     [0.0, 0.2, 0.6, 0.2, 0.0],
     [0.0, 0.0, 0.2, 0.6, 0.2],
     [0.2, 0.0, 0.0, 0.2, 0.6]]
```

### Properties of the System
1. **State Space**: 5 discrete states (rooms 1-5)
2. **Transition Properties**:
   - Diagonal elements: 0.6 (staying probability)
   - Adjacent elements: 0.2 (movement probability)
   - Non-adjacent elements: 0.0 (impossible transitions)
3. **Matrix Properties**:
   - Symmetric
   - Stochastic (columns sum to 1)
   - Irreducible (all states communicate)
   - Aperiodic (due to self-transitions)

## Analysis Results

### Stationary Distribution
Due to the symmetric structure of the transition matrix and the uniform movement probabilities:
- The stationary distribution is uniform: π = [0.2, 0.2, 0.2, 0.2, 0.2]
- Each room is visited equally often in the long run
- This makes intuitive sense due to the symmetric nature of the problem

### Convergence Properties
1. **Rate of Convergence**:
   - Fast convergence due to high self-transition probabilities
   - Typically reaches near-stationary behavior within 20-30 steps

2. **Mixing Time**:
   - The time needed to approach the stationary distribution
   - Influenced by the second-largest eigenvalue of P

### Simulation Results

#### Short-term Behavior
- Initial transient period shows dependence on starting position
- Local exploration of adjacent rooms dominates early behavior
- High probability of remaining in the same room creates "sticky" behavior

#### Long-term Behavior
1. **Visit Frequencies**:
   - Converges to uniform distribution across rooms
   - Validates theoretical stationary distribution

2. **Path Characteristics**:
   - Continuous segments of same-room stays
   - Gradual exploration of all rooms
   - No "jumps" between non-adjacent rooms

## Implementation Details

### Simulation Components
1. **State Tracking**:
   - Current room position
   - History of visited rooms
   - Time spent in each room

2. **Visualization Tools**:
   - Room transition heatmap
   - Visit frequency histogram
   - Path trajectory plot

### Key Metrics
1. **Average Stay Duration**:
   - Expected number of steps in same room
   - Theoretical mean: 2.5 steps (1/0.4)

2. **Room Coverage**:
   - Time to visit all rooms
   - Distribution of visit frequencies

## Discussion Points
1. **Physical Interpretation**:
   - Models realistic animal movement patterns
   - Balance between exploration and stability

2. **Mathematical Properties**:
   - Ergodicity ensures long-term predictability
   - Symmetry leads to uniform stationary distribution

3. **Applications**:
   - Animal behavior modeling
   - Space utilization analysis
   - Movement pattern prediction 