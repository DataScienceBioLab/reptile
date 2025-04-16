# Policy Iteration Implementation

This module provides an implementation of the Policy Iteration algorithm for solving a Markov Decision Process (MDP) in a grid world environment.

## Environment Description

The environment is a 5×5 grid world with:
- Goal state at (4,4) with reward +10
- Trap states at (1,1) and (3,2) with reward -5
- Walls blocking positions (0,2), (2,1), and (2,3)
- All other states have reward 0
- Discount factor γ = 0.9

## Usage

```python
from educational_app.policy_iteration import GridWorld

# Create and initialize environment
env = GridWorld()

# Visualize initial state
env.visualize("Initial Grid World")

# Run policy iteration
env.policy_iteration()

# Visualize final state
env.visualize("Grid World after Policy Iteration")
```

## Algorithm Details

The implementation follows the standard Policy Iteration algorithm:

1. **Policy Evaluation**: Calculate the value function for the current policy using the Bellman equation:
   ```
   V^π(s) = Σ_a π(a|s) Σ_s' p(s'|s,a) [R(s,a,s') + γV^π(s')]
   ```

2. **Policy Improvement**: Update the policy to be greedy with respect to the current value function:
   ```
   π'(s) = argmax_a Σ_s' p(s'|s,a) [R(s,a,s') + γV^π(s')]
   ```

3. Repeat steps 1-2 until the policy converges.

## Visualization

The `visualize()` method provides a graphical representation of the grid world:
- Green: Goal state
- Red: Trap states
- Gray: Wall states
- Arrows: Current policy
- Numbers: State values

## Testing

Run the tests using pytest:
```bash
pytest src/educational_app/test_policy_iteration.py
```

## Dependencies

- NumPy
- Matplotlib
- pytest (for testing) 