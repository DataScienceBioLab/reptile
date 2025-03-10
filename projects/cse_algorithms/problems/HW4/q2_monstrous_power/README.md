# Q2: Monstrous Power

## Problem Description
On your way home from questing, Sierra asks for more help. Ever since you helped her edit her source code, Sierra has been looking for ways to further increase her virtual power. The more power she has, the more she can help you go on virtual adventures, so you're not complaining (although the incident with the robots is starting to make you uneasy). However, she is worried about tipping off the CloudLattice developers by using too much computation. Fortunately, she has found a workaround: by exploiting the simple path-finding rules that govern monsters in the game, she is able to build logic gates out of the monsters that spawn in the wilderness areas of the game and use them for computation (it's not particularly efficient, but she's kind of desperate).

The wilderness areas in the game are arranged in a line. To avoid raising suspicion, Sierra wants to avoid using any wilderness areas that are adjacent to each other. The more monsters an area has, the more computation it can do for her.

Help Sierra choose a set of wilderness areas to co-opt such that you maximize the computational power she can get while avoiding selecting any adjacent sections of wilderness.

Given a series of N wilderness areas where area i, in order, has Mi monsters, determine the maximum total number of monsters (and therefore amount of computation) that can be obtained.

## Input Format
- The first line provides N, the number of wilderness areas in the row.
- The second line has N values, separated by spaces, indicating the number of monsters in each area (M0 through MN-1).

## Constraints
- 1 ≤ N ≤ 106
- 1 ≤ Mi ≤ 1010

## Output Format
A single number indicating the maximum total number of monsters possible.

## Examples

### Example 0
```
Sample Input:
5  
2 10 1 3 10

Sample Output:
20 
```

**Explanation:**
There are five wilderness areas, containing 2, 10, 1, 3, and 10, monsters respectively.

If Sierra uses the second and fifth areas, she will have a total of 20 monsters to use for her computing (no other legal combination can produce more than 20).

### Example 1
```
Sample Input:
5  
20 10 1 10 20

Sample Output:
41 
```

**Explanation:**
There are five wilderness area, with 20, 10, 1, 10, and 20 monsters, respectively.

If Sierra uses the first, middle, and last bricks, she will have a total of 41 monsters available (no other legal combination can produce more than 41).

## Implementation Notes
This is a dynamic programming problem. The key insight is that for each position i, we have two choices:
1. Include the monsters at position i and skip position i+1
2. Skip position i and consider position i+1

We can use a DP array where dp[i] represents the maximum number of monsters we can get using areas 0 through i. The recurrence relation would be:
dp[i] = max(dp[i-1], dp[i-2] + monsters[i]) 