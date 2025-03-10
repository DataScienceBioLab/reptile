# Q1: Deteriorating Conditions

## Problem Description
You brought m pieces of armor with you in your quest to explore the towers. The designers of the game want to encourage you to rotate your armor (and try all the latest fashions!) so they created a system where your armor wears out faster if you use it in multiple fights per day.

Each fight has a baseline number of strength points that it will cause your armor to lose if you wear it while fighting. In the first battle you wear it for, it only loses that many points. The second time, you lose twice that many points, the third time is triple and so on, such that the kth time is k-times the base cost.

Given m pieces of armor and n fights, and a base cost of Ci for fight i, find and print the minimum total amount of strength points your armor will lose collectively.

Note: You can choose what order the fights happen in.

## Input Format
- The first line contains two space-separated integers describing the respective values of n and m.
- The second line contains n space-separated positive integers describing the respective values of C0, C1, …, Cn-1

## Constraints
- 1 ≤ n, m ≤ 100
- 1 ≤ Ci ≤ 106
- answer < 231
- 0 ≤ i < n

## Output Format
Print the total cost for all fights.

## Examples

### Example 0
```
Sample Input:
3 3  
2 5 6

Sample Output:
13  
```

**Explanation:**
There are three fights and three pieces of armor. Since each armor is used for one fight, there is only a multiple of ×1 for each. The total cost is:
(2 × 1) + (5 × 1) + (6 × 1) = 13

### Example 1
```
Sample Input:
7 3  
6 1 28 10 15 3 21

Sample Output:
105  
```

**Explanation:**
There are seven fights and three pieces of armor.

An optimal assignment is:
- first armor: 28, 10, and 1
- second armor: 21 and 6
- third armor: 15 and 3

The first armor would lose 28 + 10×2 + 1×3 = 51
The second armor would lose 21 + 6×2 = 33
The third armor would lose 15 + 3×2 = 21

… for a grand total of 51 + 33 + 21 = 105.

## Implementation Notes
This is a greedy algorithm problem. The key insight is that we want to minimize the total cost by:
1. Assigning the highest cost fights to the first use of each armor piece
2. Assigning the next highest cost fights to the second use of each armor piece
3. And so on...

This way, we minimize the impact of the increasing multipliers on the highest cost fights. 