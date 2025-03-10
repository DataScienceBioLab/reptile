# Q3: Maximizing Monsters

## Problem Description
In her experiments with building an in-game analog computer out of monsters, Sierra has learned something else. She can increase her computational power by luring monsters to the wilderness areas she has control of, but trying to put too many monsters in the same area leads to weird glitches.

Sierra has identified many different logic gates that she can build out of monsters. Each one takes a different number of monsters. She wants to pack as many logic gates into each wilderness area as she can without things starting to get glitchy.

There are N types of logic gates that Sierra can make out of monsters. Each gate i uses Mi monsters, with K being the maximum total monsters that can be in an area without glitches. Determine that maximum number of monsters that can productively be in that area at once (i.e. if every monster has to be part of a logic gate, what is the most monsters that can be in the area with triggering glitches).

For example, if there were two types of logic gate that needed 9 and 12 monsters respectively, and the area could handle a maximum of 31 monsters, in practice, Sierra could have a maximum of 30 monsters doing computation by arranging them into two of the first logic gate type and one of the second logic gate type. If there were a type of logic gate that only needed 1 monster, she would be able to maximize the amount of computation to a total of 31 monsters, but no such logic gate exists so no existing combination of gates can get her all the way to 31.

## Input Format
- The first line contains T, the number of test cases.
- Each test comprises two lines:
  - The first line contains two integers, N and K, representing the number of logic gate types and maximum monsters that can be used in total, respectively.
  - The second line consists of space separated integers, M0 through MN-1, representing the number of monsters each logic gate type requires.

## Constraints
- 1 ≤ T ≤ 10
- 1 ≤ N, K ≤ 2000
- 1 ≤ Mi ≤ 2000

## Output Format
Output T lines, the maximum total monsters that can be used in each test case without exceeding the total limit (K).

## Examples

### Example 0
```
Sample Input:
3  
3 12  
1 6 9  
5 9  
3 4 4 4 8  
4 11  
5 7 8 9

Sample Output:
12  
9  
10
```

**Explanation:**
- In the first test case, you can use two of the 6-monster gates to achieve the maximum total of 12 monsters.
- In the second test case, you can use three of the 3-monster gates to hit the maximum total of 9 monsters.
- In the third test case, there is no way to reach the limit of 11 monsters, so the closest you can come is 10, by using two of the 5-monster gates.

### Example 1
```
Sample Input:
1  
8 10  
11 12 13 14 15 16 17 3

Sample Output:
9
```

**Explanation:**
The first seven gate types are useless since they immediately exceed the limit. The best we can do is use three of the last type for a total of 9.

## Implementation Notes
This is a dynamic programming problem similar to the unbounded knapsack problem. The key insight is that we need to:
1. For each test case, find the maximum sum of monsters that can be achieved using any combination of gates
2. The sum must be less than or equal to K
3. We can use a DP array where dp[i] represents whether it's possible to achieve a sum of i monsters

The recurrence relation would be:
dp[i] = true if there exists a gate j such that dp[i - monsters[j]] is true 