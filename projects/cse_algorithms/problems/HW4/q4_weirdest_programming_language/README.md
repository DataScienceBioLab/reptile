# Q4: The Weirdest Programming Language

## Problem Description
Now that she's got her monsters organized, Sierra wants to know how versatile the computations she can do with them are. She wants to know how many different sequences of logic gates she can make out of N monsters.

Given G types of logic gates, where gate type i uses Ni monsters, how many ways are there to make a sequence of logic gates made up of exactly Ntotal monsters? Note that order DOES matter.

## Input Format
- The first line of the input provides T, the total number of test cases.
- The next T pairs of lines describe a single test case:
  - The first line in a test case provides the number of gate types (G) and the total target number of monsters (Ntotal).
  - The second line has G values, each position indicating the number of monsters (Ni) needed to make logic gate type i.

## Constraints
- 1 ≤ T ≤ 10
- 1 ≤ G ≤ 50
- 1 ≤ Ntotal ≤ 106
- 1 ≤ Ni ≤ 106

## Output Format
T lines, one per test case, each indicating the number of combinations possible to make a sequence with that total number of monsters. Some of the values are quite large, so mod your answer by 1000000009. Note: make sure you read that last sentence!

## Examples

### Example 0
```
Sample Input:
2  
2 5  
1 2  
2 7  
1 3

Sample Output:
8  
9
```

**Explanation:**
There are 2 test cases. In the first we have 2 gate types, one that needs 1 monster and one that needs 2 monsters. We must make a sequence of length five using these two types. Let's call the first gate A and the second BB (since it's two monsters). We can then figure out all of the possible gate layouts of length 5:

- A A A A A  
- A A A BB  
- A A BB A  
- A BB A A  
- BB A A A  
- A BB BB  
- BB A BB  
- BB BB A  

For a total of 8 different options.

For the second test case, there are again 2 types of gates. The first one is uses 1 monster and the second uses 3. We need a sequence that uses a total of 7. If we call the gates A and CCC, the options this time are:

- A A A A A A A  
- A A A A CCC  
- A A A CCC A  
- A A CCC A A  
- A CCC A A A  
- CCC A A A A  
- A CCC CCC  
- CCC A CCC  
- CCC CCC A

## Implementation Notes
This is a dynamic programming problem where we need to count the number of ways to make sequences. The key insight is that:
1. For each position in the sequence, we can try using any gate type that fits
2. We need to keep track of both the current position and the remaining monsters
3. The order matters, so we need to consider all possible arrangements

The recurrence relation would be:
dp[pos][remaining] = sum(dp[pos + 1][remaining - monsters[i]] for all valid gate types i)

The base cases would be:
- dp[pos][0] = 1 if pos == target_length, 0 otherwise
- dp[pos][remaining] = 0 if remaining < 0

Remember to take the modulo 1000000009 at each step to avoid overflow. 