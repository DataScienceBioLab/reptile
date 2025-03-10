# Q5: Lickety-Split

## Problem Description
Since Sierra's new monster-based computational infrastructure is highly parallel, she can use it most efficiently if she splits her computational processes into as many smaller parts as possible.

She has a computation she wants to do that takes S calculations. How many times can she split it into more smaller jobs?

There are K different techniques that she can use to split a chunk of calculations into multiple parts. Each technique is associated with a value, Fi indicating the number of smaller chunks of calculations a chunk can be split into. Splitting technique i can be used only if the current job involves more than Fi calculations AND its calculations can be evenly divided by Fi. The chunk is then split into P/Fi smaller chunks of Fi calculations.

Since each chunk can be given to a different wilderness area, more smaller chunks is better! What is the greatest possible number of times a task requiring S calculations can be split?

## Input Format
The first line contains an integer, T, denoting the number of tests to be performed. The subsequent T pairs of lines describe each test in the following format:

- The first line contains two space-separated integers describing the respective values of S (the starting number of calculations) and K (the number of splitting techniques).
- The second line contains K space-separated integers specifying the values of F0 through FK-1.

## Constraints
- 1 ≤ T ≤ 10
- 1 ≤ S ≤ 10^12
- 1 ≤ K ≤ 1000
- 1 ≤ Fi ≤ 10^12

## Output Format
For each test, determine the maximum number of separate chunks of computation that the task can be split into; output this value on its own line.

## Examples

### Example 0
Input:
```
1
12 3
2 3 4
```
Output:
```
4
```
Explanation:
- Start with 12 calculations
- Use splitting technique 4 to create 3 chunks of 4 calculations each
- Use splitting technique 2 on each 4-calculation chunk to create 2 chunks of 2 calculations each
- Final result: 6 chunks of 2 calculations each

### Example 1
Input:
```
1
84 4
42 7 6 3
```
Output:
```
17
```
Explanation:
- Start with 84 calculations
- Split into 2 chunks of 42 (1 split)
- Split both 42 chunks into 7 chunks of 6 (2 splits)
- Split each 6-chunk into 2 chunks of 3 (14 splits)
- Total: 1 + 2 + 14 = 17 splits

## Implementation Notes
1. This is a greedy problem where we want to maximize the number of splits
2. For each chunk, we should try all possible splitting techniques and choose the one that gives us the most splits
3. We can use a recursive approach where we:
   - Try each splitting technique on the current chunk
   - For each valid split, recursively try to split the resulting chunks
   - Keep track of the maximum number of splits achieved
4. Key insights:
   - A chunk can only be split if it's larger than the splitting technique's value
   - The chunk must be evenly divisible by the splitting technique's value
   - We want to maximize the number of splits, not necessarily the number of final chunks
5. Time complexity: O(T * K * log S) where T is number of test cases, K is number of techniques, and S is the starting number of calculations
6. Space complexity: O(log S) for the recursion stack

## Key Insights
1. The problem is about maximizing the number of splits, not the final number of chunks
2. For each chunk, we need to try all possible splitting techniques and choose the one that leads to the most splits
3. The splitting must be done evenly (chunk must be divisible by the splitting technique)
4. We can use recursion to try all possible splitting paths
5. We need to handle large numbers (up to 10^12) carefully 