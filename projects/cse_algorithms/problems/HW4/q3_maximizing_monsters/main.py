def solve(n: int, k: int, monsters: list[int]) -> int:
    """
    Solve the maximizing monsters problem.
    
    Args:
        n: Number of logic gate types
        k: Maximum total monsters allowed
        monsters: List of monsters needed for each gate type
        
    Returns:
        Maximum total monsters that can be used without exceeding k
    """
    # Initialize DP array
    dp = [False] * (k + 1)
    dp[0] = True  # Base case: can achieve sum of 0
    
    # For each possible sum up to k
    for i in range(1, k + 1):
        # Try each gate type
        for monsters_needed in monsters:
            if monsters_needed <= i and dp[i - monsters_needed]:
                dp[i] = True
                break
    
    # Find the largest sum that can be achieved
    for i in range(k, -1, -1):
        if dp[i]:
            return i
    
    return 0

def main():
    import sys
    input_data = sys.stdin.read().split()
    
    # Parse input
    t = int(input_data[0])
    current_index = 1
    
    # Process each test case
    for _ in range(t):
        n = int(input_data[current_index])
        k = int(input_data[current_index + 1])
        monsters = [int(x) for x in input_data[current_index + 2:current_index + 2 + n]]
        
        # Solve the problem
        result = solve(n, k, monsters)
        
        # Print output
        print(result)
        
        # Update index for next test case
        current_index += 2 + n

if __name__ == "__main__":
    main() 