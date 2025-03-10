def solve(n: int, monsters: list[int]) -> int:
    """
    Solve the monstrous power problem.
    
    Args:
        n: Number of wilderness areas
        monsters: List of number of monsters in each area
        
    Returns:
        Maximum total number of monsters possible without using adjacent areas
    """
    # Handle base cases
    if n == 0:
        return 0
    if n == 1:
        return monsters[0]
    if n == 2:
        return max(monsters[0], monsters[1])
    
    # Initialize DP array
    dp = [0] * n
    
    # Base cases
    dp[0] = monsters[0]
    dp[1] = max(monsters[0], monsters[1])
    
    # Fill DP array
    for i in range(2, n):
        # For each position, we can either:
        # 1. Include current area and skip previous (dp[i-2] + monsters[i])
        # 2. Skip current area and use previous (dp[i-1])
        dp[i] = max(dp[i-1], dp[i-2] + monsters[i])
    
    return dp[n-1]

def main():
    import sys
    input_data = sys.stdin.read().split()
    
    # Parse input
    n = int(input_data[0])
    monsters = [int(x) for x in input_data[1:1+n]]
    
    # Solve the problem
    result = solve(n, monsters)
    
    # Print output
    print(result)

if __name__ == "__main__":
    main() 