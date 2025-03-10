def solve(n: int, m: int, costs: list[int]) -> int:
    """
    Solve the deteriorating conditions problem.
    
    Args:
        n: Number of fights
        m: Number of armor pieces
        costs: List of base costs for each fight
        
    Returns:
        Minimum total cost for all fights
    """
    # Sort costs in descending order to minimize impact of multipliers
    costs.sort(reverse=True)
    
    # Initialize total cost
    total_cost = 0
    
    # For each fight, assign it to the armor piece that will minimize the total cost
    for i in range(n):
        # Calculate which armor piece this fight should use (0-based)
        armor_index = i % m
        # Calculate the multiplier (1-based)
        multiplier = (i // m) + 1
        # Add the cost with multiplier
        total_cost += costs[i] * multiplier
    
    return total_cost

def main():
    import sys
    input_data = sys.stdin.read().split()
    
    # Parse input
    n = int(input_data[0])
    m = int(input_data[1])
    costs = [int(x) for x in input_data[2:2+n]]
    
    # Solve the problem
    result = solve(n, m, costs)
    
    # Print output
    print(result)

if __name__ == "__main__":
    main() 