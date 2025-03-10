def solve(s: int, k: int, techniques: list[int]) -> int:
    """
    Solve the Lickety-Split problem.
    
    Args:
        s: Starting number of calculations
        k: Number of splitting techniques
        techniques: List of splitting technique values
        
    Returns:
        Maximum number of splits possible
    """
    def max_splits(chunk_size: int) -> int:
        # Base case: if chunk is too small to split
        if chunk_size <= 1:
            return 0
            
        max_splits_count = 0
        
        # Try each splitting technique
        for technique in techniques:
            # Check if we can use this technique
            if chunk_size > technique and chunk_size % technique == 0:
                # Calculate number of splits for this technique
                # 1 split for current split + splits for each resulting chunk
                splits = 1 + (chunk_size // technique) * max_splits(technique)
                max_splits_count = max(max_splits_count, splits)
        
        return max_splits_count
    
    return max_splits(s)

def main():
    import sys
    input_data = sys.stdin.read().split()
    
    # Parse input
    t = int(input_data[0])
    current_index = 1
    
    # Process each test case
    for _ in range(t):
        s = int(input_data[current_index])
        k = int(input_data[current_index + 1])
        techniques = [int(x) for x in input_data[current_index + 2:current_index + 2 + k]]
        
        # Solve the problem
        result = solve(s, k, techniques)
        
        # Print output
        print(result)
        
        # Update index for next test case
        current_index += 2 + k

if __name__ == "__main__":
    main() 