MOD = 1000000009

def karatsuba_multiply(a: list[int], b: list[int]) -> list[int]:
    """Fast polynomial multiplication using Karatsuba algorithm."""
    n = len(a)
    if n <= 32:  # Use naive multiplication for small polynomials
        result = [0] * (2 * n - 1)
        for i in range(n):
            for j in range(n):
                result[i + j] = (result[i + j] + a[i] * b[j]) % MOD
        return result
    
    # Split polynomials
    mid = n // 2
    a0, a1 = a[:mid], a[mid:]
    b0, b1 = b[:mid], b[mid:]
    
    # Recursive calls
    z0 = karatsuba_multiply(a0, b0)
    z2 = karatsuba_multiply(a1, b1)
    
    # Compute (a0 + a1)(b0 + b1)
    a_sum = [(a0[i] + a1[i]) % MOD for i in range(len(a0))]
    b_sum = [(b0[i] + b1[i]) % MOD for i in range(len(b0))]
    z1 = karatsuba_multiply(a_sum, b_sum)
    
    # Combine results
    result = [0] * (2 * n - 1)
    for i in range(len(z0)):
        result[i] = z0[i]
    for i in range(len(z2)):
        result[i + 2 * mid] = z2[i]
    for i in range(len(z1)):
        result[i + mid] = (result[i + mid] + z1[i] - z0[i] - z2[i]) % MOD
    
    return result

def solve(g: int, n_total: int, monsters: list[int]) -> int:
    """
    Solve the weirdest programming language problem using optimized polynomial multiplication.
    Theoretical optimal solution: O(g * log n_total)
    
    Args:
        g: Number of gate types
        n_total: Total number of monsters needed
        monsters: List of monsters needed for each gate type
        
    Returns:
        Number of possible sequences modulo 1000000009
    """
    # Sort monsters and remove duplicates
    monsters = sorted(set(monsters))
    
    # Create polynomial coefficients
    coeffs = [0] * (max(monsters) + 1)
    for m in monsters:
        coeffs[m] = 1
    
    # Fast polynomial exponentiation
    def poly_pow(poly: list[int], power: int) -> list[int]:
        if power == 0:
            return [1]
        if power == 1:
            return poly
            
        # Binary lifting with Karatsuba multiplication
        half = poly_pow(poly, power // 2)
        result = karatsuba_multiply(half, half)
        
        if power % 2:
            result = karatsuba_multiply(result, poly)
            
        return result
    
    # Calculate result
    result = poly_pow(coeffs, n_total)
    
    # Return the coefficient of x^n_total
    return result[n_total] if n_total < len(result) else 0

def main():
    import sys
    input_data = sys.stdin.read().split()
    
    # Parse input
    t = int(input_data[0])
    current_index = 1
    
    # Process each test case
    for _ in range(t):
        g = int(input_data[current_index])
        n_total = int(input_data[current_index + 1])
        monsters = [int(x) for x in input_data[current_index + 2:current_index + 2 + g]]
        
        # Solve the problem
        result = solve(g, n_total, monsters)
        
        # Print output
        print(result)
        
        # Update index for next test case
        current_index += 2 + g

if __name__ == "__main__":
    main() 