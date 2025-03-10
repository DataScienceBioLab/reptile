#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>

constexpr int MOD = 1000000009;
constexpr int MAX_MONSTERS = 1000;
constexpr int MAX_SIZE = 1 << 20;  // 2^20 for polynomial size

// Fast modulo multiplication to avoid overflow
inline long long mul_mod(long long a, long long b) {
    return (a * b) % MOD;
}

// Fast polynomial multiplication using Karatsuba algorithm
void karatsuba_multiply(const std::vector<long long>& a, const std::vector<long long>& b, 
                       std::vector<long long>& result) {
    int n = a.size();
    if (n <= 32) {  // Use naive multiplication for small polynomials
        std::fill(result.begin(), result.end(), 0);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                result[i + j] = (result[i + j] + mul_mod(a[i], b[j])) % MOD;
            }
        }
        return;
    }
    
    int mid = n / 2;
    
    // Split polynomials
    std::vector<long long> a0(a.begin(), a.begin() + mid);
    std::vector<long long> a1(a.begin() + mid, a.end());
    std::vector<long long> b0(b.begin(), b.begin() + mid);
    std::vector<long long> b1(b.begin() + mid, b.end());
    
    // Temporary vectors for results
    std::vector<long long> z0(2 * mid - 1);
    std::vector<long long> z1(2 * mid - 1);
    std::vector<long long> z2(2 * (n - mid) - 1);
    
    // Recursive calls
    karatsuba_multiply(a0, b0, z0);
    karatsuba_multiply(a1, b1, z2);
    
    // Compute (a0 + a1)(b0 + b1)
    for (int i = 0; i < mid; i++) {
        a0[i] = (a0[i] + a1[i]) % MOD;
        b0[i] = (b0[i] + b1[i]) % MOD;
    }
    karatsuba_multiply(a0, b0, z1);
    
    // Combine results
    std::fill(result.begin(), result.end(), 0);
    for (int i = 0; i < 2 * mid - 1; i++) {
        result[i] = z0[i];
    }
    for (int i = 0; i < 2 * (n - mid) - 1; i++) {
        result[i + 2 * mid] = z2[i];
    }
    for (int i = 0; i < 2 * mid - 1; i++) {
        result[i + mid] = (result[i + mid] + z1[i] - z0[i] - z2[i] + 3 * MOD) % MOD;
    }
}

// Fast polynomial exponentiation
void poly_pow(const std::vector<long long>& poly, std::vector<long long>& result, int power) {
    if (power == 0) {
        std::fill(result.begin(), result.end(), 0);
        result[0] = 1;
        return;
    }
    if (power == 1) {
        std::copy(poly.begin(), poly.end(), result.begin());
        return;
    }
    
    // Binary lifting
    poly_pow(poly, result, power / 2);
    std::vector<long long> temp(result.size());
    karatsuba_multiply(result, result, temp);
    std::copy(temp.begin(), temp.end(), result.begin());
    
    if (power % 2) {
        std::copy(result.begin(), result.end(), temp.begin());
        karatsuba_multiply(temp, poly, result);
    }
}

long long solve(int g, long long n_total, const std::vector<int>& monsters) {
    // Find maximum monster value
    int max_monster = *std::max_element(monsters.begin(), monsters.end());
    
    // Create polynomial coefficients
    std::vector<long long> coeffs(max_monster + 1, 0);
    for (int monster : monsters) {
        coeffs[monster] = 1;
    }
    
    // Calculate result
    std::vector<long long> result(2 * max_monster + 1);
    poly_pow(coeffs, result, n_total);
    
    return n_total < result.size() ? result[n_total] : 0;
}

int main() {
    std::ios_base::sync_with_stdio(false);  // Optimize I/O
    std::cin.tie(nullptr);
    
    int t;
    std::cin >> t;
    
    while (t--) {
        int g;
        long long n_total;
        std::cin >> g >> n_total;
        
        std::vector<int> monsters(g);
        for (int i = 0; i < g; i++) {
            std::cin >> monsters[i];
        }
        
        std::cout << solve(g, n_total, monsters) << '\n';
    }
    
    return 0;
}