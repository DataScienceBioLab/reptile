#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MOD 1000000009
#define MAX_MONSTERS 1000
#define MAX_SIZE (1 << 20)  // 2^20 for polynomial size

// Fast modulo multiplication to avoid overflow
static inline long long mul_mod(long long a, long long b) {
    return (a * b) % MOD;
}

// Fast polynomial multiplication using Karatsuba algorithm
void karatsuba_multiply(const long long* a, const long long* b, long long* result, int n) {
    if (n <= 32) {  // Use naive multiplication for small polynomials
        memset(result, 0, (2 * n - 1) * sizeof(long long));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                result[i + j] = (result[i + j] + mul_mod(a[i], b[j])) % MOD;
            }
        }
        return;
    }
    
    int mid = n / 2;
    
    // Allocate temporary arrays on stack for better cache locality
    long long a0[MAX_SIZE/2], a1[MAX_SIZE/2];
    long long b0[MAX_SIZE/2], b1[MAX_SIZE/2];
    long long z0[MAX_SIZE], z1[MAX_SIZE], z2[MAX_SIZE];
    
    // Split polynomials
    memcpy(a0, a, mid * sizeof(long long));
    memcpy(a1, a + mid, (n - mid) * sizeof(long long));
    memcpy(b0, b, mid * sizeof(long long));
    memcpy(b1, b + mid, (n - mid) * sizeof(long long));
    
    // Recursive calls
    karatsuba_multiply(a0, b0, z0, mid);
    karatsuba_multiply(a1, b1, z2, n - mid);
    
    // Compute (a0 + a1)(b0 + b1)
    for (int i = 0; i < mid; i++) {
        a0[i] = (a0[i] + a1[i]) % MOD;
        b0[i] = (b0[i] + b1[i]) % MOD;
    }
    karatsuba_multiply(a0, b0, z1, mid);
    
    // Combine results
    memset(result, 0, (2 * n - 1) * sizeof(long long));
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
void poly_pow(const long long* poly, long long* result, int n, int power) {
    if (power == 0) {
        memset(result, 0, n * sizeof(long long));
        result[0] = 1;
        return;
    }
    if (power == 1) {
        memcpy(result, poly, n * sizeof(long long));
        return;
    }
    
    // Allocate temporary array for intermediate results
    long long temp[MAX_SIZE];
    
    // Binary lifting
    poly_pow(poly, temp, n, power / 2);
    karatsuba_multiply(temp, temp, result, n);
    
    if (power % 2) {
        memcpy(temp, result, n * sizeof(long long));
        karatsuba_multiply(temp, poly, result, n);
    }
}

long long solve(int g, long long n_total, const int* monsters) {
    // Find maximum monster value
    int max_monster = 0;
    for (int i = 0; i < g; i++) {
        if (monsters[i] > max_monster) {
            max_monster = monsters[i];
        }
    }
    
    // Create polynomial coefficients
    long long coeffs[MAX_SIZE] = {0};
    for (int i = 0; i < g; i++) {
        coeffs[monsters[i]] = 1;
    }
    
    // Calculate result
    long long result[MAX_SIZE];
    poly_pow(coeffs, result, max_monster + 1, n_total);
    
    return n_total < MAX_SIZE ? result[n_total] : 0;
}

int main() {
    int t;
    scanf("%d", &t);
    
    while (t--) {
        int g;
        long long n_total;
        scanf("%d %lld", &g, &n_total);
        
        int monsters[MAX_MONSTERS];
        for (int i = 0; i < g; i++) {
            scanf("%d", &monsters[i]);
        }
        
        printf("%lld\n", solve(g, n_total, monsters));
    }
    
    return 0;
} 