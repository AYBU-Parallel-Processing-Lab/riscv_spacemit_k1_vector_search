/* Autovectorized baselines under pgvector's floating-point flags.
 * This translation unit is compiled with
 *   -ftree-vectorize -fassociative-math -fno-signed-zeros
 *   -fno-trapping-math -ffp-contract=fast
 * (see Makefile), matching pgvector's build, so the compiler may
 * reassociate reductions. Loop bodies are identical to the strict
 * baselines in kernels.c. */
#include <stddef.h>
#include <math.h>

float l2sq_autovec_pgv(const float *a, const float *b, size_t n) {
    float acc = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = a[i] - b[i];
        acc += d * d;
    }
    return acc;
}

float dot_autovec_pgv(const float *a, const float *b, size_t n) {
    float acc = 0.0f;
    for (size_t i = 0; i < n; i++)
        acc += a[i] * b[i];
    return acc;
}

float l1_autovec_pgv(const float *a, const float *b, size_t n) {
    float acc = 0.0f;
    for (size_t i = 0; i < n; i++)
        acc += fabsf(a[i] - b[i]);
    return acc;
}

float cos_autovec_pgv(const float *a, const float *b, size_t n) {
    float dp = 0.0f, na = 0.0f, nb = 0.0f;
    for (size_t i = 0; i < n; i++) {
        dp += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    return 1.0f - dp / sqrtf(na * nb);
}
