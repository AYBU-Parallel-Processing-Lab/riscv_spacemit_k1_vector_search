#include "kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

static uint64_t rng_state = 0x243F6A8885A308D3ull;
static float frand(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (float)((int64_t)(rng_state % 2000001) - 1000000) / 1000000.0f;
}

/* Float64 reference. `scale` returns the magnitude the tolerance is relative
 * to: the sum of absolute term magnitudes, so ops with cancellation (dot)
 * are not judged against a near-zero result. */
static double ref_op(const char *op, const float *a, const float *b,
                     size_t n, double *scale) {
    double acc = 0, sabs = 0, dp = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; i++) {
        double x = a[i], y = b[i];
        if (!strcmp(op, "l2sq")) { acc += (x - y) * (x - y); sabs = acc; }
        else if (!strcmp(op, "l1")) { acc += fabs(x - y); sabs = acc; }
        else if (!strcmp(op, "dot")) { acc += x * y; sabs += fabs(x * y); }
        else { dp += x * y; na += x * x; nb += y * y; }
    }
    if (!strcmp(op, "cos")) { *scale = 1.0; return 1.0 - dp / sqrt(na * nb); }
    *scale = sabs > 1e-12 ? sabs : 1.0;
    return acc;
}

int main(void) {
    size_t dims[] = {1, 3, 7, 8, 9, 15, 63, 64, 65, 100, 127, 128, 129, 200,
                     256, 384, 512, 768, 1024, 1536, 2048, 4096, 8192};
    int n_dims = sizeof(dims) / sizeof(dims[0]);
    int failures = 0, checked = 0;

    for (int d = 0; d < n_dims; d++) {
        size_t n = dims[d];
        float *a = aligned_alloc(64, ((n * 4 + 63) / 64) * 64);
        float *b = aligned_alloc(64, ((n * 4 + 63) / 64) * 64);
        for (size_t i = 0; i < n; i++) { a[i] = frand(); b[i] = frand(); }

        for (int k = 0; k < N_KERNELS; k++) {
            double scale;
            double ref = ref_op(KERNELS[k].op, a, b, n, &scale);
            float got = KERNELS[k].fn(a, b, n);
            if (fabs((double)got - ref) > 1e-6 + 1e-5 * scale) {
                printf("FAIL op=%s kernel=%s dim=%zu got=%.9g ref=%.9g\n",
                       KERNELS[k].op, KERNELS[k].name, n, (double)got, ref);
                failures++;
            }
            checked++;
        }
        free(a); free(b);
    }

    if (failures == 0) {
        printf("PASS: %d op-kernel-dim combinations within tol 1e-5 of float64 reference\n",
               checked);
        return 0;
    }
    printf("%d failures\n", failures);
    return 1;
}
