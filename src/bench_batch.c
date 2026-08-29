/* IVFFLAT-style centroid assignment benchmark: nq query vectors, each
 * compared against nc centroids. Row-major baselines (scalar / rvv_m4 from
 * the main kernel table) vs batched transposed kernels (lane v accumulates
 * the distance to centroid v; no reduction). Layout transform is done once,
 * outside the timed region, matching index-build-time cost. Single thread.
 * CSV on stdout. */
#include "kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define BLOCKS 7
#define TARGET_BLOCK_S 0.2
#define NQ 256

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static float frand(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (float)((int64_t)(rng_state % 2000001) - 1000000) / 1000000.0f;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* blocks of lane_B centroids, dimension-major inside each block */
static void transpose(const float *C, float *T, size_t nc, size_t dim,
                      size_t lane_B) {
    for (size_t blk = 0; blk < nc / lane_B; blk++)
        for (size_t i = 0; i < dim; i++)
            for (size_t v = 0; v < lane_B; v++)
                T[blk * dim * lane_B + i * lane_B + v] =
                    C[(blk * lane_B + v) * dim + i];
}

static dist_fn find_dist(const char *name) {
    for (int k = 0; k < N_KERNELS; k++)
        if (!strcmp(KERNELS[k].op, "l2sq") && !strcmp(KERNELS[k].name, name))
            return KERNELS[k].fn;
    return NULL;
}

typedef struct { const char *name; int is_batch; dist_fn dist; batch_desc *bd; } variant;

int main(void) {
    batch_init();
    size_t dims[] = {128, 384, 768, 1536};
    size_t ncs[] = {64, 256, 1024};

    variant vars[3 + 16];
    int nv = 0;
    vars[nv++] = (variant){"row_scalar", 0, find_dist("scalar"), NULL};
    vars[nv++] = (variant){"row_rvv_m4", 0, find_dist("rvv_m4"), NULL};
    vars[nv++] = (variant){"row_rvv_m8", 0, find_dist("rvv_m8"), NULL};
    for (int i = 0; i < N_BATCH_KERNELS; i++)
        vars[nv++] = (variant){BATCH_KERNELS[i].name, 1, NULL, &BATCH_KERNELS[i]};

    printf("kernel,dim,nc,reps,blocks,median_s,pairs_per_s,gflops,checksum\n");

    for (int d = 0; d < 4; d++) {
        size_t dim = dims[d];
        for (int c = 0; c < 3; c++) {
            size_t nc = ncs[c];
            float *C = aligned_alloc(64, nc * dim * sizeof(float));
            float *T = aligned_alloc(64, nc * dim * sizeof(float));
            float *Q = aligned_alloc(64, NQ * dim * sizeof(float));
            float *out = aligned_alloc(64, nc * sizeof(float));
            float *ref = aligned_alloc(64, nc * sizeof(float));
            for (size_t i = 0; i < nc * dim; i++) C[i] = frand();
            for (size_t i = 0; i < NQ * dim; i++) Q[i] = frand();

            dist_fn sc = find_dist("scalar");
            for (size_t i = 0; i < nc; i++) ref[i] = sc(Q, C + i * dim, dim);

            for (int v = 0; v < nv; v++) {
                variant *V = &vars[v];
                size_t lane_B = V->is_batch ? V->bd->lane_B : 0;
                size_t out_W = V->is_batch ? V->bd->out_W : 0;
                if (V->is_batch) {
                    if (nc % out_W) continue;
                    transpose(C, T, nc, dim, lane_B);
                }

                /* correctness vs scalar row-major on query 0 */
                memset(out, 0, nc * sizeof(float));
                if (V->is_batch) {
                    for (size_t o = 0; o < nc; o += out_W)
                        V->bd->fn(Q, T + o * dim, dim, out + o);
                } else {
                    for (size_t i = 0; i < nc; i++)
                        out[i] = V->dist(Q, C + i * dim, dim);
                }
                for (size_t i = 0; i < nc; i++) {
                    if (fabsf(out[i] - ref[i]) / (ref[i] + 1e-9f) > 1e-4f) {
                        fprintf(stderr, "CORRECTNESS FAIL %s dim=%zu nc=%zu i=%zu\n",
                                V->name, dim, nc, i);
                        return 1;
                    }
                }

                /* one pass = all NQ queries x all nc centroids */
                double t1, checksum = 0;
                int reps = 1;
                for (int cal = 0; cal < 2; cal++) {
                    double t0 = now_s();
                    for (int r = 0; r < reps; r++)
                        for (int qi = 0; qi < NQ; qi++) {
                            const float *q = Q + (size_t)qi * dim;
                            if (V->is_batch)
                                for (size_t o = 0; o < nc; o += out_W)
                                    V->bd->fn(q, T + o * dim, dim, out + o);
                            else
                                for (size_t i = 0; i < nc; i++)
                                    out[i] = V->dist(q, C + i * dim, dim);
                        }
                    t1 = (now_s() - t0) / reps;
                    reps = (int)(TARGET_BLOCK_S / t1);
                    if (reps < 1) reps = 1;
                }

                double times[BLOCKS];
                for (int b = 0; b < BLOCKS; b++) {
                    double t0 = now_s();
                    for (int r = 0; r < reps; r++)
                        for (int qi = 0; qi < NQ; qi++) {
                            const float *q = Q + (size_t)qi * dim;
                            if (V->is_batch)
                                for (size_t o = 0; o < nc; o += out_W)
                                    V->bd->fn(q, T + o * dim, dim, out + o);
                            else
                                for (size_t i = 0; i < nc; i++)
                                    out[i] = V->dist(q, C + i * dim, dim);
                        }
                    times[b] = now_s() - t0;
                    checksum += out[0] + out[nc - 1];
                }
                qsort(times, BLOCKS, sizeof(double), cmp_double);
                double per_pass = times[BLOCKS / 2] / reps;
                double pairs = (double)NQ * nc / per_pass;
                double gflops = 3.0 * (double)NQ * nc * dim / per_pass / 1e9;

                printf("%s,%zu,%zu,%d,%d,%.6e,%.6e,%.4f,%.6e\n",
                       V->name, dim, nc, reps, BLOCKS, times[BLOCKS / 2],
                       pairs, gflops, checksum);
                fflush(stdout);
            }
            free(C); free(T); free(Q); free(out); free(ref);
        }
    }
    return 0;
}
