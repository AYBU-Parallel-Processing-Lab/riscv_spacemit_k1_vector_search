#ifndef KERNELS_H
#define KERNELS_H

#include <stddef.h>

typedef float (*dist_fn)(const float *a, const float *b, size_t n);

typedef struct {
    const char *op;     /* l2sq | dot | cos | l1 */
    const char *name;   /* scalar | autovec | rvv_m1..m8 */
    dist_fn fn;
    double flops_per_elem;
} kernel_desc;

extern const kernel_desc KERNELS[];
extern const int N_KERNELS;

/* kernels_pgv.c: same naive loops compiled with pgvector's FP flags */
float l2sq_autovec_pgv(const float *a, const float *b, size_t n);
float dot_autovec_pgv(const float *a, const float *b, size_t n);
float l1_autovec_pgv(const float *a, const float *b, size_t n);
float cos_autovec_pgv(const float *a, const float *b, size_t n);

/* Batched transposed kernels: one query vs a block of B vectors stored
 * dimension-major (blk[i*B + v] = vector v, dimension i), B = vlmax of the
 * LMUL variant. Writes B squared-L2 distances to out; no reduction needed. */
typedef void (*batch_fn)(const float *q, const float *blk, size_t dim, float *out);

typedef struct {
    const char *name;
    batch_fn fn;
    size_t lane_B;      /* vlmax of the variant = lanes per block (layout unit) */
    size_t out_W;       /* results written per call = lane_B * unroll factor */
} batch_desc;

extern batch_desc BATCH_KERNELS[];
extern const int N_BATCH_KERNELS;
void batch_init(void);

#endif
