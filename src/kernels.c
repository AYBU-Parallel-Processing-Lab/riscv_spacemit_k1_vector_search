#include "kernels.h"
#include <riscv_vector.h>
#include <math.h>

/* ---------------- scalar / autovec baselines ----------------
 * GCC at -O3 with V enabled autovectorizes these loops (LMUL=1 strip +
 * per-strip ordered vfredosum), so the true-scalar variants disable
 * vectorization explicitly and the autovec variants leave it on. */

#ifdef __clang__
#define NOVEC_ATTR
#define NOVEC_LOOP _Pragma("clang loop vectorize(disable) interleave(disable)")
#else
#define NOVEC_ATTR __attribute__((optimize("no-tree-vectorize")))
#define NOVEC_LOOP
#endif

#define DEF_BASELINES(OP, INIT, STEP, RET)                                    \
NOVEC_ATTR                                                                    \
static float OP##_scalar(const float *a, const float *b, size_t n) {         \
    INIT;                                                                     \
    NOVEC_LOOP                                                                \
    for (size_t i = 0; i < n; i++) { STEP; }                                  \
    return RET;                                                               \
}                                                                             \
static float OP##_autovec(const float *a, const float *b, size_t n) {        \
    INIT;                                                                     \
    for (size_t i = 0; i < n; i++) { STEP; }                                  \
    return RET;                                                               \
}

DEF_BASELINES(l2sq, float acc = 0.0f,
              float d = a[i] - b[i]; acc += d * d, acc)
DEF_BASELINES(dot, float acc = 0.0f,
              acc += a[i] * b[i], acc)
DEF_BASELINES(l1, float acc = 0.0f,
              acc += fabsf(a[i] - b[i]), acc)
DEF_BASELINES(cos, float dp = 0.0f; float na = 0.0f; float nb = 0.0f,
              dp += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i],
              1.0f - dp / sqrtf(na * nb))

/* ---------------- RVV kernels, one set per LMUL ----------------
 * Persistent vector accumulators with tail-undisturbed (_tu) updates so a
 * short final strip cannot clobber accumulated lanes; single reduction at
 * the end over vlmax. */

#define DEF_RVV_OPS(LMUL)                                                     \
static float reduce_f32m##LMUL(vfloat32m##LMUL##_t acc, size_t vlmax) {       \
    vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);                      \
    vfloat32m1_t red = __riscv_vfredusum_vs_f32m##LMUL##_f32m1(acc, zero, vlmax); \
    return __riscv_vfmv_f_s_f32m1_f32(red);                                   \
}                                                                             \
static float l2sq_rvv_m##LMUL(const float *a, const float *b, size_t n) {     \
    size_t vlmax = __riscv_vsetvlmax_e32m##LMUL();                            \
    vfloat32m##LMUL##_t acc = __riscv_vfmv_v_f_f32m##LMUL(0.0f, vlmax);       \
    for (size_t i = 0; i < n;) {                                              \
        size_t vl = __riscv_vsetvl_e32m##LMUL(n - i);                         \
        vfloat32m##LMUL##_t va = __riscv_vle32_v_f32m##LMUL(a + i, vl);       \
        vfloat32m##LMUL##_t vb = __riscv_vle32_v_f32m##LMUL(b + i, vl);       \
        vfloat32m##LMUL##_t vd = __riscv_vfsub_vv_f32m##LMUL(va, vb, vl);     \
        acc = __riscv_vfmacc_vv_f32m##LMUL##_tu(acc, vd, vd, vl);             \
        i += vl;                                                              \
    }                                                                         \
    return reduce_f32m##LMUL(acc, vlmax);                                     \
}                                                                             \
static float dot_rvv_m##LMUL(const float *a, const float *b, size_t n) {      \
    size_t vlmax = __riscv_vsetvlmax_e32m##LMUL();                            \
    vfloat32m##LMUL##_t acc = __riscv_vfmv_v_f_f32m##LMUL(0.0f, vlmax);       \
    for (size_t i = 0; i < n;) {                                              \
        size_t vl = __riscv_vsetvl_e32m##LMUL(n - i);                         \
        vfloat32m##LMUL##_t va = __riscv_vle32_v_f32m##LMUL(a + i, vl);       \
        vfloat32m##LMUL##_t vb = __riscv_vle32_v_f32m##LMUL(b + i, vl);       \
        acc = __riscv_vfmacc_vv_f32m##LMUL##_tu(acc, va, vb, vl);             \
        i += vl;                                                              \
    }                                                                         \
    return reduce_f32m##LMUL(acc, vlmax);                                     \
}                                                                             \
static float l1_rvv_m##LMUL(const float *a, const float *b, size_t n) {       \
    size_t vlmax = __riscv_vsetvlmax_e32m##LMUL();                            \
    vfloat32m##LMUL##_t acc = __riscv_vfmv_v_f_f32m##LMUL(0.0f, vlmax);       \
    for (size_t i = 0; i < n;) {                                              \
        size_t vl = __riscv_vsetvl_e32m##LMUL(n - i);                         \
        vfloat32m##LMUL##_t va = __riscv_vle32_v_f32m##LMUL(a + i, vl);       \
        vfloat32m##LMUL##_t vb = __riscv_vle32_v_f32m##LMUL(b + i, vl);       \
        vfloat32m##LMUL##_t vd = __riscv_vfsub_vv_f32m##LMUL(va, vb, vl);     \
        vfloat32m##LMUL##_t vabs = __riscv_vfsgnjx_vv_f32m##LMUL(vd, vd, vl); \
        acc = __riscv_vfadd_vv_f32m##LMUL##_tu(acc, acc, vabs, vl);           \
        i += vl;                                                              \
    }                                                                         \
    return reduce_f32m##LMUL(acc, vlmax);                                     \
}                                                                             \
static float cos_rvv_m##LMUL(const float *a, const float *b, size_t n) {      \
    size_t vlmax = __riscv_vsetvlmax_e32m##LMUL();                            \
    vfloat32m##LMUL##_t dp = __riscv_vfmv_v_f_f32m##LMUL(0.0f, vlmax);        \
    vfloat32m##LMUL##_t na = dp, nb = dp;                                     \
    for (size_t i = 0; i < n;) {                                              \
        size_t vl = __riscv_vsetvl_e32m##LMUL(n - i);                         \
        vfloat32m##LMUL##_t va = __riscv_vle32_v_f32m##LMUL(a + i, vl);       \
        vfloat32m##LMUL##_t vb = __riscv_vle32_v_f32m##LMUL(b + i, vl);       \
        dp = __riscv_vfmacc_vv_f32m##LMUL##_tu(dp, va, vb, vl);               \
        na = __riscv_vfmacc_vv_f32m##LMUL##_tu(na, va, va, vl);               \
        nb = __riscv_vfmacc_vv_f32m##LMUL##_tu(nb, vb, vb, vl);               \
        i += vl;                                                              \
    }                                                                         \
    float d = reduce_f32m##LMUL(dp, vlmax);                                   \
    float xa = reduce_f32m##LMUL(na, vlmax);                                  \
    float xb = reduce_f32m##LMUL(nb, vlmax);                                  \
    return 1.0f - d / sqrtf(xa * xb);                                         \
}

DEF_RVV_OPS(1)
DEF_RVV_OPS(2)
DEF_RVV_OPS(4)
DEF_RVV_OPS(8)

#define OP_ROW(OP, FLOPS)                                                     \
    {#OP, "scalar", OP##_scalar, FLOPS},                                      \
    {#OP, "autovec", OP##_autovec, FLOPS},                                    \
    {#OP, "autovec_pgv", OP##_autovec_pgv, FLOPS},                            \
    {#OP, "rvv_m1", OP##_rvv_m1, FLOPS},                                      \
    {#OP, "rvv_m2", OP##_rvv_m2, FLOPS},                                      \
    {#OP, "rvv_m4", OP##_rvv_m4, FLOPS},                                      \
    {#OP, "rvv_m8", OP##_rvv_m8, FLOPS},

const kernel_desc KERNELS[] = {
    OP_ROW(l2sq, 3.0)
    OP_ROW(dot, 2.0)
    OP_ROW(l1, 3.0)
    OP_ROW(cos, 6.0)
};
const int N_KERNELS = sizeof(KERNELS) / sizeof(KERNELS[0]);

/* ---------------- batched transposed squared-L2 ----------------
 * blk holds B vectors dimension-major; lane v accumulates the distance of
 * vector v. No reduction: results stored directly with vse32. The u2/u4
 * variants process 2/4 blocks with independent accumulators to break the
 * vfmacc dependency chain (one chained vfmacc per dimension otherwise). */

#define DEF_BATCH(LMUL)                                                       \
static void batch_l2sq_m##LMUL(const float *q, const float *blk,              \
                               size_t dim, float *out) {                      \
    size_t B = __riscv_vsetvlmax_e32m##LMUL();                                \
    vfloat32m##LMUL##_t acc = __riscv_vfmv_v_f_f32m##LMUL(0.0f, B);           \
    for (size_t i = 0; i < dim; i++) {                                        \
        vfloat32m##LMUL##_t vr = __riscv_vle32_v_f32m##LMUL(blk + i * B, B);  \
        vfloat32m##LMUL##_t vd = __riscv_vfsub_vf_f32m##LMUL(vr, q[i], B);    \
        acc = __riscv_vfmacc_vv_f32m##LMUL(acc, vd, vd, B);                   \
    }                                                                         \
    __riscv_vse32_v_f32m##LMUL(out, acc, B);                                  \
}

DEF_BATCH(1)
DEF_BATCH(2)
DEF_BATCH(4)
DEF_BATCH(8)

#define DEF_BATCH_U2(LMUL)                                                    \
static void batch_l2sq_m##LMUL##u2(const float *q, const float *blk,          \
                                   size_t dim, float *out) {                  \
    size_t B = __riscv_vsetvlmax_e32m##LMUL();                                \
    const float *b0 = blk, *b1 = blk + dim * B;                               \
    vfloat32m##LMUL##_t a0 = __riscv_vfmv_v_f_f32m##LMUL(0.0f, B), a1 = a0;   \
    for (size_t i = 0; i < dim; i++) {                                        \
        float qi = q[i];                                                      \
        vfloat32m##LMUL##_t r0 = __riscv_vle32_v_f32m##LMUL(b0 + i * B, B);   \
        vfloat32m##LMUL##_t r1 = __riscv_vle32_v_f32m##LMUL(b1 + i * B, B);   \
        vfloat32m##LMUL##_t d0 = __riscv_vfsub_vf_f32m##LMUL(r0, qi, B);      \
        vfloat32m##LMUL##_t d1 = __riscv_vfsub_vf_f32m##LMUL(r1, qi, B);      \
        a0 = __riscv_vfmacc_vv_f32m##LMUL(a0, d0, d0, B);                     \
        a1 = __riscv_vfmacc_vv_f32m##LMUL(a1, d1, d1, B);                     \
    }                                                                         \
    __riscv_vse32_v_f32m##LMUL(out, a0, B);                                   \
    __riscv_vse32_v_f32m##LMUL(out + B, a1, B);                               \
}

DEF_BATCH_U2(1)
DEF_BATCH_U2(2)
DEF_BATCH_U2(4)

static void batch_l2sq_m2u4(const float *q, const float *blk,
                            size_t dim, float *out) {
    size_t B = __riscv_vsetvlmax_e32m2();
    const float *b0 = blk, *b1 = blk + dim * B,
                *b2 = blk + 2 * dim * B, *b3 = blk + 3 * dim * B;
    vfloat32m2_t a0 = __riscv_vfmv_v_f_f32m2(0.0f, B), a1 = a0, a2 = a0, a3 = a0;
    for (size_t i = 0; i < dim; i++) {
        float qi = q[i];
        vfloat32m2_t d0 = __riscv_vfsub_vf_f32m2(__riscv_vle32_v_f32m2(b0 + i * B, B), qi, B);
        vfloat32m2_t d1 = __riscv_vfsub_vf_f32m2(__riscv_vle32_v_f32m2(b1 + i * B, B), qi, B);
        vfloat32m2_t d2 = __riscv_vfsub_vf_f32m2(__riscv_vle32_v_f32m2(b2 + i * B, B), qi, B);
        vfloat32m2_t d3 = __riscv_vfsub_vf_f32m2(__riscv_vle32_v_f32m2(b3 + i * B, B), qi, B);
        a0 = __riscv_vfmacc_vv_f32m2(a0, d0, d0, B);
        a1 = __riscv_vfmacc_vv_f32m2(a1, d1, d1, B);
        a2 = __riscv_vfmacc_vv_f32m2(a2, d2, d2, B);
        a3 = __riscv_vfmacc_vv_f32m2(a3, d3, d3, B);
    }
    __riscv_vse32_v_f32m2(out, a0, B);
    __riscv_vse32_v_f32m2(out + B, a1, B);
    __riscv_vse32_v_f32m2(out + 2 * B, a2, B);
    __riscv_vse32_v_f32m2(out + 3 * B, a3, B);
}

batch_desc BATCH_KERNELS[] = {
    {"batch_m1", batch_l2sq_m1, 1, 1},
    {"batch_m2", batch_l2sq_m2, 2, 2},
    {"batch_m4", batch_l2sq_m4, 4, 4},
    {"batch_m8", batch_l2sq_m8, 8, 8},
    {"batch_m1u2", batch_l2sq_m1u2, 1, 2},
    {"batch_m2u2", batch_l2sq_m2u2, 2, 4},
    {"batch_m4u2", batch_l2sq_m4u2, 4, 8},
    {"batch_m2u4", batch_l2sq_m2u4, 2, 8},
};
const int N_BATCH_KERNELS = sizeof(BATCH_KERNELS) / sizeof(BATCH_KERNELS[0]);

/* lane_B/out_W are stored as multiples of vlmax(e32,m1) in the table;
 * scale them to element counts for the running VLEN here. */
void batch_init(void) {
    size_t m1 = __riscv_vsetvlmax_e32m1();
    for (int i = 0; i < N_BATCH_KERNELS; i++) {
        BATCH_KERNELS[i].lane_B *= m1;
        BATCH_KERNELS[i].out_W *= m1;
    }
}
