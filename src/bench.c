/* 1-query-vs-N-database-rows squared-L2 benchmark.
 *
 * Regimes:
 *   cache  - database sized to ~CACHE_KB total, repeated many times (ALU/LMUL bound)
 *   stream - database sized to ~STREAM_MB total (memory bound)
 *
 * Threads split database rows; each worker is pinned to core = worker index.
 * Timing: BLOCKS measurement blocks of `reps` passes each, median block time
 * reported. Warmup pass before measurement. Output: CSV on stdout.
 */
#define _GNU_SOURCE
#include "kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <sched.h>

#define BLOCKS 7
#define TARGET_BLOCK_S 0.25
#define MAX_REPS 100000

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static long read_khz(int cpu) {
    char path[128];
    snprintf(path, sizeof(path),
             "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", cpu);
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    long khz = -1;
    if (fscanf(f, "%ld", &khz) != 1) khz = -1;
    fclose(f);
    return khz;
}

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static float frand(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (float)((int64_t)(rng_state % 2000001) - 1000000) / 1000000.0f;
}

/* core maps: compact fills cluster 0 first; spread splits across clusters */
static const int PIN_COMPACT[8] = {0, 1, 2, 3, 4, 5, 6, 7};
static const int PIN_SPREAD[8]  = {0, 4, 1, 5, 2, 6, 3, 7};
static const int *pin_map = PIN_COMPACT;

typedef struct {
    int id;
    dist_fn fn;
    const float *q;
    const float *db;
    float *out;
    size_t dim;
    size_t row_begin, row_end;
    int reps;
    int blocks;
    pthread_barrier_t *go, *done;
} worker_arg;

static void *worker(void *p) {
    worker_arg *w = p;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(pin_map[w->id], &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);

    for (int b = 0; b < w->blocks; b++) {
        pthread_barrier_wait(w->go);
        for (int r = 0; r < w->reps; r++)
            for (size_t i = w->row_begin; i < w->row_end; i++)
                w->out[i] = w->fn(w->q, w->db + i * w->dim, w->dim);
        pthread_barrier_wait(w->done);
    }
    return NULL;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* One timed pass with `reps` repetitions per block, `blocks` blocks.
 * Returns median block time in seconds. */
static double run_config(dist_fn fn, const float *q, const float *db,
                         float *out, size_t dim, size_t nrows, int threads,
                         int reps, int blocks, double *times_out) {
    pthread_barrier_t go, done;
    pthread_barrier_init(&go, NULL, threads + 1);
    pthread_barrier_init(&done, NULL, threads + 1);

    pthread_t tid[8];
    worker_arg wa[8];
    for (int t = 0; t < threads; t++) {
        wa[t] = (worker_arg){
            .id = t, .fn = fn, .q = q, .db = db, .out = out, .dim = dim,
            .row_begin = nrows * t / threads,
            .row_end = nrows * (t + 1) / threads,
            .reps = reps, .blocks = blocks, .go = &go, .done = &done,
        };
        pthread_create(&tid[t], NULL, worker, &wa[t]);
    }

    double times[BLOCKS + 1];
    for (int b = 0; b < blocks; b++) {
        double t0 = now_s();
        pthread_barrier_wait(&go);
        pthread_barrier_wait(&done);
        times[b] = now_s() - t0;
    }
    for (int t = 0; t < threads; t++) pthread_join(tid[t], NULL);
    pthread_barrier_destroy(&go);
    pthread_barrier_destroy(&done);

    if (times_out) memcpy(times_out, times, blocks * sizeof(double));
    qsort(times, blocks, sizeof(double), cmp_double);
    return times[blocks / 2];
}

int main(int argc, char **argv) {
    size_t dims[] = {128, 256, 384, 512, 768, 1024, 1536, 2048, 4096, 8192};
    int thread_counts[] = {1, 2, 4, 8};
    const char *regimes[] = {"cache", "stream"};
    size_t cache_kb = 256, stream_mb = 256, per_thread_kb = 0;
    const char *only_kernel = NULL, *only_regime = NULL, *only_op = "l2sq";
    const char *dims_arg = NULL, *pin_name = "compact", *fvecs_path = NULL;
    int only_threads = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--kernel") && i + 1 < argc) only_kernel = argv[++i];
        else if (!strcmp(argv[i], "--regime") && i + 1 < argc) only_regime = argv[++i];
        else if (!strcmp(argv[i], "--op") && i + 1 < argc) only_op = argv[++i];
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) only_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--stream-mb") && i + 1 < argc) stream_mb = atol(argv[++i]);
        else if (!strcmp(argv[i], "--cache-kb") && i + 1 < argc) cache_kb = atol(argv[++i]);
        else if (!strcmp(argv[i], "--per-thread-kb") && i + 1 < argc) per_thread_kb = atol(argv[++i]);
        else if (!strcmp(argv[i], "--dims") && i + 1 < argc) dims_arg = argv[++i];
        else if (!strcmp(argv[i], "--fvecs") && i + 1 < argc) fvecs_path = argv[++i];
        else if (!strcmp(argv[i], "--pin") && i + 1 < argc) pin_name = argv[++i];
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 1; }
    }
    if (!strcmp(pin_name, "spread")) pin_map = PIN_SPREAD;
    else if (strcmp(pin_name, "compact")) { fprintf(stderr, "bad --pin\n"); return 1; }

    size_t dim_list[16];
    int n_dims = 0;
    if (dims_arg) {
        char buf[256];
        snprintf(buf, sizeof(buf), "%s", dims_arg);
        for (char *tok = strtok(buf, ","); tok && n_dims < 16; tok = strtok(NULL, ","))
            dim_list[n_dims++] = atol(tok);
    } else {
        for (int i = 0; i < (int)(sizeof(dims) / sizeof(dims[0])); i++)
            dim_list[n_dims++] = dims[i];
    }

    printf("regime,op,kernel,dim,threads,nrows,reps,blocks,median_s,min_s,max_s,"
           "vecs_per_s,gflops,db_gbps,checksum,cpu0_khz,cpu4_khz\n");

    size_t max_bytes = stream_mb * 1024 * 1024;
    float *db = aligned_alloc(64, max_bytes);
    float *q = aligned_alloc(64, 8192 * sizeof(float));
    if (!db || !q) { fprintf(stderr, "alloc failed\n"); return 1; }
    for (size_t i = 0; i < max_bytes / 4; i++) db[i] = frand();
    for (size_t i = 0; i < 8192; i++) q[i] = frand();

    /* Real-data validation: load an .fvecs file (int32 dim + dim floats per
     * record), restrict the sweep to that dimension, and fill db and q by
     * cycling through the dataset vectors instead of synthetic ones. */
    if (fvecs_path) {
        FILE *f = fopen(fvecs_path, "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", fvecs_path); return 1; }
        int32_t fdim;
        if (fread(&fdim, 4, 1, f) != 1 || fdim <= 0 || fdim > 8192) {
            fprintf(stderr, "bad fvecs header\n"); return 1;
        }
        fseek(f, 0, SEEK_SET);
        size_t total_floats = max_bytes / 4, pos = 0;
        float *v = malloc((size_t)fdim * 4);
        while (pos + (size_t)fdim <= total_floats) {
            int32_t d2;
            if (fread(&d2, 4, 1, f) != 1) { rewind(f); continue; }
            if (fread(v, 4, fdim, f) != (size_t)fdim) { rewind(f); continue; }
            memcpy(db + pos, v, (size_t)fdim * 4);
            pos += fdim;
        }
        memcpy(q, db, (size_t)fdim * 4);
        free(v);
        fclose(f);
        dim_list[0] = fdim;
        n_dims = 1;
        fprintf(stderr, "fvecs: dim=%d loaded %zu floats\n", fdim, pos);
    }

    for (int rg = 0; rg < 2; rg++) {
        if (only_regime && strcmp(regimes[rg], only_regime)) continue;
        if (per_thread_kb && rg != 0) continue;
        const char *regime_name = per_thread_kb ? "cache_pt" : regimes[rg];

        for (int d = 0; d < n_dims; d++) {
            size_t dim = dim_list[d];
            size_t base_bytes = rg == 0 ? cache_kb * 1024 : max_bytes;
            size_t max_rows = (per_thread_kb ? per_thread_kb * 1024 * 8
                                             : base_bytes) / (dim * sizeof(float));
            if (max_rows < 8) max_rows = 8;
            float *out = aligned_alloc(64, ((max_rows * 4 + 63) / 64) * 64);

            for (int k = 0; k < N_KERNELS; k++) {
                if (only_kernel && strcmp(KERNELS[k].name, only_kernel)) continue;
                if (strcmp(only_op, "all") && strcmp(KERNELS[k].op, only_op)) continue;

                for (int tc = 0; tc < 4; tc++) {
                    int threads = thread_counts[tc];
                    if (only_threads && threads != only_threads) continue;

                    size_t nrows = per_thread_kb
                        ? per_thread_kb * 1024 * threads / (dim * sizeof(float))
                        : (rg == 0 ? cache_kb * 1024 : max_bytes) / (dim * sizeof(float));
                    if (nrows < 8) nrows = 8;
                    if (nrows > max_rows) nrows = max_rows;

                    /* calibrate reps: single warm block of 1 rep */
                    double t1 = run_config(KERNELS[k].fn, q, db, out, dim,
                                           nrows, threads, 1, 1, NULL);
                    int reps = (int)(TARGET_BLOCK_S / t1);
                    if (reps < 1) reps = 1;
                    if (reps > MAX_REPS) reps = MAX_REPS;

                    double times[BLOCKS];
                    double med = run_config(KERNELS[k].fn, q, db, out, dim,
                                            nrows, threads, reps, BLOCKS, times);
                    qsort(times, BLOCKS, sizeof(double), cmp_double);

                    double per_pass = med / reps;
                    double vecs_s = (double)nrows / per_pass;
                    double gflops = KERNELS[k].flops_per_elem * (double)nrows
                                    * (double)dim / per_pass / 1e9;
                    double gbps = (double)nrows * dim * sizeof(float) / per_pass / 1e9;

                    double checksum = 0;
                    for (size_t i = 0; i < nrows; i += nrows / 16 + 1)
                        checksum += out[i];

                    printf("%s,%s,%s,%zu,%d,%zu,%d,%d,%.6e,%.6e,%.6e,"
                           "%.6e,%.4f,%.4f,%.6e,%ld,%ld\n",
                           regime_name, KERNELS[k].op, KERNELS[k].name, dim, threads, nrows,
                           reps, BLOCKS, med, times[0], times[BLOCKS - 1],
                           vecs_s, gflops, gbps, checksum,
                           read_khz(0), read_khz(4));
                    fflush(stdout);
                    fprintf(stderr, "done %s %s dim=%zu t=%d\n",
                            regime_name, KERNELS[k].name, dim, threads);
                }
            }
            free(out);
        }
    }
    free(db);
    free(q);
    return 0;
}
