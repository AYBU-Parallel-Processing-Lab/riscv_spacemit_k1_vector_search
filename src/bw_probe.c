/* Sustained DRAM read bandwidth probe.
 *
 * Method: a 512 MiB buffer is initialized once, split statically into
 * per-thread contiguous ranges, and read sequentially with unit-stride
 * LMUL=8 vector loads accumulated into an integer vector register so the
 * reads cannot be elided. Threads are pinned according to the chosen core
 * map. Each configuration reports the median of 7 blocks whose repetition
 * count is calibrated to about 0.25 s, same protocol as the kernel
 * benchmark. Output CSV: pin,threads,bytes,reps,median_s,min_s,max_s,gbps.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <sched.h>
#include <riscv_vector.h>

#define BLOCKS 7
#define TARGET_BLOCK_S 0.25
#define BUF_MB 512

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static const int PIN_COMPACT[8] = {0, 1, 2, 3, 4, 5, 6, 7};
static const int PIN_SPREAD[8]  = {0, 4, 1, 5, 2, 6, 3, 7};

static uint32_t read_range(const uint32_t *p, size_t n) {
    size_t vlmax = __riscv_vsetvlmax_e32m8();
    vuint32m8_t acc = __riscv_vmv_v_x_u32m8(0, vlmax);
    for (size_t i = 0; i < n;) {
        size_t vl = __riscv_vsetvl_e32m8(n - i);
        vuint32m8_t v = __riscv_vle32_v_u32m8(p + i, vl);
        acc = __riscv_vadd_vv_u32m8_tu(acc, acc, v, vl);
        i += vl;
    }
    vuint32m1_t z = __riscv_vmv_v_x_u32m1(0, 1);
    return __riscv_vmv_x_s_u32m1_u32(
        __riscv_vredsum_vs_u32m8_u32m1(acc, z, vlmax));
}

typedef struct {
    int id;
    const int *pin;
    const uint32_t *begin;
    size_t n;
    int reps, blocks;
    pthread_barrier_t *go, *done;
    uint32_t sink;
} warg;

static void *worker(void *p) {
    warg *w = p;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(w->pin[w->id], &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    for (int b = 0; b < w->blocks; b++) {
        pthread_barrier_wait(w->go);
        uint32_t s = 0;
        for (int r = 0; r < w->reps; r++)
            s += read_range(w->begin, w->n);
        w->sink += s;
        pthread_barrier_wait(w->done);
    }
    return NULL;
}

static int cmpd(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double run(const uint32_t *buf, size_t words, int threads,
                  const int *pin, int reps, int blocks, double *times) {
    pthread_barrier_t go, done;
    pthread_barrier_init(&go, NULL, threads + 1);
    pthread_barrier_init(&done, NULL, threads + 1);
    pthread_t tid[8];
    warg wa[8];
    for (int t = 0; t < threads; t++) {
        size_t b = words * t / threads, e = words * (t + 1) / threads;
        wa[t] = (warg){.id = t, .pin = pin, .begin = buf + b, .n = e - b,
                       .reps = reps, .blocks = blocks, .go = &go, .done = &done};
        pthread_create(&tid[t], NULL, worker, &wa[t]);
    }
    double tms[BLOCKS];
    for (int b = 0; b < blocks; b++) {
        double t0 = now_s();
        pthread_barrier_wait(&go);
        pthread_barrier_wait(&done);
        tms[b] = now_s() - t0;
    }
    for (int t = 0; t < threads; t++) pthread_join(tid[t], NULL);
    pthread_barrier_destroy(&go);
    pthread_barrier_destroy(&done);
    if (times) memcpy(times, tms, blocks * sizeof(double));
    qsort(tms, blocks, sizeof(double), cmpd);
    return tms[blocks / 2];
}

int main(void) {
    size_t bytes = (size_t)BUF_MB * 1024 * 1024;
    size_t words = bytes / 4;
    uint32_t *buf = aligned_alloc(64, bytes);
    if (!buf) { fprintf(stderr, "alloc failed\n"); return 1; }
    for (size_t i = 0; i < words; i++) buf[i] = (uint32_t)i * 2654435761u;

    printf("pin,threads,bytes,reps,median_s,min_s,max_s,gbps\n");
    const struct { const char *name; const int *map; } pins[] = {
        {"compact", PIN_COMPACT}, {"spread", PIN_SPREAD},
    };
    for (int p = 0; p < 2; p++) {
        for (int threads = 1; threads <= 8; threads *= 2) {
            double t1 = run(buf, words, threads, pins[p].map, 1, 1, NULL);
            int reps = (int)(TARGET_BLOCK_S / t1);
            if (reps < 1) reps = 1;
            if (reps > 1000) reps = 1000;
            double times[BLOCKS];
            double med = run(buf, words, threads, pins[p].map, reps, BLOCKS, times);
            qsort(times, BLOCKS, sizeof(double), cmpd);
            double per = med / reps;
            printf("%s,%d,%zu,%d,%.6e,%.6e,%.6e,%.3f\n",
                   pins[p].name, threads, bytes, reps, med,
                   times[0], times[BLOCKS - 1], bytes / per / 1e9);
            fflush(stdout);
        }
    }
    free(buf);
    return 0;
}
