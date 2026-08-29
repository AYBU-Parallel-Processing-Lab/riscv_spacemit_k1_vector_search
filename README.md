# RVV vector search kernels (SpacemiT X60)

Distance-kernel benchmarks (squared L2, inner product, L1, cosine) for
RVV 1.0, comparing scalar, autovectorized, and hand-written intrinsics
kernels at LMUL 1/2/4/8, plus a centroid-batched transposed kernel, a
read-bandwidth probe, and a correctness test.

## Build (native, on the board)

```sh
make CC=gcc-15 MARCH="-march=rv64gcv_zvfh_zba_zbb_zbc_zbs_zicond_zicbom_zicboz"
```

Builds `bench`, `bench_batch`, `bw_probe`, `test_correctness`.
`src/kernels_pgv.c` is compiled with reassociation flags (see Makefile)
to form the relaxed autovec baseline; the same flags work for clang.

## Cross-compile (SpacemiT toolchain, static)

```sh
CC=riscv64-unknown-linux-gnu-gcc
$CC -O3 -mcpu=spacemit-x60 -static -std=gnu11 -ftree-vectorize \
    -fassociative-math -fno-signed-zeros -fno-trapping-math \
    -ffp-contract=fast -c src/kernels_pgv.c -o kernels_pgv.o
$CC -O3 -mcpu=spacemit-x60 -static -std=gnu11 \
    src/bench.c src/kernels.c kernels_pgv.o -o bench -lpthread -lm
```

## Run

```sh
./test_correctness           # all kernels vs float64 reference; expect PASS
./bench > results.csv        # full sweep: dims 128-8192, 1/2/4/8 threads,
                             # cache (256 KiB) + stream (256 MiB) regimes
./bench --op cos --kernel rvv_m4 --dims 768 --threads 1 --regime cache
./bench --pin spread         # threads across clusters (default: compact)
./bench --per-thread-kb 64   # fixed per-thread working set
./bench --fvecs file.fvecs   # replace synthetic data with an .fvecs set
./bench_batch > batch.csv    # row-major vs centroid-batched kernels
./bw_probe > bw.csv          # sustained sequential read bandwidth
```

Ops: `l2sq` (default), `dot`, `l1`, `cos`, `all`. Kernels: `scalar`,
`autovec`, `autovec_pgv`, `rvv_m1`..`rvv_m8`.

## Interpreting the CSV

One row per configuration. `median_s` is the median wall time of 7 blocks
of `reps` passes (`min_s`/`max_s` give the block spread); per-pass time is
`median_s / reps`. `vecs_per_s` is stored vectors compared per second,
`gflops` the arithmetic rate, `db_gbps` the bytes of stored vectors read
per second. Speedups: divide per-pass times of two rows with equal
`op`, `dim`, `regime`, and `threads`. `cpu0_khz`/`cpu4_khz` confirm the
clock held steady. `bw_probe` reports GB/s per thread count and pinning;
`bench_batch` reports pairs per second for row-major and batched kernels.
