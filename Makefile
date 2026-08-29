CC ?= gcc
MARCH ?= -march=rv64gcv
CFLAGS = -O3 $(MARCH) -Wall -Wextra -std=gnu11
LDFLAGS = -lpthread -lm

# pgvector's floating-point flags for the reassociated autovec baseline
PGV_FLAGS = -ftree-vectorize -fassociative-math -fno-signed-zeros \
            -fno-trapping-math -ffp-contract=fast

all: bench test_correctness bench_batch bw_probe

kernels_pgv.o: src/kernels_pgv.c
	$(CC) $(CFLAGS) $(PGV_FLAGS) -c src/kernels_pgv.c -o $@

bench: src/bench.c src/kernels.c src/kernels.h kernels_pgv.o
	$(CC) $(CFLAGS) src/bench.c src/kernels.c kernels_pgv.o -o $@ $(LDFLAGS)

test_correctness: src/test_correctness.c src/kernels.c src/kernels.h kernels_pgv.o
	$(CC) $(CFLAGS) src/test_correctness.c src/kernels.c kernels_pgv.o -o $@ $(LDFLAGS)

bench_batch: src/bench_batch.c src/kernels.c src/kernels.h kernels_pgv.o
	$(CC) $(CFLAGS) src/bench_batch.c src/kernels.c kernels_pgv.o -o $@ $(LDFLAGS)

bw_probe: src/bw_probe.c
	$(CC) $(CFLAGS) src/bw_probe.c -o $@ $(LDFLAGS)

kernels.s: src/kernels.c src/kernels.h
	$(CC) $(CFLAGS) -S src/kernels.c -o $@

clean:
	rm -f bench test_correctness bench_batch bw_probe kernels_pgv.o kernels.s

.PHONY: all clean
