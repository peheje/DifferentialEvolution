CC ?= gcc
CFLAGS ?= -O3 -march=native -ffast-math -fopenmp -DNDEBUG
LDFLAGS ?= -fopenmp

all: de_c de_bench de_xor

de_c: main.c de_solver.h
	$(CC) $(CFLAGS) main.c -o de_c $(LDFLAGS)

de_bench: de_bench.c de_solver.h
	$(CC) $(CFLAGS) de_bench.c -o de_bench $(LDFLAGS) -lm

de_xor: de_xor.c de_solver.h third_party/genann/genann.c third_party/genann/genann.h
	$(CC) $(CFLAGS) de_xor.c third_party/genann/genann.c -o de_xor $(LDFLAGS) -lm

clean:
	rm -f de_c de_bench de_experiments de_xor

.PHONY: all clean
