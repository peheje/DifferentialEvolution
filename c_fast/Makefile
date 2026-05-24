CC ?= gcc
CFLAGS ?= -O3 -march=native -ffast-math -fopenmp -DNDEBUG
LDFLAGS ?= -fopenmp

all: de_c de_bench de_experiments

de_c: main.c
	$(CC) $(CFLAGS) main.c -o de_c $(LDFLAGS)

de_bench: de_bench.c
	$(CC) $(CFLAGS) de_bench.c -o de_bench $(LDFLAGS) -lm

de_experiments: de_experiments.c
	$(CC) $(CFLAGS) de_experiments.c -o de_experiments $(LDFLAGS) -lm

clean:
	rm -f de_c de_bench de_experiments

.PHONY: all clean
