CC ?= gcc
CFLAGS ?= -O3 -march=native -ffast-math -fopenmp -DNDEBUG
LDFLAGS ?= -fopenmp

all: de_c de_bench

de_c: main.c
	$(CC) $(CFLAGS) main.c -o de_c $(LDFLAGS)

de_bench: de_bench.c
	$(CC) $(CFLAGS) de_bench.c -o de_bench $(LDFLAGS) -lm

clean:
	rm -f de_c de_bench

.PHONY: all clean
