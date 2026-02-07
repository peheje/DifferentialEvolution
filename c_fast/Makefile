CC ?= gcc
CFLAGS ?= -O3 -march=native -ffast-math -fopenmp -DNDEBUG
LDFLAGS ?= -fopenmp

all: de_c

de_c: main.c
	$(CC) $(CFLAGS) main.c -o de_c $(LDFLAGS)

clean:
	rm -f de_c

.PHONY: all clean
