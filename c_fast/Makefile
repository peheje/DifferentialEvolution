CC ?= gcc
CFLAGS ?= -O3 -march=native -ffast-math -fopenmp -DNDEBUG
LDFLAGS ?= -fopenmp

all: de_c settlement

de_c: main.c
	$(CC) $(CFLAGS) main.c -o de_c $(LDFLAGS)

settlement: settlement.c
	$(CC) $(CFLAGS) settlement.c -o settlement $(LDFLAGS) -lm

clean:
	rm -f de_c settlement

.PHONY: all clean
