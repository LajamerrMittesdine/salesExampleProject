CC ?= gcc
CFLAGS ?= -std=c99 -O2 -Wall -Wextra -Iinclude -lm

.PHONY: test clean

test_goldens: src/ops.c src/test_goldens.c include/kimi_k3_ref.h ../fixtures/goldens_embedded.h
	$(CC) -std=c99 -O2 -Wall -Wextra -Iinclude -o test_goldens src/ops.c src/test_goldens.c -lm

test: test_goldens
	./test_goldens

clean:
	rm -f test_goldens
