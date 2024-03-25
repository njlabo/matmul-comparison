# choose compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
CC = gcc
LIBOMP_PREFIX = $(shell brew --prefix libomp)
CFLAGS = -Os -Wall -lm

ifeq ($(CC),gcc)
	CFLAGS += -lcblas
else
	CFLAGS += -I$(LIBOMP_PREFIX)/include -L$(LIBOMP_PREFIX)/lib -framework Accelerate -DACCELERATE_NEW_LAPACK
endif

compile:
	$(CC) run-linear.c -o run-linear $(CFLAGS)
	$(CC) -fopenmp run-linear.c -o run-linear-p $(CFLAGS)
	$(CC) run-conv.c -o run-conv $(CFLAGS)
	$(CC) -fopenmp run-conv.c -o run-conv-p $(CFLAGS)

.PHONY: compile