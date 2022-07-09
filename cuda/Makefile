CC=nvcc
CFLAGS=-O3
OPTIONS=-lineinfo
ARCH=-arch=sm_35
default: mmm

mmm: mmm.cu
	$(CC) $(CFLAGS) $(ARCH) -o mmm mmm.cu
