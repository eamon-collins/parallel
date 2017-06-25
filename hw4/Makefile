CC=gcc
CFLAGS=-O3
OPTIONS=-lnuma -lpthread

default: pthread

pthread: pthread.c
	$(CC) $(CFLAGS) -o pthread pthread.c $(OPTIONS)
