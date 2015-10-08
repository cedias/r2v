CC = gcc
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS = -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result

all: d2v

d2v : w2p.c
	$(CC) w2p.c -o d2v $(CFLAGS)

debug : clean deb

deb :  w2p.c
	$(CC) w2p.c -o d2v $(CFLAGS) -g

clean:
	rm -rf d2v

