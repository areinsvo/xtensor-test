# matrix multiplication with openmp intended as a Caliper test

CC=icpc
# CC=g++
INC=-I${CALIPER_DIR}/include
INC+=-I/home/users/gravelle/soft/anaconda3/include
LIB=-L${CALIPER_DIR}/lib64 -lcaliper

OPT=-g
OPT+=-O2

all: xtensor-test

xtensor-test: xtensor-test.cc MatriplXT.h
	${CC} -O2 -march=native -g -o xtest.exe ${INC} xtensor-test.cc ${LIB}

clean:
	rm -f a.out *.exe *.o *.cali *.json

