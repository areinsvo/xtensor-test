# matrix multiplication with openmp intended as a Caliper test

CC=icpc
# CC=g++
INC=-I${CALIPER_DIR}/include
INC+=-I${CONDA_DIR}include
LIB=-L${CALIPER_DIR}/lib64 -lcaliper

OPT=-g
OPT+=-O3
OPT+=-march=skylake-avx512 
OPT+=-DXTENSOR_USE_XSIMD

all: xtensor-test

xtensor-test: xtensor-test.cc MatriplXT.h
	${CC} -O2 -march=native -g -o xtest.exe ${INC} xtensor-test.cc ${LIB}

clean:
	rm -f a.out *.exe *.o *.cali *.json

