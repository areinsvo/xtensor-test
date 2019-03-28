# matrix multiplication with openmp intended as a Caliper test

CC=icpc
# CC=g++
INC=-I${CALIPER_DIR}/include
INC+=-I${CONDA_DIR}include
LIB=-L${CALIPER_DIR}/lib64 -lcaliper

OPT=-g
OPT+=-O3
OPT+=-march=skylake-avx512 
OPT+=-qopt-zmm-usage=high 
OPT+=-DXTENSOR_USE_XSIMD

C11=-std=c++11
C17=-std=c++17

all: xtensor-test

xtensor-test: xtensor-test.cc MatriplXT.h
	${CC} ${OPT} ${C11} -o xtest.exe ${INC} xtensor-test.cc ${LIB}

plain-test: plain-test.cc
	${CC} ${OPT} ${C17} -o plaintest.exe ${INC} plain-test.cc ${LIB}

clean:
	rm -f a.out *.exe *.o *.cali *.json

