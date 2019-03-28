# matrix multiplication with openmp intended as a Caliper test

CC=icpc
# CC=g++

USE_CALI:=1
USE_OPTS:=1

INC=-I${CONDA_DIR}include

ifdef USE_OPTS
	OPT=-g
	OPT+=-O3
	OPT+=-march=skylake-avx512 
	OPT+=-qopt-zmm-usage=high 
	OPT+=-DXTENSOR_USE_XSIMD
else
	OPT=-g
	OPT+=-no-vec
endif

ifdef USE_CALI
	LIB=-L${CALIPER_DIR}/lib64 -lcaliper
	INC+=-I${CALIPER_DIR}/include
	OPT+=-DUSE_CALI
endif

C11=-std=c++11
C17=-std=c++17

all: xtensor-test

xtensor-test: xtensor-test.cc MatriplXT.h
	${CC} ${OPT}  -o xtest.exe ${INC} xtensor-test.cc ${LIB}

plain-test: plain-test.cc
	${CC} ${OPT} ${C17} -o plaintest.exe ${INC} plain-test.cc ${LIB}

clean:
	rm -f a.out *.exe *.o *.cali *.json

