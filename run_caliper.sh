#!/bin/bash

CMD=./xtest.exe
DIR=xtensor-caliper
mkdir $DIR

export CALI_CONFIG_FILE=~/soft/src/Caliper/examples/configs/papi_cycles.conf

export CALI_REPORT_FILENAME=$DIR/cali_tot_cyc_many.json
export CALI_PAPI_COUNTERS=PAPI_TOT_CYC
$CMD

export CALI_REPORT_FILENAME=$DIR/cali_L2_tca_many.json
export CALI_PAPI_COUNTERS=PAPI_L2_TCA
$CMD

export CALI_REPORT_FILENAME=$DIR/cali_L2_tcm_many.json
export CALI_PAPI_COUNTERS=PAPI_L2_TCM
$CMD

export CALI_REPORT_FILENAME=$DIR/cali_128_packed_ops_many.json
# note the difference from TAU counters - no PAPI_NATIVE_
export CALI_PAPI_COUNTERS=FP_ARITH:128B_PACKED_DOUBLE
$CMD

export CALI_REPORT_FILENAME=$DIR/cali_256_packed_ops_many.json
# note the difference from TAU counters - no PAPI_NATIVE_
export CALI_PAPI_COUNTERS=FP_ARITH:256B_PACKED_DOUBLE
$CMD

export CALI_REPORT_FILENAME=$DIR/cali_512_packed_ops_many.json
# note the difference from TAU counters - no PAPI_NATIVE_
export CALI_PAPI_COUNTERS=FP_ARITH:512B_PACKED_DOUBLE
$CMD

export CALI_REPORT_FILENAME=$DIR/cali_scalar_double_many.json
export CALI_PAPI_COUNTERS=FP_ARITH:SCALAR_DOUBLE
$CMD

export CALI_REPORT_FILENAME=$DIR/cali_scalar_single_many.json
export CALI_PAPI_COUNTERS=FP_ARITH:SCALAR_SINGLE
$CMD
