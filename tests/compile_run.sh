#!/usr/bin/env sh

######################################################################
# @author      : nabeelooo (nabeelooo@$HOSTNAME)
# @file        : compile_run
# @created     : Sunday Jun 20, 2021 12:56:06 +03
#
# @description : 
######################################################################
#np=$1
#if [ $np -eq 4 ]
#then
#    srcf="./emb_test_n3.c"
#else
#    srcf="./emb_test.c"
#fi
#
#
#echo "compiling ${srcf} on $np processors"
#mpicc -ggdb -DNA_DBG -DNA_DBG_L2 -DNA_DBG_L3 ${srcf} ../src/emb.c ../src/util.c -o embtest -lm
#echo "running ${srcf} on $np processors"
#mpiexec -np ${np} ./embtest 2>&1 | tee tmp.err

mpicc -ggdb -DNA_DBG -DNA_DBG_L2 ./emb_test_n3.c ../src/emb.c ../src/util.c -o embtest -lm
mpiexec -np 8 ./embtest 2>&1 | tee tmp.err

