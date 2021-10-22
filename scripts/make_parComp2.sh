

module load GCC/8.3.0
module load ParaStationMPI/5.2.2-1
module load imkl/2019.3.199

mpic++ -O3 -std=c++11 -o cpalsozno3 cp-als-bin-ozan.c -lm -ldl -Wl,--start-group -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -Wl,--end-group
