MKLROOT="/opt/intel/compilers_and_libraries_2018.1.163/linux/mkl"
INTELMPIROOT="/opt/intel/compilers_and_libraries_2018.1.163/linux/mpi_2019/intel64"
CC="${INTELMPIROOT}"/bin/mpicc


link="-Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a ../libstfw/libstfw.a -Wl,--end-group -lpthread -lm -ldl"

#mpicc -O3 *.c -o parComp -m64 -I${MKLROOT}/include $link

mpicc -ggdb *.c -o parComp -m64 -I${MKLROOT}/include $link
