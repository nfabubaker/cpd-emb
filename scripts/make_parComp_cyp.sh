MKLROOT="/opt/intel/compilers_and_libraries_2018.1.163/linux/mkl"
INTELMPIROOT="/opt/intel/compilers_and_libraries_2018.1.163/linux/mpi_2019/intel64"
#CC="${INTELMPIROOT}"/bin/mpicc
CC="/usr/bin/mpicc.mpich"


link="-Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a ../libstfw/build/libstfw.a -Wl,--end-group -lpthread -lm -ldl"

#mpicc -O3 *.c -o parComp -m64 -I${MKLROOT}/include $link
CFLAGS="-DNA_DBG -O3" 
rm -f ../src/parComp

${CC} ${CFLAGS} ../src/*.c -o ../src/parComp -m64 -I${MKLROOT}/include $link -L../libstfw/build -lstfw
