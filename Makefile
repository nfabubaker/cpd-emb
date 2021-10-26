######################################################################
# @author      : Nabil Abubaker (nabil.abubaker@bilkent.edu.tr)
# @file        : Makefile
# @created     : Tuesday Aug 18, 2020 16:34:49 +03
######################################################################

MKLROOT=/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl
INTELMPIROOT=/opt/intel/compilers_and_libraries_2018.1.163/linux/mpi_2019/intel64

dir_guard=@mkdir -p $(@D)
IDIR =./include
LDIR=/home/nabil/research/libstfw/build
SDIR=src
ODIR=obj
BDIR=bin

CC=mpicc
CFLAGS=-O3 -m64 -I$(MKLROOT)/include 
#CFLAGS=-ggdb -DNA_DBG -m64 -I$(MKLROOT)/include

LIBS=-Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl

SOURCES = genst.c tensor.c mttkrp.c stat.c fibertensor.c emb.c comm.c cpd.c csf.c io.c util.c cp_als_mpi.c  
OBJ = $(patsubst %.c, $(ODIR)/%.o, $(SOURCES))
EXECS=$(BDIR)/parCPD

all: $(EXECS)

$(ODIR)/%.o: $(SDIR)/%.c
	$(dir_guard)
	$(CC) -c -o $@ $< $(CFLAGS)

$(BDIR)/parCPD: $(OBJ)
	$(dir_guard)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

.PHONY: clean, mrproper

clean:
	rm -f $(ODIR)/*.o

mrproper: clean
	rm -f $(BDIR)/parCPD

