# CPD-EMB: An MPI-based Distributed CP Decomposition with Latency hiding

## Synopsis
The aim of this software is to compute the CANICOM/PARAFAC (CP) decomposition of huge sprase tensors on large-scale HPC systems.
Fine-grain task parallelization of CP decomposition is known to be the most efficient in terms of computations. However, this type of parallelization incurs very high number of messages when the number of processors increase to thousands and tens of thousands. 
This software solves the high messages problem using a novel framework that embeds the point-to-point messages into a gloabal All-Reduce operation.


## Usage
This code requires MKL (can be found in Intel's OneAPI).
To compile the code, run:  
```
make
```
A sample usage can be as follows:
```
mpirun -np 1024 cpd-emb [options] inputTensorFile
```
-m row to processor assignment options: 0 = random 1 = random-respect-comm
-p partitionfile: in the partition file, indices numbered in the mode order
-r rank: (int) rank of CP decomposition. default: 16
-i number of CP-ALS iterations (max). default: 10
-c communication type, can be one of the following:
    0: Point-to-poidx_t communication (default), you can specify -a option with this type
    2: Embedded communication (hypercube), use -d and -b option with this type
-a Perform P2P communications with gloal collectives (0:disable or 1: enabled). default: 0
-f tensor storage option: 0: COO format, 1:CSR-like format 2: CSF 
format. default: 1
-b use hypercube imap file for embedded communication, a file name should be provided
    
