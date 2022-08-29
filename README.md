# CPD-EMB: An MPI-based Distributed CP Decomposition with Latency hiding

## Synopsis
The aim of this software is to compute the CANICOM/PARAFAC (CP) decomposition of huge sprase tensors on large-scale HPC systems.
Fine-grain task parallelization of CP decomposition is known to be the most efficient in terms of computations. However, this type of parallelization incurs very high number of messages when the number of processors increase to thousands and tens of thousands. 
This software solves the high messages problem using a novel framework that embeds the point-to-point messages into a gloabal All-Reduce operation.  

For full context, please refer to the relevant IEEE TPDS publication: N. Abubaker, M. O. Karsavuran and C. Aykanat, "Scalable Unsupervised ML: Latency Hiding in Distributed Sparse Tensor Decomposition," in IEEE Transactions on Parallel and Distributed Systems, vol. 33, no. 11, pp. 3028-3040, 1 Nov. 2022, doi: 10.1109/TPDS.2021.3128827.  


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
-c How sparse communication takes place, can be one of the following:  
    0: Point-to-point communication (default), you can specify -a option with this type  
    2: Embedded communication (hypercube), use -d and -b option with this type  
-a Perform the sparse P2P communications with gloal collectives (all-to-all) (0:disable or 1: enabled). default: 0  
-f tensor storage option: 0: COO format, 1:CSR-like format 2: CSF   
format. default: 1  
-b Use hypercube imap file (topology-aware assignment) for embedded communication, a file name should be provided   


## Issues and Bug reporting:  
Please report any bugs or issues to Nabil Abubaker (abubaker.nf@gmail.com)  


## Citing:  

```
@ARTICLE{9618826,
  author={Abubaker, Nabil and Karsavuran, M. Ozan and Aykanat, Cevdet},
  journal={IEEE Transactions on Parallel and Distributed Systems}, 
  title={Scalable Unsupervised ML: Latency Hiding in Distributed Sparse Tensor Decomposition}, 
  year={2022},
  volume={33},
  number={11},
  pages={3028-3040},
  doi={10.1109/TPDS.2021.3128827}}
```  

## Acknowledgment:

The initial version of this code was written by Dr. Seher Acer (seheracer@gmail.com) in 2016-2017. Nabil Abubaker improved on the initial version and added the following:  
- Support for CSF storage and CSF-oriented MTTKRP  
- Implementing embedded communication from scratch.  
- A new CPD algorithm to accomodate the new embedded communication scheme.  


