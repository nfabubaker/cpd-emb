#ifndef GENST_H_
#define GENST_H_
#include "basic.h"
#include <mpi.h>
#include <math.h>
struct genst;

#include "comm.h"

/*
 *@breif A general struct that contains all necessery data for most of the cp-als operations, inlcuding mttkrp
 *
 * */

typedef struct genst
{
  int mype;
  int npes;
	
  idx_t gnnz;      // number of nonzeros in the overall tensor
  int nmodes;    // number of modes of the tensor 
  idx_t *gdims;    // array of size nmodes, gdims[i] denotes the size in mode i+1 
  idx_t *ldims;    // local dimensions
  idx_t **indmap;  // relabeling of indices local to the processor

  real_t *cpbuff;
  real_t *cpsqbuff;


  idx_t **gpart;   // array of pointers, gpart[i] points to the partition array of rows in mode i+1 
  // for cyclic dist: gpart[i][j] denotes the owner processor of row j of mode i, for checkerboard dist: gpart[i][j] denoted the mesh layer of row j 
  idx_t **interpart;   //ckbd distribution, mesh layer id
  idx_t **intrapart;   //ckbd distribution, owner of the row inside mesh layer


  idx_t meshsize;  // overall number of processors
  idx_t *meshdims; // array of size nmodes, meshdims[i] denotes the number of processors in mode i in virtual processor mesh 
  idx_t *meshinds; // the indices of the processor at each mode

  idx_t *chunksize; // array of size nmodes, denoting the size of the chunk along each mode IN TERMS OF PARTS

  idx_t *sporder;    //sparsity order
  idx_t nnz;         //total number of nonzeros of my subtensor

  //cartesian nd processors 
  MPI_Comm *layercomm;
  int *layermype;
  int *layersize;
	
  struct comm *comm;


  // factor matrices
  int cprank;
  real_t **mat;
  real_t *matm; /* this to be used as A^, B^ .. etc */

  // small dense U^T.U matrices for each factor matrix U
  real_t *uTu;

  // fiber storage?
  int fiber;

  // all-to-all communication?
  int alltoall;

  // checkerboard
  int comm_type;

  char hc_imap_FN[1024];
  int use_hc_imap;
} genst;

void init_genst(struct genst *gs);

void get_chunk_info(struct genst *gs);
void compute_mesh_dim(struct genst *gs);
void init_matrices(struct genst *gs);
void free_genst(struct genst *gs);

#endif
