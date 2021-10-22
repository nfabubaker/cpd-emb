#ifndef TP_TENSOR_H
#define TP_TENSOR_H
#include "basic.h"
typedef struct tensor
{
  idx_t gnnz;      // number of nonzeros in the overall tensor
  idx_t nmodes;    // number of modes of the tensor 
  idx_t *inds;
  real_t *vals;
  idx_t nnz;         //total number of nonzeros of my subtensor	 
  // factor matrices
} tensor;

void init_tensor(tensor *t);
void free_tensor(tensor *t);

#endif

