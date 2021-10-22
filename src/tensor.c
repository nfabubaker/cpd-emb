#include <stdlib.h>
#include "tensor.h"

void init_tensor(struct tensor *t)
{
  t->inds = NULL;
  t->vals = NULL;
}


void free_tensor(struct tensor *t)
{
  idx_t i;

  if(t->inds != NULL)
    free(t->inds);

  if(t->vals != NULL)
    free(t->vals);
  free(t);
}
