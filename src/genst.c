#include <stdlib.h>
#include "genst.h"
void init_genst(struct genst *gs)
{
  gs->gdims = NULL;
  gs->ldims = NULL;
  gs->indmap = NULL;
  gs->gpart = NULL;
  gs->interpart = NULL;
  gs->intrapart = NULL;
  gs->meshdims = NULL;
  gs->meshinds = NULL;
  gs->chunksize = NULL;
  gs->comm = NULL;
  gs->layercomm = NULL;
  gs->layermype = NULL;
  gs->layersize = NULL;
  gs->mat = NULL;
  gs->uTu = NULL;

}

void get_chunk_info(struct genst *gs)
{
  idx_t i, j, nchunks, nmodes, meshsize, mult, *meshdims, *chunksize;

  nchunks = -1;
  nmodes = gs->nmodes;
  meshdims = gs->meshdims;
  meshsize = gs->meshsize;

  // compute number of chunks per processor
  for(i = 0; i < nmodes; i++)
    if(meshdims[i] != 1)
      {	if(nchunks == -1)
          nchunks = 1;
        else
          for(j = i; j < nmodes; j++)
            nchunks *= meshdims[j];
      }

  if(meshsize == 1)
    nchunks = 1;

  // compute chunk size in terms of parts
  gs->chunksize = (idx_t *)malloc(sizeof(int)*nmodes);
  chunksize = gs->chunksize;

  mult = 1;
  for(i = 0; i < nmodes; i++)
    {
      if(meshdims[i] == 1)
        chunksize[i] = meshsize;
      else
        {
          chunksize[i] = mult;
          mult *= meshdims[i];
        }
    }

}



/* void compute_mesh_dim(struct genst *gs)
 * {
 *   idx_t i, j, nfac, *fac, number, nmodes, target, furthest, *dist;
 * 
 *   number = gs->npes;
 *   //nfac= (int)log2((real_t)number)+1;
 *   fac = (idx_t *)malloc(sizeof(int)*nfac);
 * 
 *   nfac = 0;
 *   for(i = 2; i <= number; i++)
 *     while(number % i == 0 )
 *       {
 *         fac[nfac++] = i;
 *         number = number / i;
 *       }
 * 
 *   nmodes = gs->nmodes;
 *   target = 0;
 * 
 *   gs->meshdims = (idx_t *)malloc(sizeof(int)*nmodes);
 *   for(i = 0; i < nmodes; i++)
 *     {
 *       gs->meshdims[i] = 1;
 *       target += gs->gdims[i];
 *     }
 * 
 *   target = target/gs->npes;
 * 
 *   for(i = nfac-1; i >= 0; i--)
 *     {
 *       furthest = 0;
 *       dist = (idx_t *)malloc(nmodes*sizeof(int));
 *       setintzero(dist, nmodes);
 * 		
 *       for(j = 0; j < nmodes; j++)
 *         {
 *           dist[j] = gs->gdims[j]/gs->meshdims[j] - target;
 *           if(dist[j] > dist[furthest])
 *             furthest = j;
 *         }
 * 
 *       gs->meshdims[furthest] = gs->meshdims[furthest]*fac[i];
 *       free(dist);
 *     }
 *   gs->meshsize = gs->npes;
 * 
 *   free(fac);
 * 
 * }
 */

void init_matrices(struct genst *gs)
{
  idx_t i, j, k, l, nmodes, cprank, size, base;
  real_t *mat, v, *tmp, w;

  //	srand (time(NULL));

  nmodes = gs->nmodes;
  cprank = gs->cprank;
  tmp = (real_t *)malloc(nmodes*cprank*cprank*sizeof(real_t));
  gs->mat = (real_t **)malloc(nmodes*sizeof(real_t *));

  for(i = 0; i < nmodes; i++)
    {	
      size = gs->ldims[i]*cprank;
      //      posix_memalign(&(t->mat[i]), 64, size*sizeof(real_t));
      //t->mat[i] = (real_t *)malloc(size*sizeof(real_t));
      gs->mat[i] = (real_t *)malloc(size*sizeof(real_t));
      mat = gs->mat[i];
      for(j = 0; j < size; j++)
        {
          /*			mat[j] = 3.0 * (((real_t)rand()+1) / (real_t) RAND_MAX);
                        if(rand() % 2 == 0)
                        mat[j] *= -1;
          */
          mat[j] = 1.0;
				
        }
      base = cprank*cprank*i;
      size = gs->ldims[i];
      for(j = 0; j < cprank; j++)
        {
          v = 0;
          for(l = 0; l < size; l++)
            {	
              w = mat[l*cprank+j];
              v += w*w;
            }
          tmp[base+j*cprank+j] = v;
			
          for(k = j+1; k < cprank; k++)
            {
              v = 0;
              for(l = 0; l < size; l++)
                v += mat[l*cprank+j]*mat[l*cprank+k];

              tmp[base+k*cprank+j] = v;
              tmp[base+j*cprank+k] = v;
            }
        }
    }


  gs->uTu = (real_t *)malloc(nmodes*cprank*cprank*sizeof(real_t));

  /*idx_t pid;
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);

  printf("[%d] \t jsut before all-reduce comm in init mats\n", pid);

  MPI_Barrier(MPI_COMM_WORLD);*/

  MPI_Allreduce(tmp,gs->uTu, cprank*cprank*nmodes, MPI_REAL_T, MPI_SUM, MPI_COMM_WORLD);

  //printf("[%d] \t jsut after all-reduce comm in init mats\n", pid);

  free(tmp);
}

void free_genst(struct genst *gs)
{
  idx_t i;

  if(gs->gdims != NULL)
    free(gs->gdims);

  if(gs->ldims != NULL)
    free(gs->ldims);

  if(gs->indmap != NULL)
    {
      for(i = 0; i < gs->nmodes; i++)
        free(gs->indmap[i]);
    }

  if(gs->gpart != NULL)
    {	
      for(i = 0; i < gs->nmodes; i++)
        free(gs->gpart[i]);
      free(gs->gpart);
    }
  if(gs->interpart != NULL)
    {	
      for(i = 0; i < gs->nmodes; i++)
        free(gs->interpart[i]);
      free(gs->interpart);
    }
  if(gs->intrapart != NULL)
    {	
      for(i = 0; i < gs->nmodes; i++)
        free(gs->intrapart[i]);
      free(gs->intrapart);
    }

  if(gs->meshdims != NULL)
    free(gs->meshdims);

  if(gs->meshinds != NULL)
    free(gs->meshinds);

  if(gs->chunksize != NULL)
    free(gs->chunksize);

   if(gs->layercomm != NULL)
    free(gs->layercomm);
	
  if(gs->layermype != NULL)
    free(gs->layermype);

  if(gs->layersize != NULL)
    free(gs->layersize);

  if(gs->comm_type != EMB){
  if(gs->comm != NULL)
    free_comm(gs->comm, gs->nmodes);
  }

  if(gs->mat != NULL)
    {
      for(i = 0; i < gs->nmodes; i++)
        free(gs->mat[i]);
      free(gs->mat);
    }

  if(gs->uTu != NULL)
    free(gs->uTu);

  free(gs);
}
