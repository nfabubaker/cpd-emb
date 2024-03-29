#include <stdlib.h>
#include <limits.h>
#include "fibertensor.h"
#include <string.h>

void init_fibertensor(struct fibertensor *ft)
{
  ft->fibermode = NULL;
  ft->order = NULL;
  ft->topmostcnt = NULL;
  ft->xfibers = NULL;
  ft->lfibers = NULL;
  ft->slfibers = NULL;
  ft->lvals = NULL;
  ft->slvals = NULL;
}

idx_t get_longest_fibers(const idx_t *inds, const real_t *vals, const idx_t nmodes, const idx_t longestmode, const idx_t nnz, struct fibertensor *ft)
{
  idx_t i, *fibers, ptr;

  ft->lfibers = (idx_t *)malloc(nnz*sizeof(*ft->lfibers));
  fibers = ft->lfibers;  
  ptr = longestmode;
  for(i = 0; i < nnz; i++)
    {	
      fibers[i] = inds[ptr];
      ptr += nmodes;
    }
  ft->lvals = (real_t *)malloc(nnz*sizeof(*ft->lvals));
  memcpy(ft->lvals, vals, nnz*sizeof(real_t));
  return 0;
}

idx_t get_secondlongest_fibers(const idx_t *inds, const real_t *vals, const idx_t nmodes, const idx_t secondlongestmode, const idx_t nnz, struct fibertensor *ft)
{
  idx_t i, *fibers, ptr;

  ft->slfibers = (idx_t *)malloc(nnz*sizeof(*ft->slfibers));
  fibers = ft->slfibers;

  ptr = secondlongestmode;
  for(i = 0; i < nnz; i++)
    {	
      fibers[i] = inds[ptr];
      ptr += nmodes;
    }

  ft->slvals = (real_t *)malloc(nnz*sizeof(*ft->slvals));
  memcpy(ft->slvals, vals, nnz*sizeof(*vals));
  return 0;
}


idx_t point_fibers(const idx_t mype, const idx_t *inds, const idx_t *order, const idx_t nmodes, const idx_t nnz, const idx_t mode, struct fibertensor *ft, const idx_t longest)
{
  idx_t i, j, *shrdinds, last, nfib, **xfibers, *fibers, ptr, cont, lastfibcnt, lastptd;
  idx_t imode;
	

  ft->xfibers[mode] = (idx_t **)malloc((nmodes-1)*sizeof(*ft->xfibers[mode]));
  shrdinds = (idx_t *)malloc((nmodes-1)*sizeof(*shrdinds));

  if(longest)
    fibers = ft->lfibers;
  else
    fibers = ft->slfibers;

  xfibers = ft->xfibers[mode];

  lastfibcnt = nnz;
  for(imode = 1; imode < nmodes; imode++)
    {
		
      //count the number of fibers
      last = 0;
      nfib = 0;
      while(last < lastfibcnt)
        {
          if(imode == 1)
            lastptd = last;
          else if(imode == 2)
            lastptd = xfibers[nmodes-imode][last*2];
          else
            {
              lastptd = xfibers[nmodes-imode][last*2];
              lastptd = xfibers[nmodes-imode+1][lastptd*2];
            }

          for(j = 0; j < nmodes-imode; j++)
            shrdinds[j] = inds[lastptd*nmodes + order[j]];
			
          cont = 1;
          i = last+1;
          while( cont && i< lastfibcnt) 
            {
              if(imode == 1)
                lastptd = i;
              else if(imode == 2)
                lastptd = xfibers[nmodes-imode][i*2];
              else
                {
                  lastptd = xfibers[nmodes-imode][i*2];
                  lastptd = xfibers[nmodes-imode+1][lastptd*2];
                }

              for(j = 0; j < nmodes-imode; j++)
                if(shrdinds[j] != inds[lastptd*nmodes+order[j]])
                  cont = 0;
              if(cont)
                  i++;
            }
			
          nfib++;
          last = i;	
        } 
      //allocate space for the last level fiber pointer
      xfibers[nmodes-imode-1] = (idx_t *)malloc((nfib+1)*2*sizeof(*xfibers[nmodes-imode-1]));

      //find pointers to fiber starting positions
      //xfibers[nmodes-2][fid*2]: starting position of the fiber in fibers
      //xfibers[nmodes-2][fid*2+1]: the index of the fiber in the previous dimension
      last = 0;
      nfib = 0;
      xfibers[nmodes-imode-1][nfib*2] = last;
      while(last < lastfibcnt)
        {

          if(imode == 1)
            lastptd = last;
          else if(imode == 2)
            lastptd = xfibers[nmodes-imode][last*2];
          else
            {
              lastptd = xfibers[nmodes-imode][last*2];
              lastptd = xfibers[nmodes-imode+1][lastptd*2];
            }

          for(j = 0; j < nmodes-imode; j++)
            shrdinds[j] = inds[lastptd*nmodes + order[j]];
			
          xfibers[nmodes-imode-1][nfib*2+1] = shrdinds[nmodes-imode-1];
			
          cont = 1;
          i = last+1;
          while( cont && i < lastfibcnt)
            {

              if(imode == 1)
                lastptd = i;
              else if(imode == 2)
                lastptd = xfibers[nmodes-imode][i*2];
              else
                {
                  lastptd = xfibers[nmodes-imode][i*2];
                  lastptd = xfibers[nmodes-imode+1][lastptd*2];
                }

              for(j = 0; j < nmodes-imode; j++)
                if(shrdinds[j] != inds[lastptd*nmodes+order[j]])
                  cont = 0;
              if(cont)
                  i++;
            }
			
          nfib++;
          last = i;
          xfibers[nmodes-imode-1][nfib*2] = last;
				
        }
      lastfibcnt = nfib;
    }
  ft->topmostcnt[mode] = nfib;

  free(shrdinds);
return 0;
}

void get_sparsity_order(const idx_t *gdims, idx_t *order, const idx_t nmodes)
{
    idx_t i,j, k, cnt, midx;
    idx_t dim, min, least;
    idx_t *tmpArr = (idx_t *) malloc(sizeof(*tmpArr) * nmodes);
    for(i=0; i< nmodes; i++){
        tmpArr[ i ] = gdims[i];
    }
    cnt = 0;
    least = 0; 
    while ( cnt < nmodes ){
        min = IDX_T_MAX;
        for(i=0; i< nmodes; i++){
            if( tmpArr[i] < min && tmpArr[i] >=least && tmpArr[i]!= 0){
                min = gdims[i];
                midx = i;
            }
        }
        tmpArr[midx] = 0;
        least = min;
        order[cnt] = midx;
        cnt++; 
    }
    free(tmpArr);
}


/*idx_t get_sparsity_order_old(struct tensor *t, idx_t *inds, idx_t *order)
{
  idx_t i, j, nmodes, nnz, *mark, *minecnt, **mine, ptr, ldim, *minei, cnt, min;

  nmodes = t->nmodes;
  nnz = t->nnz;

  mark = (idx_t *)malloc(nmodes*sizeof(int));
  setintzero(mark, nmodes);
  minecnt = (idx_t *)malloc(nmodes*sizeof(int));
  setintzero(minecnt, nmodes);
  mine = (idx_t **)malloc(nmodes*sizeof(idx_t *));

  for(i = 0; i < nmodes; i++)
    {
      mine[i] = (idx_t *)malloc(t->ldims[i]*sizeof(int));
      setintzero(mine[i], t->ldims[i]);
    }

  ptr = 0;
  for(i = 0; i < nnz; i++)
    {
      for(j = 0; j < nmodes; j++)
        mine[j][inds[ptr++]] = 1;
    }

  for(i = 0; i < nmodes; i++)
    {
      cnt = 0;
      ldim = t->ldims[i];
      minei = mine[i];
      for(j = 0; j < ldim; j++)
        cnt += minei[j];

      minecnt[i] = cnt;
		
    }

  for(i = 0; i < nmodes; i++)
    {
      min = IDX_T_MAX;
      for(j = 0; j < nmodes; j++)
        {
          if(mark[j] == 0 && minecnt[j] < min)
            {
              min = minecnt[j];
              order[i] = j;
            }
        }
      mark[order[i]] = 1;

    }

  for(i = 0; i < nmodes; i++)
    free(mine[i]);
  free(mine);
  free(minecnt);
  free(mark);

}*/

idx_t get_fibertensor(genst *gs, const tensor *t, struct fibertensor *ft)
{
  idx_t nmodes,  *order, lmode;
  idx_t i, c, j, nnz, *inds, min;
  nmodes = gs->nmodes;
  nnz = t->nnz;

#ifdef NA_DBG
        na_log(dbgfp, "hello from get_fibertensor\n");
#endif
  gs->sporder = malloc(nmodes*sizeof(*gs->sporder));
  get_sparsity_order(gs->gdims, gs->sporder, gs->nmodes);
  //get_sparsity_order(t, t->inds, t->sporder);
  ft->lmode = gs->sporder[nmodes-1];

  ft->order = malloc(nmodes*sizeof(*ft->order));
  ft->xfibers = malloc(nmodes*sizeof(*ft->xfibers));
  ft->topmostcnt = malloc(nmodes*sizeof(*ft->topmostcnt));
       
  for(i = 0; i < nmodes; i++)
    {
      ft->order[i] = malloc(nmodes*sizeof(*ft->order[i]));
      order = ft->order[i];
      c = 0;
      order[c++] = i;
      for(j = 0; j < nmodes; j++)
        if(gs->sporder[j] != i)
          order[c++] = gs->sporder[j];

#ifdef NA_DBG
      MPI_Barrier(MPI_COMM_WORLD);
        na_log(dbgfp, "\tmode %d before sort\n", i);
#endif
/*           {
 *               volatile idx_t tt = 0;
 *               printf("PID %d on %d ready for attach\n", gs->mype,  getpid());
 *               fflush(stdout);
 *               while (0 == tt)
 *                   sleep(5);
 *           }
 */
      radixsort(t->inds, t->vals, nnz, nmodes, order, gs->ldims);
#ifdef NA_DBG
      MPI_Barrier(MPI_COMM_WORLD);
        na_log(dbgfp, "\tmode %d after sort\n", i);
#endif
      checksort(t->inds, nnz, nmodes, order);
#ifdef NA_DBG
      MPI_Barrier(MPI_COMM_WORLD);
        na_log(dbgfp, "\tmode %d after checksort\n", i);
#endif
      if(i != ft->lmode)
        {
          if(ft->lfibers == NULL)
            get_longest_fibers(t->inds, t->vals, nmodes, ft->lmode, nnz, ft);       
          point_fibers(gs->mype, t->inds, order, nmodes, nnz, i, ft, 1);
        }
      else
        {
          if(ft->slfibers == NULL)
            get_secondlongest_fibers(t->inds, t->vals, nmodes, gs->sporder[nmodes-2], nnz, ft);
          point_fibers(gs->mype, t->inds, order, nmodes, nnz, i, ft, 0);
        }
#ifdef NA_DBG
      MPI_Barrier(MPI_COMM_WORLD);
        na_log(dbgfp, "\tmode %d after point_fibers\n", i);
#endif
    }

}

void free_fibertensor(struct fibertensor *ft, idx_t nmodes)
{
  idx_t i, j;
  if(ft->fibermode != NULL)
    free(ft->fibermode);

  if(ft->order != NULL)
    {
      for(i = 0; i < nmodes; i++)
        free(ft->order[i]);
      free(ft->order);
    }

  if(ft->topmostcnt != NULL)
    free(ft->topmostcnt);

  if(ft->xfibers != NULL)
    {
      for(i = 0; i < nmodes; i++)
        {
          for(j = 0; j < nmodes-1; j++)
            free(ft->xfibers[i][j]);
          free(ft->xfibers[i]);
        }
      free(ft->xfibers);
    }

  if(ft->lfibers != NULL)
    free(ft->lfibers);

  if(ft->slfibers != NULL)
    free(ft->slfibers);

  if(ft->lvals != NULL)
    free(ft->lvals);

  if(ft->slvals != NULL)
    free(ft->slvals);
}
