#include <stdlib.h>
#include <limits.h>
#include "fibertensor.h"

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

idx_t get_longest_fibers(idx_t *inds, real_t *vals, idx_t nmodes, idx_t longestmode, idx_t nnz, struct fibertensor *ft)
{
  idx_t i, *fibers, ptr;

  ft->lfibers = (idx_t *)malloc(nnz*sizeof(int));
  fibers = ft->lfibers;  
  ptr = longestmode;
  for(i = 0; i < nnz; i++)
    {	
      fibers[i] = inds[ptr];
      ptr += nmodes;
    }
  ft->lvals = (real_t *)malloc(nnz*sizeof(real_t));
  memcpy(ft->lvals, vals, nnz*sizeof(real_t));

}

idx_t get_secondlongest_fibers(idx_t *inds, real_t *vals, idx_t nmodes, idx_t secondlongestmode, idx_t nnz, struct fibertensor *ft)
{
  idx_t i, *fibers, ptr;

  ft->slfibers = (idx_t *)malloc(nnz*sizeof(int));
  fibers = ft->slfibers;

  ptr = secondlongestmode;
  for(i = 0; i < nnz; i++)
    {	
      fibers[i] = inds[ptr];
      ptr += nmodes;
    }

  ft->slvals = (real_t *)malloc(nnz*sizeof(real_t));
  memcpy(ft->slvals, vals, nnz*sizeof(real_t));
}


idx_t point_fibers(idx_t mype, idx_t *inds, idx_t *order, idx_t nmodes, idx_t nnz, idx_t mode, struct fibertensor *ft, idx_t longest)
{
  idx_t i, j, *shrdinds, last, nfib, **xfibers, *fibers, ptr, imode, cont, lastfibcnt, lastptd;
	

  ft->xfibers[mode] = (idx_t **)malloc((nmodes-1)*sizeof(idx_t *));
  shrdinds = (idx_t *)malloc((nmodes-1)*sizeof(int));

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
      xfibers[nmodes-imode-1] = (idx_t *)malloc((nfib+1)*2*sizeof(int));

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

}

idx_t get_sparsity_order(idx_t *gdims, idx_t *order, idx_t nmodes)
{
    idx_t i,j, k, dim, cnt, least, min, midx;
    idx_t *tmpArr = (idx_t *) malloc(sizeof(int) * nmodes);
    for(i=0; i< nmodes; i++){
        tmpArr[ i ] = gdims[i];
    }
    cnt = 0;
    least = -1; 
    while ( cnt < nmodes ){
        min = INT_MAX;
        for(i=0; i< nmodes; i++){
            if( tmpArr[i] < min && tmpArr[i] >=least && tmpArr[i]!= -2){
                min = gdims[i];
                midx = i;
            }
        }
        tmpArr[midx] = -2;
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
      min = INT_MAX;
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

idx_t get_fibertensor(struct genst *gs, struct tensor *t, struct fibertensor *ft)
{
  idx_t nmodes, i, c, j, nnz, *inds, min, *order, lmode;

  nmodes = gs->nmodes;
  nnz = t->nnz;

  gs->sporder = (idx_t *)malloc(nmodes*sizeof(int));
  get_sparsity_order(gs->gdims, gs->sporder, gs->nmodes);
  //get_sparsity_order(t, t->inds, t->sporder);
  ft->lmode = gs->sporder[nmodes-1];

  ft->order = (idx_t **)malloc(nmodes*sizeof(idx_t *));
  ft->xfibers = (idx_t ***)malloc(nmodes*sizeof(idx_t **));
  ft->topmostcnt = (idx_t *)malloc(nmodes*sizeof(int));
       
  for(i = 0; i < nmodes; i++)
    {
      ft->order[i] = (idx_t *)malloc(nmodes*sizeof(int));
      order = ft->order[i];
      c = 0;
      order[c++] = i;
      for(j = 0; j < nmodes; j++)
        if(gs->sporder[j] != i)
          order[c++] = gs->sporder[j];

      radixsort(t->inds, t->vals, nnz, nmodes, order, gs->ldims);
      checksort(t->inds, nnz, nmodes, order);
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
