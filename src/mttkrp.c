#include "mttkrp.h"
#include <stdlib.h>
#include <string.h>
void mttkrp_nnz(struct genst *gs, struct tensor *t, idx_t mode, real_t *matm)
{
  idx_t c, i, j, k, size, cprank, nmodes, iwrite, iread, *inds, nnz;
  real_t *vals, v, *acc, *mat, **mats;

  nmodes = gs->nmodes;
  cprank = gs->cprank;
  mats = gs->mat;

  //acc = (real_t *)malloc(sizeof(real_t)*cprank);
  acc = gs->cpbuff;

	
  nnz = t->nnz;
  inds = t->inds;
  vals = t->vals;
		
  for(i = 0; i < nnz; i++)
    {
      v = vals[i];
      for(k = 0; k < cprank; k++)
        acc[k] = v;

      for(j = 0; j < nmodes; j++)
        {
          if(j != mode)
            {
              iread = inds[i*nmodes+j]*cprank;
              mat = mats[j];
				
              for(k = 0; k < cprank; k++)
                acc[k] *= mat[iread++];
            }	
        }

      iwrite = inds[i*nmodes+mode]*cprank;
      for(k = 0; k < cprank; k++)
        matm[iwrite++] += acc[k];
    }
	
  //free(acc);
}

void mttkrp_fiber_3(struct genst *gs, struct fibertensor *ft, idx_t mode, real_t *matm)
{
  idx_t j, nmodes, cprank, cnt, *xfibers0, *xfibers1, *fibers, i0, st0, en0, base0, i1, st1, en1, base1, i2, base2;
  real_t *mat1, *mat2, *vals, *acc;

  nmodes = gs->nmodes;
  cprank = gs->cprank;

  xfibers0 = ft->xfibers[mode][0];
  xfibers1 = ft->xfibers[mode][1];

  cnt = ft->topmostcnt[mode];
	
  mat1 = gs->mat[ft->order[mode][1]];
  mat2 = gs->mat[ft->order[mode][2]];

  if(mode == ft->lmode)
    {
      fibers = ft->slfibers;
      vals = ft->slvals;
    }
  else
    {
      fibers = ft->lfibers;
      vals = ft->lvals;
    }

  //acc = (real_t *)malloc(sizeof(real_t)*cprank);
  acc = gs->cpbuff;


  for(i0 = 0; i0 < cnt; i0++)
    {
      base0 = i0*2;
      st0 = xfibers0[base0];
      en0 = xfibers0[base0+2];
      base0 = xfibers0[base0+1]*cprank;

      for(i1 = st0; i1 < en0; i1++)
        {
          base1 = i1*2;
          st1 = xfibers1[base1];
          en1 = xfibers1[base1+2];
          base1 = xfibers1[base1+1]*cprank;

          memset(acc, 0, sizeof(real_t)*cprank);

          for(i2 = st1; i2 < en1; i2++)
            {
              base2 = fibers[i2]*cprank; 
              for(j = 0; j < cprank; j++)
                acc[j] += mat2[base2+j]*vals[i2];
            }
			
          for(j = 0; j < cprank; j++)
            matm[base0+j] += acc[j]*mat1[base1+j];
        }

    }

  //free(acc);
}

void mttkrp_fiber_4(struct genst *gs, struct fibertensor *ft, idx_t mode, real_t *matm)
{
  idx_t j, nmodes, cprank, cnt, *xfibers0, *xfibers1, *xfibers2, *fibers, i0, st0, en0, i1, st1, en1, i2, st2, en2, en3, i3, base0, base1, base2, base3;
  real_t *mat1, *mat2, *mat3, *vals, *acc, *acc2;

  nmodes = gs->nmodes;
  cprank = gs->cprank;

  xfibers0 = ft->xfibers[mode][0];
  xfibers1 = ft->xfibers[mode][1];
  xfibers2 = ft->xfibers[mode][2];

  cnt = ft->topmostcnt[mode];
	
  mat1 = gs->mat[ft->order[mode][1]];
  mat2 = gs->mat[ft->order[mode][2]];
  mat3 = gs->mat[ft->order[mode][3]];

  if(mode == ft->lmode)
    {
      fibers = ft->slfibers;
      vals = ft->slvals;
    }
  else
    {
      fibers = ft->lfibers;
      vals = ft->lvals;
    }

  //acc = (real_t *)malloc(sizeof(real_t)*cprank);
  //acc2 = (real_t *)malloc(sizeof(real_t)*cprank); 
  
  acc = gs->cpbuff;
  acc2 = gs->cpsqbuff;
	

  for(i0 = 0; i0 < cnt; i0++)
    {
      base0 = i0*2;
      st0 = xfibers0[base0];
      en0 = xfibers0[base0+2];
      base0 = xfibers0[base0+1]*cprank;

      for(i1 = st0; i1 < en0; i1++)
        {
          base1 = i1*2;
          st1 = xfibers1[base1];
          en1 = xfibers1[base1+2];
          base1 = xfibers1[base1+1]*cprank;

          memset(acc, 0, sizeof(real_t)*cprank);
          for(i2 = st1; i2 < en1; i2++)
            {
              base2 = i2*2;
              st2 = xfibers2[base2];
              en2 = xfibers2[base2+2];
              base2 = xfibers2[base2+1]*cprank;

              memset(acc2, 0, sizeof(real_t)*cprank);
              for(i3 = st2; i3 < en2; i3++)
                {
                  base3 = fibers[i3]*cprank;
                  for(j = 0; j < cprank; j++)
                    acc2[j] += mat3[base3+j]*vals[i3];
                }
				
              for(j = 0; j < cprank; j++)
                acc[j] += acc2[j]*mat2[base2+j];
				

            }
          for(j = 0; j < cprank; j++)
            matm[base0+j] += acc[j]*mat1[base1+j];
			
        }

    }

  //free(acc);
  //free(acc2);
}


real_t mttkrp_nnz_stats(struct genst *gs, struct tensor *t, idx_t mode, real_t *matm, idx_t niters)
{
  idx_t it, c, i, j, k, size, cprank, nmodes, iwrite, iread, *inds, nnz;
  real_t *vals, v, *acc, *mat, **mats, time = 0;

  nmodes = gs->nmodes;
  cprank = gs->cprank;
  mats = gs->mat;

  //acc = (real_t *)malloc(sizeof(real_t)*cprank);
  acc = gs->cpbuff;
  nnz = t->nnz;
  inds = t->inds;
  vals = t->vals;


  MPI_Barrier(MPI_COMM_WORLD);
	
  for(it = 0; it < niters; it++)
    {
	
      for(i = 0; i < nnz; i++)
        {
          v = vals[i];
          for(k = 0; k < cprank; k++)
            acc[k] = v;

          for(j = 0; j < nmodes; j++)
            {
              if(j != mode)
                {
                  iread = inds[i*nmodes+j]*cprank;
                  mat = mats[j];
				
                  for(k = 0; k < cprank; k++)
                    acc[k] *= mat[iread++];
                }	
            }

          iwrite = inds[i*nmodes+mode]*cprank;
          for(k = 0; k < cprank; k++)
            matm[iwrite++] += acc[k];
        }
    }	
  MPI_Barrier(MPI_COMM_WORLD);

  //free(acc);

  return time;
}

idx_t  mttkrp_fiber_3_stats(struct genst *gs, struct fibertensor *ft, idx_t mode, real_t *matm)
{
  idx_t it, j, nmodes, cprank, cnt, *xfibers0, *xfibers1, *fibers, i0, st0, en0, base0, i1, st1, en1, base1, i2, base2;
  real_t *mat1, *mat2, *vals, *acc, time;

  nmodes = gs->nmodes;
  cprank = gs->cprank;

  xfibers0 = ft->xfibers[mode][0];
  xfibers1 = ft->xfibers[mode][1];

  cnt = ft->topmostcnt[mode];
	
  mat1 = gs->mat[ft->order[mode][1]];
  mat2 = gs->mat[ft->order[mode][2]];

  if(mode == ft->lmode)
    {
      fibers = ft->slfibers;
      vals = ft->slvals;
    }
  else
    {
      fibers = ft->lfibers;
      vals = ft->lvals;
    }

  //acc = (real_t *)malloc(sizeof(real_t)*cprank);
  acc = gs->cpbuff;
  
  idx_t fc = 0;
  for(i0 = 0; i0 < cnt; i0++)
    {
      base0 = i0*2;
      st0 = xfibers0[base0];
      en0 = xfibers0[base0+2];
      base0 = xfibers0[base0+1]*cprank;

      for(i1 = st0; i1 < en0; i1++)
        {
          base1 = i1*2;
          st1 = xfibers1[base1];
          en1 = xfibers1[base1+2];
          base1 = xfibers1[base1+1]*cprank;

          memset(acc, 0, sizeof(real_t)*cprank);
          fc++;
          for(i2 = st1; i2 < en1; i2++)
            {
              base2 = fibers[i2]*cprank; 
              for(j = 0; j < cprank; j++)
                acc[j] += mat2[base2+j]*vals[i2];
            }
        
          for(j = 0; j < cprank; j++)
            matm[base0+j] += acc[j]*mat1[base1+j];
        }

    }
  //fc /= niters;
  //fc += t->nnz;

  //cnt += fc;
  
  //idx_t avg, max, avgs, maxs;
  //MPI_Reduce(&fc, &avg, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  //MPI_Reduce(&fc, &max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

  //MPI_Reduce(&cnt, &avgs, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  //MPI_Reduce(&cnt, &maxs, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

//  cnt_st[mode*4+0] = max;
//  cnt_st[mode*4+1] = avg/t->npes;
//  cnt_st[mode*4+2] = maxs;
//  cnt_st[mode*4+3] = avgs/t->npes;

  //free(acc);

  return fc;
}

idx_t mttkrp_fiber_4_stats(struct genst *gs, struct fibertensor *ft, idx_t mode, real_t *matm)
{
  idx_t it, j, nmodes, cprank, cnt, *xfibers0, *xfibers1, *xfibers2, *fibers, i0, st0, en0, i1, st1, en1, i2, st2, en2, en3, i3, base0, base1, base2, base3;
  real_t *mat1, *mat2, *mat3, *vals, *acc, *acc2, time;

  nmodes = gs->nmodes;
  cprank = gs->cprank;

  xfibers0 = ft->xfibers[mode][0];
  xfibers1 = ft->xfibers[mode][1];
  xfibers2 = ft->xfibers[mode][2];

  cnt = ft->topmostcnt[mode];
	
  mat1 = gs->mat[ft->order[mode][1]];
  mat2 = gs->mat[ft->order[mode][2]];
  mat3 = gs->mat[ft->order[mode][3]];

  if(mode == ft->lmode)
    {
      fibers = ft->slfibers;
      vals = ft->slvals;
    }
  else
    {
      fibers = ft->lfibers;
      vals = ft->lvals;
    }

  //acc = (real_t *)malloc(sizeof(real_t)*cprank);
  //acc2 = (real_t *)malloc(sizeof(real_t)*cprank); 
  acc = gs->cpbuff;
  acc2 = gs->cpsqbuff;
 
  idx_t fc = 0, sc = 0;
  for(i0 = 0; i0 < cnt; i0++)
    {
      base0 = i0*2;
      st0 = xfibers0[base0];
      en0 = xfibers0[base0+2];
      base0 = xfibers0[base0+1]*cprank;

      for(i1 = st0; i1 < en0; i1++)
        {
          base1 = i1*2;
          st1 = xfibers1[base1];
          en1 = xfibers1[base1+2];
          base1 = xfibers1[base1+1]*cprank;

          memset(acc, 0, sizeof(real_t)*cprank);
          sc++;
          for(i2 = st1; i2 < en1; i2++)
            {
              base2 = i2*2;
              st2 = xfibers2[base2];
              en2 = xfibers2[base2+2];
              base2 = xfibers2[base2+1]*cprank;

              memset(acc2, 0, sizeof(real_t)*cprank);
              fc++;
              for(i3 = st2; i3 < en2; i3++)
                {
                  base3 = fibers[i3]*cprank;
                  for(j = 0; j < cprank; j++)
                    acc2[j] += mat3[base3+j]*vals[i3];
                }
            
              for(j = 0; j < cprank; j++)
                acc[j] += acc2[j]*mat2[base2+j];
            

            }
          for(j = 0; j < cprank; j++)
            matm[base0+j] += acc[j]*mat1[base1+j];
        
        }

    }
    
  //free(acc);
  //free(acc2);

  return fc;
}


void mttkrp(struct genst *gs, struct tensor *t, struct fibertensor *ft, struct csftensor *csftns, idx_t mode, real_t *matm)
{
  if(gs->fiber == 1)
    {
      if(gs->nmodes == 3)
        mttkrp_fiber_3(gs, ft, mode, matm);
      else if(gs->nmodes == 4)
        mttkrp_fiber_4(gs, ft, mode, matm);
      else
        printf("Not yet\n");


    }
  else if(gs->fiber == 2)
  {
      mttkrp_csf(gs, csftns, mode, matm);
  }
  else
    mttkrp_nnz(gs, t, mode, matm);
		
}



idx_t mttkrp_stats(struct genst *gs, struct tensor *t, struct fibertensor *ft, struct csftensor *csftns,  idx_t mode, real_t *matm)
{
  if(gs->fiber == 1)
    {
      if(gs->nmodes == 3)
		  return mttkrp_fiber_3_stats(gs, ft, mode, matm);
      else if(gs->nmodes == 4)
		  return mttkrp_fiber_4_stats(gs, ft, mode, matm);
      else
        {
          printf("Not yet\n");
          return -1;
        }
    }
  else if(gs->fiber == 2)
  {
      mttkrp_csf(gs, csftns, mode, matm);

      if(gs->nmodes == 3)
          return csftns->pt->nfibs[1];
      else if(gs->nmodes == 4)
          return csftns->pt->nfibs[2];
  }
  /*else
    mttkrp_nnz(gs, t, mode, matm);*/
  //else
    //return mttkrp_nnz_stats(t, mode, matm, niters);
		
}

void mttkrp_csf(struct genst *gs, struct csftensor *ft, idx_t mode, real_t *matm)
{

    idx_t od = csf_mode_to_depth(ft, mode);
    if(od == 0){
        mttkrp_csf_root(gs, ft, mode, matm);
    }
    else if (od == ft->nmodes-1){
        //leaf alg
        mttkrp_csf_leaf(gs, ft, mode, matm);
    }
    else{
        //internal alg
        mttkrp_csf_int(gs, ft, mode, matm); 
    }
}

void mttkrp_csf_root(struct genst *gs, struct csftensor *ft, idx_t mode, real_t *matm)
{
    //real_t *vals = ft->pt->vals;

    if(gs->nmodes == 3){
        mttkrp_csf_root_3m(gs, ft, matm);
        return;
    }
    else if(gs->nmodes == 4){
        mttkrp_csf_root_4m(gs, ft, matm);
        return;
    }



}


void mttkrp_csf_int(struct genst *gs, struct csftensor *ft, idx_t mode, real_t *matm)
{
   

 //   real_t *vals = ft->pt->vals;

    if(gs->nmodes == 3){
        mttkrp_csf_int_3m(gs, ft, matm);
        return;
    }
    else if(gs->nmodes == 4){
        mttkrp_csf_int_4m(gs, ft, matm, mode);
        return;
    }



}

void mttkrp_csf_leaf(struct genst *gs, struct csftensor *ft, idx_t mode, real_t *matm)
{
    

 //   real_t *vals = ft->pt->vals;

    if(gs->nmodes == 3){
        mttkrp_csf_leaf_3m(gs, ft, matm);
        return;
    }

    if(gs->nmodes == 4){
        mttkrp_csf_leaf_4m(gs, ft, matm);
        return;
    }

}
/*
 *@TODO allocate acc in a general data stucture
 *
 * */

void mttkrp_csf_root_3m(struct genst *gs, struct csftensor *ft, real_t *matm)
{

    idx_t i0, i1, i2, j, st0, st1, en0, en1, idxfirst;
    real_t v, *vals, *avals, *bvals, *mv, *av, *bv, vfirst, *acc;

    vals = ft->pt->vals;

    idx_t cprank = gs->cprank;
    idx_t *sptr = ft->pt->fptr[0];
    idx_t *fptr = ft->pt->fptr[1];

    idx_t *sids = ft->pt->fids[0]; //should be NULL in case of no-tiling, use the fptr indices instead
    idx_t *fids = ft->pt->fids[1];
    idx_t *inds = ft->pt->fids[2];


    avals = gs->mat[csf_depth_to_mode(ft, 1)];
    bvals = gs->mat[csf_depth_to_mode(ft, 2)];

    acc = gs->cpbuff;
    
    for(i0 = 0; i0 < ft->pt->nfibs[0]; i0++)
    {

         mv = matm + (sids[i0] * cprank);
        //for each fiber 
        for(i1 = sptr[i0]; i1< sptr[i0+1]; i1++){
            idxfirst = fptr[i1];
            vfirst = vals[idxfirst];

            bv = bvals +(inds[idxfirst] * cprank); 

            for(j = 0; j < cprank; j++)
                acc[j] = vfirst * bv[j];
            //for each nnz
            for (i2= fptr[i1]+1; i2< fptr[i1+1]; i2++){
                v = vals[i2];
                bv = bvals+(inds[i2] * cprank);
                for(j = 0; j < cprank; j++)
                    acc[j] += v * bv[j];
            }

            av = avals + (fids[i1] * cprank); 
            for(j = 0; j < cprank; j++)
                mv[j] += acc[j] * av[j];
        }
    }
}

void mttkrp_csf_int_3m(struct genst *gs, struct csftensor *ft, real_t *matm)
{
    idx_t i0, ii0, i1, i2, j, st0, st1, en0, en1, idxfirst;
    real_t v, *vals, *avals, *bvals, *mv, *av, *bv, vfirst, *acc;

    vals = ft->pt->vals;

    idx_t cprank = gs->cprank;
    idx_t *sptr = ft->pt->fptr[0];
    idx_t *fptr = ft->pt->fptr[1];

    idx_t *sids = ft->pt->fids[0]; //should be NULL in case of no-tiling, use the fptr indices instead
    idx_t *fids = ft->pt->fids[1];
    idx_t *inds = ft->pt->fids[2];
    
    avals = gs->mat[csf_depth_to_mode(ft, 0)];
    bvals = gs->mat[csf_depth_to_mode(ft, 2)];
    

    bv = bvals + ( 1 * cprank);    
    acc = gs->cpbuff;
    for(i0 = 0; i0 < ft->pt->nfibs[0]; i0++)
    {
        ii0 = ((sids == NULL) ? i0: sids[i0]); 
         av = avals + (ii0 * cprank); 
        //for each fiber 
         for(i1 = sptr[i0]; i1< sptr[i0+1]; i1++){
            idxfirst = fptr[i1];
            vfirst = vals[idxfirst];
            bv = bvals +(inds[idxfirst] * cprank); 

            for(j = 0; j < cprank; j++)
                acc[j] = vfirst * bv[j];
            //for each nnz
            for (i2= fptr[i1]+1; i2< fptr[i1+1]; i2++){
                v = vals[i2];
                bv = bvals + (inds[i2] * cprank);
                for(j = 0; j < cprank; j++)
                    acc[j] += v * bv[j]; 
            }
            mv = matm + (fids[i1] * cprank); 
            for(j = 0; j < cprank; j++)
                mv[j] += acc[j] * av[j];
        }
    }
}

void mttkrp_csf_leaf_3m(struct genst *gs, struct csftensor *ft, real_t *matm)
{
    idx_t i0, ii0, i1, i2, j, st0, st1, en0, en1;
    real_t v, *vals, *avals, *bvals, *mv, *av, *bv, *acc;

    vals = ft->pt->vals;

    idx_t cprank = gs->cprank;
    idx_t *sptr = ft->pt->fptr[0];
    idx_t *fptr = ft->pt->fptr[1];

    idx_t *sids = ft->pt->fids[0]; //should be NULL in case of no-tiling, use the fptr indices instead
    idx_t *fids = ft->pt->fids[1];
    idx_t *inds = ft->pt->fids[2];


    avals = gs->mat[csf_depth_to_mode(ft, 0)];
    bvals = gs->mat[csf_depth_to_mode(ft, 1)];

    acc = gs->cpbuff;
    
    for(i0 = 0; i0 < ft->pt->nfibs[0]; i0++)
    {
        ii0 = (sids == NULL) ? i0 : sids[i0];
         av = avals + (ii0 * cprank);
        //for each fiber 
        for(i1 = sptr[i0]; i1< sptr[i0+1]; i1++){ 

            bv = bvals +(fids[i1] * cprank); 
            for(j = 0; j < cprank; j++)
                acc[j] = av[j] * bv[j];
            //for each nnz
            for (i2= fptr[i1]; i2< fptr[i1+1]; i2++){
                v = vals[i2];
                mv = matm + (inds[i2] * cprank);
                for(j = 0; j < cprank; j++)
                    mv[j] += acc[j] * v;
            } 
        }
    }
}


void mttkrp_csf_root_4m(struct genst *gs, struct csftensor *ft, real_t *matm)
{

    idx_t i0, i1, i2, i3, j, idxfirst;
    real_t v, *vals, *avals, *bvals, *cvals, *mv, *av, *bv, *cv, vfirst, *acc, *acc2;

    vals = ft->pt->vals;

    idx_t cprank = gs->cprank;
    idx_t *ssptr = ft->pt->fptr[0];
    idx_t *sptr = ft->pt->fptr[1];
    idx_t *fptr = ft->pt->fptr[2];

    idx_t *ssids = ft->pt->fids[0];
    idx_t *sids = ft->pt->fids[1];
    idx_t *fids = ft->pt->fids[2];
    idx_t *inds = ft->pt->fids[3];


    avals = gs->mat[csf_depth_to_mode(ft, 1)];
    bvals = gs->mat[csf_depth_to_mode(ft, 2)];
    cvals = gs->mat[csf_depth_to_mode(ft, 3)];

    acc = gs->cpbuff;
    acc2 = gs->cpsqbuff;
    for(i0 = 0; i0 < ft->pt->nfibs[0]; i0++){

        mv = matm + (ssids[i0] *cprank);
        
        for(i1 = ssptr[i0]; i1 < ssptr[i0+1]; i1++)
        { 
            //for each slice
            av = avals + (sids[i1] * cprank);
            for(i2 = sptr[i1]; i2< sptr[i1+1]; i2++){
                idxfirst = fptr[i2];
                vfirst = vals[idxfirst];

                cv = cvals +(inds[idxfirst] * cprank); 

                for(j = 0; j < cprank; j++)
                    acc2[j] = vfirst * cv[j];
                //for each nnz
                for (i3= fptr[i2]+1; i3< fptr[i2+1]; i3++){
                    v = vals[i3];
                    cv = cvals + (inds[i3] * cprank);
                    for(j = 0; j < cprank; j++)
                        acc2[j] += v * cv[j];
                }

                bv = bvals + (fids[i2] * cprank); 
                for(j = 0; j < cprank; j++)
                    acc[j] += acc2[j] * bv[j];
            }

            for(j= 0; j< cprank; j++)
                mv[j] += acc[j] * av[j];
        }
    }
}

void mttkrp_csf_int_4m(struct genst *gs, struct csftensor *ft, real_t *matm, idx_t mode)
{
    idx_t i0, ii0, i1, i2, i3, j, st0, st1, en0, en1, idxfirst;
    real_t v, *vals, *avals, *bvals, *bbvals, *cvals, *mv, *av, *bv, *cv, vfirst, *acc, *acc2;

    idx_t depth = csf_mode_to_depth(ft, mode);
    vals = ft->pt->vals;

    idx_t cprank = gs->cprank;
    idx_t *ssptr = ft->pt->fptr[0];
    idx_t *sptr = ft->pt->fptr[1];
    idx_t *fptr = ft->pt->fptr[2];

    idx_t *ssids = ft->pt->fids[0];
    idx_t *sids = ft->pt->fids[1]; //should be NULL in case of no-tiling, use the fptr indices instead
    idx_t *fids = ft->pt->fids[2];
    idx_t *inds = ft->pt->fids[3];
    
    avals = gs->mat[csf_depth_to_mode(ft, 0)];
    bvals = gs->mat[csf_depth_to_mode(ft, 1)];
    bbvals= gs->mat[csf_depth_to_mode(ft, 2)];
    cvals = gs->mat[csf_depth_to_mode(ft, 3)];
    

    //bv = bvals + ( 1 * cprank);    
    acc = gs->cpbuff;
    acc2 = gs->cpsqbuff;
    for(i0 = 0; i0 < ft->pt->nfibs[0]; i0++)
    {
        ii0 = ((ssids == NULL) ? i0: ssids[i0]);

        av = avals + (ii0 * cprank); 
        //for each fiber 
         for(i1 = ssptr[i0]; i1< ssptr[i0+1]; i1++){
             if(depth == 2){
                bv = bvals + (sids[i1] * cprank);
                for(j = 0; j < cprank; j++)
                    acc[j] = bv[j]*av[j];
             }
             else
                 mv = matm + (sids[i1] * cprank);

             for(i2 = sptr[i1]; i2 < sptr[i1+1]; i2++){
                idxfirst = fptr[i1];
                vfirst = vals[idxfirst];
                cv = cvals +(inds[idxfirst] * cprank); 

                for(j = 0; j < cprank; j++)
                    acc2[j] = vfirst * cv[j];
                //for each nnz
                for (i3= fptr[i2]+1; i3< fptr[i2+1]; i3++){
                    v = vals[i3];
                    cv = cvals + (inds[i3] * cprank);
                    for(j = 0; j < cprank; j++)
                        acc2[j] += v * cv[j]; 
                }

                if(depth == 2){
                    mv = matm + (fids[i2] * cprank);
                    for(j= 0; j < cprank; j++)
                        mv[j] += acc[j] * acc2[j]; 
                }
                else{
                    bv = bbvals + (fids[i2] * cprank);
                    for(j = 0; j < cprank; j++)
                        acc[j] += bv[j] * av[j];
                }
             }
             if(depth == 1)
                for(j = 0; j < cprank; j++)
                     mv[j] = acc[j] * acc2[j];
        }
    }
}

void mttkrp_csf_leaf_4m(struct genst *gs, struct csftensor *ft, real_t *matm)
{
    idx_t i0, ii0, i1, i2, i3, j;
    real_t v, *vals, *avals, *bvals, *cvals, *mv, *av, *bv, *cv, *acc;

    vals = ft->pt->vals;

    idx_t cprank = gs->cprank;
    idx_t *ssptr = ft->pt->fptr[0];
    idx_t *sptr = ft->pt->fptr[1];
    idx_t *fptr = ft->pt->fptr[2];

    idx_t *ssids = ft->pt->fids[0];
    idx_t *sids = ft->pt->fids[1]; //should be NULL in case of no-tiling, use the fptr indices instead
    idx_t *fids = ft->pt->fids[2];
    idx_t *inds = ft->pt->fids[3];


    avals = gs->mat[csf_depth_to_mode(ft, 0)];
    bvals = gs->mat[csf_depth_to_mode(ft, 1)];
    cvals = gs->mat[csf_depth_to_mode(ft, 2)];

    acc = gs->cpbuff;
    
    for(i0 = 0; i0 < ft->pt->nfibs[0]; i0++)
    {
        ii0 = (ssids == NULL) ? i0 : ssids[i0];
         av = avals + (ii0 * cprank);
        //for each fiber 
        for(i1 = ssptr[i0]; i1< ssptr[i0+1]; i1++){
            bv = bvals +(sids[i1] * cprank);
            for(i2 = sptr[i1]; i2< sptr[i1+1]; i2++){

                cv = cvals +(fids[i2] * cprank); 
                for(j = 0; j < cprank; j++)
                    acc[j] = av[j] * bv[j] * cv[j];
                //for each nnz
                for (i3= fptr[i2]; i3< fptr[i2+1]; i3++){
                    v = vals[i3];
                    mv = matm + (inds[i3] * cprank);
                    for(j = 0; j < cprank; j++)
                        mv[j] += acc[j] * v;
                } 
            }
        }
    }
}

