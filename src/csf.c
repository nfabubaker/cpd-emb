#include "csf.h"
#include <limits.h>
#include <stdlib.h>
#include <stdio.h>

void init_csf(struct csftensor *ft)
{
    ft->dims = NULL;
    ft->dim_perm = NULL;
    ft->dim_iperm = NULL;
    ft->tile_dims = NULL;
    ft->pt = NULL;
}
void csf_get_sparsity_order(idx_t *gdims, idx_t *order, idx_t nmodes)
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


struct csftensor * csf_alloc(struct genst *gs, struct tensor *tns )
{

    struct csftensor *csftns = NULL;
    
    csftns = (struct csftensor *)malloc(sizeof(struct csftensor));
    init_csf(csftns);

    p_mk_csf(gs, tns, csftns, 0);

    return csftns;

}

void p_mk_csf(struct genst *gs, struct tensor *tns, struct csftensor *csftns, idx_t mode)
{
    idx_t i;

    csftns->nnz = gs->nnz;
    csftns->nmodes = gs->nmodes;

    //assign dims from tns (local ?) -- later
   /* for( i=0; i < tns->nmodes; i++){
    }*/
    csftns->dim_perm = (idx_t *)malloc(gs->nmodes*sizeof(int));
    csftns->dim_iperm = (idx_t *)malloc(gs->nmodes*sizeof(int));
    csf_get_sparsity_order(gs->gdims, csftns->dim_perm, gs->nmodes); 
    for(i = 0; i < gs->nmodes; i++)
        csftns->dim_iperm[csftns->dim_perm[i]] = i;

    csftns->ntiles = 1;
    p_csf_alloc_untiled(gs, tns, csftns);
}

void p_csf_alloc_untiled(struct genst *gs, struct tensor *tns, struct csftensor *csftns){
    idx_t i, nmodes = gs->nmodes;
    
    radixsort(tns->inds, tns->vals, tns->nnz, tns->nmodes, csftns->dim_perm, gs->ldims);
    checksort(tns->inds, tns->nnz, nmodes, csftns->dim_perm);
 
    csftns->pt = (struct csfsparsity *)malloc(sizeof(struct csfsparsity));
    struct csfsparsity *pt = csftns->pt;
    pt->nfibs = (idx_t *) malloc(nmodes * sizeof(int));
    pt->fptr  = (idx_t **)malloc(nmodes * sizeof(idx_t *));
    pt->fids  = (idx_t **)malloc(nmodes * sizeof(idx_t *));
    pt->fptr  [nmodes-1] = NULL;
    pt->nfibs [nmodes-1] = csftns->nnz;
    pt->fids  [nmodes-1]  = (idx_t *)malloc(csftns->nnz*sizeof(int));
    pt->vals            = (real_t*)malloc(csftns->nnz*sizeof(real_t));

    /* copy data of leaf nodes*/
    memcpy(pt->vals, tns->vals, tns->nnz * sizeof(real_t)); 

    //idx_t pid;
    //MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    idx_t *nptr = tns->inds +  csf_depth_to_mode(csftns, nmodes-1);
    for(i=0; i < tns->nnz; i++){
        //if(pid == 0 && i < 1000)
          //  printf("inds[%d] = %d\n", i, *nptr);
        pt->fids[nmodes-1][i] = *nptr;
        nptr += nmodes;
    }

    idx_t nnz_ptr[2];
    nnz_ptr[0] = 0;
    nnz_ptr[1] = tns->nnz;

    for(i=0; i< nmodes-1; i++)
        p_mk_fptr(tns, csftns, 0, nnz_ptr, i);

}

void p_mk_fptr(struct tensor *tns, struct csftensor *csftns, idx_t tile_id, idx_t * nnztile_ptr, idx_t mode)
{
    /*for tiling purposes (future maybe)*/
    idx_t nnzstart    = nnztile_ptr[tile_id];
    idx_t nnzend      = nnztile_ptr[tile_id+1];
    idx_t nnz         = nnzend - nnzstart;

    idx_t nmodes = tns->nmodes;

    if(mode == 0){
        p_mk_outerptr(tns, csftns, tile_id, nnztile_ptr);
        return ;
    }

    idx_t *tnsinds = tns->inds + (nnzstart * tns->nmodes + csf_depth_to_mode(csftns, mode));

    struct csfsparsity *pt = csftns->pt;// + tile_id;

    idx_t *fprev = pt->fptr[mode-1];

    idx_t i,j, nfibs = 0;
    //idx_t *cnfibs = (idx_t *)malloc( ( pt->nfibs[mode-1] + 1 ) * sizeof(int));

    for(i= 0; i < pt->nfibs[mode-1]; i++){
        ++nfibs;
        for(j=fprev[i]+1; j < fprev[i+1]; j++){
            if(tnsinds[j*nmodes] != tnsinds[(j-1)*nmodes])
                ++nfibs;
        }
    }

    pt->nfibs[mode] = nfibs;
    pt->fptr[mode] = (idx_t *)malloc((nfibs+1)*sizeof(int));
    pt->fptr[mode][0] = 0;
    pt->fids[mode] = (idx_t *)malloc(nfibs*sizeof(int));
    //pt->fids[mode][0] = tnsinds[fprev[0]*nmodes];

    idx_t *fp = pt->fptr[mode];
    idx_t *fi = pt->fids[mode];

    nfibs = 0;
    idx_t start, end;
    for(i= 0; i < pt->nfibs[mode-1]; i++){
        start = fprev[i] +1;
        end = fprev[i+1];
        fprev[i] = nfibs;
        fi[nfibs] = tnsinds[(start-1)*nmodes];
        fp[nfibs++] = start-1;
        for(j=start; j < end; j++){
            if(tnsinds[j*nmodes] != tnsinds[(j-1)*nmodes]){
                fi[nfibs] = tnsinds[j*nmodes];
                fp[nfibs++] = j;
            }
        }
    }

    fprev[pt->nfibs[mode-1]] = nfibs;
    fp[nfibs] = nnz;
}

void p_mk_outerptr(struct tensor *tns, struct csftensor *csftns, idx_t tile_id, idx_t *nnztile_ptr)
{
    /*for tiling purposes (future maybe)*/
    idx_t nnzstart    = nnztile_ptr[tile_id];
    idx_t nnzend      = nnztile_ptr[tile_id+1];
    idx_t nnz         = nnzend - nnzstart; 
    idx_t nmodes      = tns->nmodes;
    idx_t *tnsinds = tns->inds + ((nnzstart * tns->nmodes) + csf_depth_to_mode(csftns, 0));

    struct csfsparsity *pt = csftns->pt;// + tile_id;

    //idx_t *fprev = pt->fptr[mode-1];

    idx_t i,j, nfibs = 1;
    MPI_Barrier(MPI_COMM_WORLD);
 
    
    for(i= nnzstart+1; i < nnzend; i++){ 
        if(tnsinds[i*nmodes] != tnsinds[(i-1)*nmodes])
            ++nfibs;
    }
    pt->nfibs[0] = nfibs;
    pt->fptr[0] = (idx_t *)malloc((nfibs+1)*sizeof(int));
    pt->fptr[0][0] = 0;
    pt->fptr[0][nfibs] = nnz;
    pt->fids[0] = (idx_t *)malloc(nfibs*sizeof(int));
    pt->fids[0][0] = tnsinds[0];
    //pt->fids[0] = NULL;

    idx_t *fp = pt->fptr[0];
    idx_t *fi = pt->fids[0];

    nfibs = 1;
    for(i= nnzstart+1; i < nnzend; i++){ 
        if(tnsinds[i*nmodes] != tnsinds[(i-1)*nmodes]){
            fi[nfibs] = tnsinds[i*nmodes];
            fp[nfibs++] = i;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
 
}

void free_csf(struct csftensor *csftns)
{
    idx_t i,j;
    for(i=0; i< csftns->ntiles; i++){
        free(csftns->pt[i].nfibs);
        free(csftns->pt[i].vals);
        for(j=0; j < csftns->nmodes; j++){
            if(csftns->pt[i].fptr[j] != NULL)
                free(csftns->pt[i].fptr[j]);
            if(csftns->pt[i].fids[j] != NULL){
                free(csftns->pt[i].fids[j]);
            }
        }
        free(csftns->pt[i].fptr);
        free(csftns->pt[i].fids);
    }
    //free(csftns->dims);
    free(csftns->dim_perm);
    free(csftns->dim_iperm);
    if(csftns->tile_dims != NULL)
        free(csftns->tile_dims);
    free(csftns->pt);
    free(csftns);
}

