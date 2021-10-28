#include <stdio.h>
#include <time.h>
#include "cpd.h"
#include "mkl.h"
#include "mkl_spblas.h"
#include <stdlib.h>


void alloc_cp_buffers(struct genst *gs)
{
    gs->cpbuff = (real_t *)malloc(gs->cprank *sizeof(real_t));
    gs->cpsqbuff = (real_t *)malloc(gs->cprank*gs->cprank*sizeof(real_t));
}

void free_cp_buffers(struct genst *gs)
{
    if(gs->cpbuff != NULL){
        free(gs->cpbuff);
        gs->cpbuff = NULL;
    }
    if (gs->cpsqbuff != NULL) {
        free(gs->cpsqbuff);
        gs->cpsqbuff = NULL;
    }
}


void compute_inverse(struct genst *gs, idx_t mode, real_t *inverse)
{
    idx_t i, j, k, size, base;
    idx_t nmodes, cprank;
    real_t *uTu, inner;

    cprank = gs->cprank;
    size = cprank*cprank;
    for(i = 0; i < size; i++)
        inverse[i] = 1.0;


    nmodes = gs->nmodes;
    uTu = gs->uTu;
    for(i = 0; i < nmodes; i++) 
    {
        if(i != mode)
        {
            base = size*i;
            for(j = 0; j < size; j++) //CAN BE HALVED - SYM
                inverse[j] *= uTu[base+j]; 
        }
    }

    //Cholesky factorization
    //real_t *lmatrix = (real_t *)malloc(sizeof(real_t)*size);
    real_t *lmatrix = gs->cpsqbuff; 
    setreal_tzero(lmatrix, size);

    for(i = 0; i < cprank; i++)
        for(j = 0; j <= i; j++)
        {
            inner = 0.0;
            for(k = 0; k < j; k++)
                inner += lmatrix[i*cprank+k]*lmatrix[j*cprank+k];
            if(i == j)
                lmatrix[i*cprank+j] = sqrt(inverse[i*cprank+i]-inner);
            else
                lmatrix[i*cprank+j] = (inverse[i*cprank+j] - inner) / lmatrix[j*cprank+j];

        }

    //identity
    for(i = 0; i < size; i++)
        inverse[i] = 0.0;

    for(i = 0; i < cprank; i++)
        inverse[i*cprank+i] = 1.0;

    //forward solve

    /*	for(i = 0; i < cprank; i++)
        for(j = 0; j < cprank; j++)
        {
        inner = 0.0;
        for(k = 0; k < i; k++)
        inner += lmatrix[i*cprank+k]*inverse[k*cprank+j];
        inverse[i*cprank+j] = (inverse[i*cprank+j] - inner) / lmatrix[i*cprank+i];
        }
        */
    for(i = 0; i < cprank; i++)
        inverse[i] /= lmatrix[0];

    for(i = 1; i < cprank; i++)
    {	for(j = 0; j < i; j++)
        for(k = 0; k < cprank; k++)
            inverse[i*cprank+k] -= lmatrix[i*cprank+j]*inverse[j*cprank+k]; 

        for(k = 0; k < cprank; k++)
            inverse[i*cprank+k] /= lmatrix[i*cprank+i];
    }

    //transpose
    for(i = 0; i < cprank; i++)
        for(j = i+1; j < cprank; j++)
        {
            lmatrix[i*cprank+j] = lmatrix[j*cprank+i];
            lmatrix[j*cprank+i] = 0.0;
        }


    //backward
    /*	for(i = cprank-1; i >= 0; i--)
        for(j = 0; j < cprank; j++)
        {
        inner = 0.0;
        for(k = i+1; k < cprank; k++)
        inner += lmatrix[i*cprank+k]*inverse[k*cprank+j];
        inverse[i*cprank+j] = (inverse[i*cprank+j] - inner) / lmatrix[i*cprank+i];
        }
        */

    i = cprank-1;
    for(k = 0; k < cprank; k++)
        inverse[i*cprank+k] /= lmatrix[i*cprank+i];

    idx_t row;
    for(row = 2; row <= cprank; row++)
    {
        i = cprank - row;
        for(j = i+1; j < cprank; j++)
            for(k = 0; k < cprank; k++)
                inverse[i*cprank+k] -= lmatrix[i*cprank+j]*inverse[j*cprank+k];

        for(k = 0; k < cprank; k++)
            inverse[i*cprank+k] /= lmatrix[i*cprank+i];
    }

    //free(lmatrix);

}

void matrix_multiply(struct genst *gs, idx_t mode, real_t *matm, real_t *inverse)
{
    idx_t i, j, k, cprank, dim;
    real_t *matw, inner, alpha, beta;

    matw = gs->mat[mode];
    dim = gs->ldims[mode];
    cprank = gs->cprank;

    //  	for(i = 0; i < dim; i++)
    //  	{ 
    //  		for(j = 0; j < cprank; j++) 
    //  		{ 
    //  			inner = 0; 
    //  			for(k = 0; k < cprank; k++) 
    //  				inner += matm[i*cprank+k]*inverse[k*cprank+j];
    //  			matw[i*cprank+j] = inner; 
    //  		} 
    //  	}

    alpha = 1.0;
    beta = 0.0;
#if valsize == 32
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, cprank, cprank, alpha, matm, cprank, inverse, cprank, beta, matw, cprank);
#elif valsize == 64
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, cprank, cprank, alpha, matm, cprank, inverse, cprank, beta, matw, cprank);
#endif
}

void expand_and_normalize(struct genst *gs, ecomm *ec, idx_t mode, real_t *lambda)
{
    idx_t i, j, dim, cprank, ptr;
    real_t *mat, v, w, *locallambda;


    cprank = gs->cprank;
    mat = gs->mat[mode];
    dim = gs->ldims[mode];

    //locallambda = (real_t *)malloc(cprank*sizeof(real_t));
    locallambda = gs->cpbuff;
    setreal_tzero(locallambda, cprank);

    for(i = 0; i < cprank; i++)
    {

#if valsize == 32
            locallambda[i]=cblas_sdot(dim, &mat[i], cprank, &mat[i], cprank);
#elif valsize == 64
            locallambda[i]=cblas_ddot(dim, &mat[i], cprank, &mat[i], cprank);
#endif

        /* 		v = 0; */
        /* 		for(j = 0; j < dim; j++) */
        /* 		{ */
        /* 			w = mat[j*cprank+i]; */
        /* 			v += w*w; */
        /* 		} */
        /* 		locallambda[i] = v; */

    }


    ecomm_communicate_allreduce(ec, locallambda, lambda, gs->cprank, gs->cprank);
    //  MPI_Allreduce(locallambda, lambda, cprank, MPI_REAL_T, MPI_SUM, MPI_COMM_WORLD);

    for(i = 0; i < cprank; i++)
        lambda[i] = sqrt(lambda[i]);


    ptr = 0;
    for(i = 0; i < dim; i++)
        for(j = 0; j < cprank; j++)
            mat[ptr++] /= lambda[j];

    //free(locallambda);


}

void normalize2(struct genst *gs, idx_t mode, real_t *lambda)
{
    idx_t i, j, dim, cprank, ptr;
    real_t *mat, v, w, *locallambda;


    cprank = gs->cprank;
    mat = gs->mat[mode];
    dim = gs->ldims[mode];

    //locallambda = (real_t *)malloc(cprank*sizeof(real_t));
    locallambda = gs->cpbuff;
    setreal_tzero(locallambda, cprank);

    for(i = 0; i < cprank; i++)
    {

#if valsize == 32
            locallambda[i]=cblas_sdot(dim, &mat[i], cprank, &mat[i], cprank);
#elif valsize == 64
            locallambda[i]=cblas_ddot(dim, &mat[i], cprank, &mat[i], cprank);
#endif
        /* 		v = 0; */
        /* 		for(j = 0; j < dim; j++) */
        /* 		{ */
        /* 			w = mat[j*cprank+i]; */
        /* 			v += w*w; */
        /* 		} */
        /* 		locallambda[i] = v; */

    }


    MPI_Allreduce(locallambda, lambda, cprank, MPI_REAL_T, MPI_SUM, MPI_COMM_WORLD);

    for(i = 0; i < cprank; i++)
        lambda[i] = sqrt(lambda[i]);


    ptr = 0;
    for(i = 0; i < dim; i++)
        for(j = 0; j < cprank; j++)
            mat[ptr++] /= lambda[j];
}

void normalizemax(struct genst *gs, idx_t mode, real_t *lambda)
{
    idx_t i, j, dim, cprank, ptr;
    real_t *mat, w, *locallambda;


    cprank = gs->cprank;
    mat = gs->mat[mode];
    dim = gs->ldims[mode];

    //locallambda = (real_t *)malloc(cprank*sizeof(real_t));
    locallambda = gs->cpbuff;
    setreal_tzero(locallambda, cprank);

    ptr = 0;
    for(i = 0; i < dim; i++)
    {
        for(j = 0; j < cprank; j++)
        {
            w = mat[ptr++];
            if(locallambda[j] < w)
                locallambda[j] = w;
        }
    }


    MPI_Allreduce(locallambda, lambda, cprank, MPI_REAL_T, MPI_MAX,  MPI_COMM_WORLD);

    for(i = 0; i < cprank; i++)
        if(lambda[i] < 1.0)
            lambda[i] = 1.0;
    ptr = 0;
    for(i = 0; i < dim; i++)
        for(j = 0; j < cprank; j++)
            mat[ptr++] /= lambda[j];

    //free(locallambda);
}


void compute_aTa(struct genst *gs, idx_t mode)
{
    idx_t i, j, k, size, ldim, ptr1, ptr2;
    idx_t cprank;
    real_t *buffer, *mat, inner, alpha, beta;

    cprank = gs->cprank;
    size = cprank*cprank;
    ldim = gs->ldims[mode];
    mat = gs->mat[mode];

    //buffer = (real_t *)malloc(sizeof(real_t)*size);
    buffer = gs->cpsqbuff;
    setreal_tzero(buffer, size);

    /* 	for(i = 0; i < cprank; i++) */
    /* 		for(j = i; j < cprank; j++) */
    /* 		{ */
    /* 			inner = 0; */
    /* 			ptr1 = i; */
    /* 			ptr2 = j;      */
    /* 			for(k = 0; k < ldim; k++) */
    /* 			{ */
    /* 				inner += mat[ptr1]*mat[ptr2]; */
    /* 				ptr1 += cprank; */
    /* 				ptr2 += cprank; */
    /* 			} */

    /* 			buffer[i*cprank+j] = inner; */
    /* 			if(i != j) */
    /* 				buffer[j*cprank+i] = inner; */
    /* 		} */

    alpha = 1.0;
    beta = 0.0;
#if valsize == 32
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                cprank, cprank, ldim, alpha, mat, cprank, mat, cprank, beta, buffer, cprank);
#elif valsize == 64
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                cprank, cprank, ldim, alpha, mat, cprank, mat, cprank, beta, buffer, cprank);
#endif

    if(gs->comm_type != EMB)
        MPI_Allreduce(buffer, &(gs->uTu[mode*size]), size, MPI_REAL_T, MPI_SUM, MPI_COMM_WORLD);

}

real_t compute_input_norm(real_t *vals, idx_t lnnz)
{
    idx_t i;
    real_t mynorm, norm, v; 

    mynorm = 0;

    //vals = t->vals;
    //lnnz = t->nnz;
    for(i = 0; i < lnnz; i++)
    {
        v = vals[i];
        mynorm += v*v;
    }

    MPI_Allreduce(&mynorm, &norm, 1, MPI_REAL_T, MPI_SUM, MPI_COMM_WORLD);

    return norm;
}


real_t compute_fit(struct genst *gs, real_t *matm, real_t inputnorm, idx_t mode, real_t *lambda)
{
    idx_t i, j, ptr, size, ldim;
    idx_t nmodes, cprank;
    real_t decompnorm, myinner, inner, residual, *buffer, *uTu, *mat;

    ldim = gs->ldims[mode];
    mat = gs->mat[mode];
    uTu = gs->uTu;
    nmodes = gs->nmodes;
    cprank = gs->cprank;
    size = cprank*cprank;

    //compute decomposition norm
    decompnorm = 0.0;
    buffer = (real_t *)malloc(sizeof(real_t)*size);
    //buffer = gs->cpsqbuff;
    for(i = 0; i < size; i++)
        buffer[i] = 1.0;

    ptr = 0;
    for(i = 0; i < nmodes; i++)
        for(j = 0; j < size; j++)
            buffer[j] *= uTu[ptr++]; 

    for(i = 0; i < cprank; i++)
        for(j = 0; j < cprank; j++)
            decompnorm += buffer[i*cprank+j]*lambda[i]*lambda[j];

    decompnorm = fabs(decompnorm);

    //compute inner product (of input tensor and decomposition)
    for(i = 0; i < cprank; i++)
        buffer[i] = 0.0;

    ptr = 0;
    for(i = 0; i < ldim; i++)
        for(j = 0; j < cprank; j++)
            buffer[j] += mat[ptr]*matm[ptr++];

    myinner = 0.0;
    for(i = 0; i < cprank; i++)
        myinner += buffer[i]*lambda[i];

    free(buffer);

    MPI_Allreduce(&myinner, &inner, 1, MPI_REAL_T, MPI_SUM, MPI_COMM_WORLD);

    residual = sqrt(inputnorm + decompnorm - 2*inner);	

    return 1 - (residual/sqrt(inputnorm)); 
}


void cp_als_stats_fg(struct genst *gs, struct tensor *t, struct fibertensor *ft, struct csftensor *csftns, idx_t niters, double *mttkrptime, double *comm1time, double *comm2time, idx_t *cnt_st)
{

    idx_t it, i, maxldim;
    idx_t nmodes, pid;
    double start, end; 
    tmr_t *mttkrpT, *comm1T, *comm2T;
    pid = gs->mype; 

    nmodes = gs->nmodes;

    mttkrpT = malloc(sizeof(*mttkrpT) * nmodes);  
    comm1T = malloc(sizeof(*comm1T) * nmodes);  
    comm2T = malloc(sizeof(*comm2T) * nmodes);  
    for (i = 0; i < nmodes; ++i) {
        mttkrpT[i].elapsed = 0.0; comm1T[i].elapsed = 0.0; comm2T[i].elapsed=0.0;

    }

    maxldim = gs->ldims[0];
    for(i = 1; i < nmodes; i++)
        if(gs->ldims[i] > maxldim)
            maxldim = gs->ldims[i];
    alloc_cp_buffers(gs);
    for (it = 0; it < niters; it++){
        for(i = 0; i < nmodes; i++)
        {
            setreal_tzero(gs->matm, gs->ldims[i]*gs->cprank);

            MPI_Barrier(MPI_COMM_WORLD);
            start_timer(&mttkrpT[i]);
            cnt_st[i] += mttkrp_stats(gs,t, ft, csftns, i, gs->matm);
            MPI_Barrier(MPI_COMM_WORLD);
            stop_timer(&mttkrpT[i]); 

            MPI_Barrier(MPI_COMM_WORLD);
            start_timer(&comm1T[i]);
            receive_partial_products_fg(gs, i, gs->matm);
            MPI_Barrier(MPI_COMM_WORLD);
            stop_timer(&comm1T[i]); 

            MPI_Barrier(MPI_COMM_WORLD);
            start_timer(&comm2T[i]);
            send_updated_rows_fg(gs, i);         
            MPI_Barrier(MPI_COMM_WORLD);
            stop_timer(&comm2T[i]); 

        }
    }

    for (i = 0; i < nmodes; i++){
        comm1time[i] = comm1T[i].elapsed / niters;
        mttkrptime[i] = mttkrpT[i].elapsed/niters;
        comm2time[i] = comm2T[i].elapsed / niters;

        cnt_st[i] /= niters;
        //if(gs->fiber != 2)
        cnt_st[i] += gs->nnz;

    }
    free(comm1T); free(mttkrpT); free(comm2T);

    free_cp_buffers(gs);
}

void cp_als_fg(struct genst *gs,struct tensor *t,  struct fibertensor *ft, struct csftensor *csftns, idx_t niters, double *cptime)
{

    idx_t it, i, maxldim;
    real_t *lambda, inputnorm, fit, oldfit, *inverse, *vals;

    idx_t nmodes = gs->nmodes;
    lambda = (real_t *)malloc(gs->cprank*sizeof(real_t));
    if(gs->fiber ==1)
        vals = ft->lvals;
    else if(gs->fiber == 2)
        vals = csftns->pt->vals;
    else
        vals = t->vals;

    inputnorm = compute_input_norm(vals, gs->nnz);

    maxldim = gs->ldims[0];
    for(i = 1; i < nmodes; i++)
        if(gs->ldims[i] > maxldim)
            maxldim = gs->ldims[i];

    inverse = (real_t *)malloc(sizeof(real_t)*gs->cprank*gs->cprank);

    alloc_cp_buffers(gs); 

    tmr_t cpT;
    cpT.elapsed = 0.0;
    MPI_Barrier(MPI_COMM_WORLD);
    start_timer(&cpT);
    for(it = 0; it < niters; it++)
    {
        for(i = 0; i < nmodes; i++)
        {
            setreal_tzero(gs->matm, gs->ldims[i]*gs->cprank);
            mttkrp(gs, t, ft, csftns, i, gs->matm);
            MPI_Barrier(MPI_COMM_WORLD);

            receive_partial_products_fg(gs, i, gs->matm);

            compute_inverse(gs, i, inverse);

            matrix_multiply(gs, i, gs->matm, inverse);

            if(it == 0)
                normalize2(gs, i, lambda);
            else
                normalizemax(gs, i, lambda);

            send_updated_rows_fg(gs, i); 

            compute_aTa(gs, i);

        }
        fit = compute_fit(gs, gs->matm, inputnorm, nmodes-1, lambda);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    stop_timer(&cpT);
    *cptime = cpT.elapsed / niters;
    free_cp_buffers(gs);
    free(inverse);
}

void cp_als_fg_emb(struct genst *gs,struct tensor *t,  struct fibertensor *ft, struct csftensor *csftns, idx_t niters, double *cptime)
{

    idx_t it, i, maxldim;
    idx_t nmodes; 
    real_t *lambda, inputnorm, fit, oldfit, *inverse, *vals;

    nmodes = gs->nmodes;
    lambda = (real_t *)malloc(gs->cprank*sizeof(real_t));
    if(gs->fiber ==1)
        vals = ft->lvals;
    else if(gs->fiber == 2)
        vals = csftns->pt->vals;
    else
        vals = t->vals;

    inputnorm = compute_input_norm(vals, gs->nnz);

    inverse = (real_t *)malloc(sizeof(real_t)*gs->cprank*gs->cprank);

    alloc_cp_buffers(gs); 

    tmr_t cpT;
    cpT.elapsed = 0.0;
    MPI_Barrier(MPI_COMM_WORLD);
    start_timer(&cpT);
    for(it = 0; it < niters; it++)
    {
        for(i = 0; i < nmodes; i++)
        {
            setreal_tzero(gs->matm, gs->ldims[i]*gs->cprank);
            mttkrp(gs, t, ft, csftns, i, gs->matm);

            idx_t mode_toReduce = ((i-1)+ nmodes) % nmodes;
            idx_t s = gs->cprank * gs->cprank;
            ecomm_communicate_allreduce(gs->comm->ec[i*2], gs->cpsqbuff, &gs->uTu[s * mode_toReduce], gs->cprank, s);

            compute_inverse(gs, i, inverse);

            matrix_multiply(gs, i, gs->matm, inverse);

            /* TODO FIXME what about normalizemax ?? currently normalize2 is only supported */       
            /*           if(it == 0)
             *             normalize2(gs, i, lambda);
             *           else
             *             normalizemax(gs, i, lambda);
             */

            expand_and_normalize(gs, gs->comm->ec[i*2+1], i, lambda);

            compute_aTa(gs, i);

        }
        fit = compute_fit(gs, gs->matm, inputnorm, nmodes-1, lambda);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    stop_timer(&cpT);
    *cptime = cpT.elapsed / niters;

    free_cp_buffers(gs);
    free(inverse);
}

void cp_als_fg_emb_time(struct genst *gs, struct tensor *t, struct fibertensor *ft, struct csftensor *csftns, idx_t niters, double *cptime, double *mmtime, double *otherstime , double *mttkrptime, double *comm1time, double * comm2time)
{

    idx_t it, i, maxldim;
    real_t *lambda, inputnorm, fit, oldfit, *inverse, *vals;

    idx_t nmodes = gs->nmodes;
    lambda = (real_t *)malloc(gs->cprank*sizeof(real_t));
    tmr_t *mttkrpT, *comm1T, *comm2T, *mmT, *othersT;
    mttkrpT = malloc(sizeof(*mttkrpT) * nmodes);  
    comm1T = malloc(sizeof(*comm1T) * nmodes);  
    comm2T = malloc(sizeof(*comm2T) * nmodes);  
    mmT = malloc(sizeof(*comm2T) * nmodes);  
    othersT = malloc(sizeof(*comm2T) * nmodes);  

    for (i = 0; i < nmodes; ++i) {
        mttkrpT[i].elapsed = 0.0; comm1T[i].elapsed = 0.0; comm2T[i].elapsed=0.0;
        mmT[i].elapsed = 0.0; othersT[i].elapsed = 0.0;

    }
    if(gs->fiber ==1)
        vals = ft->lvals;
    else if(gs->fiber == 2)
        vals = csftns->pt->vals;
    else
        vals = t->vals;

    inputnorm = compute_input_norm(vals, gs->nnz);

    maxldim = gs->ldims[0];
    for(i = 1; i < nmodes; i++)
        if(gs->ldims[i] > maxldim)
            maxldim = gs->ldims[i];

    inverse = (real_t *)malloc(sizeof(real_t)*gs->cprank*gs->cprank);

    alloc_cp_buffers(gs); 

    tmr_t cpT;
    cpT.elapsed = 0.0;
    MPI_Barrier(MPI_COMM_WORLD);
    start_timer(&cpT);
    for(it = 0; it < niters; it++)
    {
        for(i = 0; i < nmodes; i++)
        {
            setreal_tzero(gs->matm, gs->ldims[i]*gs->cprank);
            MPI_Barrier(MPI_COMM_WORLD);
            start_timer(&mttkrpT[i]);
            mttkrp(gs, t, ft, csftns, i, gs->matm);
            MPI_Barrier(MPI_COMM_WORLD);
            stop_timer(&mttkrpT[i]);

            start_timer(&comm1T[i]);
            idx_t mode_toReduce = ((i-1)+ nmodes) % nmodes;
            idx_t s = gs->cprank * gs->cprank;
            ecomm_communicate_allreduce(gs->comm->ec[i*2], gs->cpsqbuff, &gs->uTu[s * mode_toReduce], gs->cprank, s);
            MPI_Barrier(MPI_COMM_WORLD);
            stop_timer(&comm1T[i]);

            start_timer(&othersT[i]);
            compute_inverse(gs, i, inverse);
            MPI_Barrier(MPI_COMM_WORLD);
            stop_timer(&othersT[i]);

            start_timer(&mmT[i]);
            matrix_multiply(gs, i, gs->matm, inverse);
            MPI_Barrier(MPI_COMM_WORLD);
            stop_timer(&mmT[i]);

            start_timer(&othersT[i]);
            /* TODO FIXME what about normalizemax ?? currently normalize2 is only supported */       
            /*           if(it == 0)
             *             normalize2(gs, i, lambda);
             *           else
             *             normalizemax(gs, i, lambda);
             */

            expand_and_normalize(gs, gs->comm->ec[i*2+1], i, lambda);
            MPI_Barrier(MPI_COMM_WORLD);
            stop_timer(&othersT[i]);

            start_timer(&comm2T[i]);
            expand_and_normalize(gs, gs->comm->ec[i*2+1], i, lambda);
            MPI_Barrier(MPI_COMM_WORLD);
            stop_timer(&comm2T[i]);

            start_timer(&othersT[i]);
            compute_aTa(gs, i);
            MPI_Barrier(MPI_COMM_WORLD);
            stop_timer(&othersT[i]);

        }
        fit = compute_fit(gs, gs->matm, inputnorm, nmodes-1, lambda);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    stop_timer(&cpT);
    *cptime = cpT.elapsed / niters;
    for (i = 0; i < nmodes; ++i) {
        mttkrptime[i] = mttkrpT[i].elapsed/ niters;
        comm1time[i] =   comm1T[i].elapsed/ niters;
        comm2time[i] =   comm2T[i].elapsed/ niters;
        otherstime[i] = othersT[i].elapsed/ niters;
        mmtime[i] =         mmT[i].elapsed / niters;
    }
    free_cp_buffers(gs);

    free(inverse);
}
void cp_als_fg_time(struct genst *gs, struct tensor *t, struct fibertensor *ft, struct csftensor *csftns, idx_t niters, double *cptime, double *mmtime, double *otherstime , double *mttkrptime, double *comm1time, double * comm2time)
{

    idx_t it, i, maxldim;
    real_t *lambda, inputnorm, fit, oldfit, *inverse, *vals;

    idx_t nmodes = gs->nmodes;
    lambda = (real_t *)malloc(gs->cprank*sizeof(real_t));
    tmr_t *mttkrpT, *comm1T, *comm2T, *mmT, *othersT;
    mttkrpT = malloc(sizeof(*mttkrpT) * nmodes);  
    comm1T = malloc(sizeof(*comm1T) * nmodes);  
    comm2T = malloc(sizeof(*comm2T) * nmodes);  
    mmT = malloc(sizeof(*comm2T) * nmodes);  
    othersT = malloc(sizeof(*comm2T) * nmodes);  

    for (i = 0; i < nmodes; ++i) {
        mttkrpT[i].elapsed = 0.0; comm1T[i].elapsed = 0.0; comm2T[i].elapsed=0.0;
        mmT[i].elapsed = 0.0; othersT[i].elapsed = 0.0;

    }
    if(gs->fiber ==1)
        vals = ft->lvals;
    else if(gs->fiber == 2)
        vals = csftns->pt->vals;
    else
        vals = t->vals;

    inputnorm = compute_input_norm(vals, gs->nnz);

    maxldim = gs->ldims[0];
    for(i = 1; i < nmodes; i++)
        if(gs->ldims[i] > maxldim)
            maxldim = gs->ldims[i];

    inverse = (real_t *)malloc(sizeof(real_t)*gs->cprank*gs->cprank);

    alloc_cp_buffers(gs); 

    tmr_t cpT;
    cpT.elapsed = 0.0;
    MPI_Barrier(MPI_COMM_WORLD);
    start_timer(&cpT);
    for(it = 0; it < niters; it++)
    {
        for(i = 0; i < nmodes; i++)
        {
            setreal_tzero(gs->matm, gs->ldims[i]*gs->cprank);
            MPI_Barrier(MPI_COMM_WORLD);
            start_timer(&mttkrpT[i]);
            mttkrp(gs, t, ft, csftns, i, gs->matm);
            MPI_Barrier(MPI_COMM_WORLD);
            stop_timer(&mttkrpT[i]);

            start_timer(&comm1T[i]);
            receive_partial_products_fg(gs, i, gs->matm);
            MPI_Barrier(MPI_COMM_WORLD);
            stop_timer(&comm1T[i]);

            start_timer(&othersT[i]);
            compute_inverse(gs, i, inverse);
            MPI_Barrier(MPI_COMM_WORLD);
            stop_timer(&othersT[i]);

            start_timer(&mmT[i]);
            matrix_multiply(gs, i, gs->matm, inverse);
            MPI_Barrier(MPI_COMM_WORLD);
            stop_timer(&mmT[i]);

            start_timer(&othersT[i]);
            if(it == 0)
                normalize2(gs, i, lambda);
            else
                normalizemax(gs, i, lambda);
            MPI_Barrier(MPI_COMM_WORLD);
            stop_timer(&othersT[i]);

            start_timer(&comm2T[i]);
            send_updated_rows_fg(gs, i); 
            MPI_Barrier(MPI_COMM_WORLD);
            stop_timer(&comm2T[i]);

            start_timer(&othersT[i]);
            compute_aTa(gs, i);
            MPI_Barrier(MPI_COMM_WORLD);
            stop_timer(&othersT[i]);

        }
        fit = compute_fit(gs, gs->matm, inputnorm, nmodes-1, lambda);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    stop_timer(&cpT);
    *cptime = cpT.elapsed / niters;
    for (i = 0; i < nmodes; ++i) {
        mttkrptime[i] = mttkrpT[i].elapsed/ niters;
        comm1time[i] =   comm1T[i].elapsed/ niters;
        comm2time[i] =   comm2T[i].elapsed/ niters;
        otherstime[i] = othersT[i].elapsed/ niters;
        mmtime[i] =         mmT[i].elapsed / niters;
    }
    free_cp_buffers(gs);
    free(inverse);
}
