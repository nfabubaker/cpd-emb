/**
 * @author      : Nabil Abubaker (nabil.abubaker@bilkent.edu.tr)
 * @file        : partition
 * @created     : Friday Oct 29, 2021 16:27:25 +03
 */

#include "partition.h"
#include "mpi.h"
#include "util.h"
#include <stdlib.h>
#include <assert.h>
#include <string.h>



static idx_t get_pid_from_ind(idx_t ind, idx_t rows_pp, idx_t rows_ppR){
    idx_t pp = ((rows_pp+1) * rows_ppR);
    if(ind >= pp) {
        return rows_ppR + ((ind-pp) / rows_pp);
    }
    else
        return ind / (rows_pp+1);
}

/* parition the rows randomly such that each row goes to one of the processors that update it*/
void partition_rows_rand(const tensor *t, const genst *gs, const idx_t mode, idx_t *interpart){

    /* defs
     * rCnts : count for each processor how many rows I update
     * rCnts_all: how many rows I own each processor update
     * tcnts: temporary counts, multi-purpose
     * rrTO: rows to send to each processor
     * rrFROM: rows to recv from each processor
     * rows_p_cnt: counts of processors updating row i
     * rows_p: list of processors updating row i;
     * indsmap: map global inds to local
     * rindsmap: local to global
     * */
    idx_t i,j,k, dimSize, npes, mype, *rCnts, *rCnts_all, *tcnts, **rrTO, **rrFROM, *rows_p_cnt, **rows_p, *indsmap, *rindsmap,   tp, nmodes, rows_ppR, *linterpart;

    idx_t rows_pp, myrowsCnt;

    npes = gs->npes;
    mype = gs->mype;
    dimSize = gs->gdims[mode];
    nmodes = gs->nmodes;
    rows_pp = dimSize / gs->npes; 
    rows_ppR = dimSize % gs->npes; 
    myrowsCnt = (mype < rows_ppR ? rows_pp + 1 : rows_pp);
    idx_t ss = mype < rows_ppR ? mype  * (rows_pp+1) : (rows_ppR * (rows_pp+1))+((mype-rows_ppR )* rows_pp) ;
    idx_t se = ss + myrowsCnt; /* index of my last global row */
    char *pflags;

    MPI_Request *reqsts;

    /* each processor should cound how many entries he will send to each other processor:
     *
     * every row I update should be sent to processor p_k that owns that row.
     * */
#ifdef NA_DBG
    MPI_Barrier(MPI_COMM_WORLD);
    na_log(dbgfp, "hello from partition_rows_rand: rows_pp=%zu, rows_ppR=%zu dimSize=%zu, myrowsCnt=%zu, ss=%zu se=%zu\n", rows_pp, rows_ppR, dimSize, myrowsCnt, ss, se);
#endif

    rCnts = malloc(sizeof(*rCnts) * npes); 
    pflags = calloc(dimSize, sizeof(*pflags)); 
    idx_t tcntsSize = myrowsCnt > npes ? myrowsCnt : npes;
    tcnts = malloc(sizeof(*tcnts) * tcntsSize); 
    rCnts_all = malloc(sizeof(*rCnts) * npes); 
    rrTO = malloc(sizeof(*rrTO) * npes); 
    rrFROM = malloc(sizeof(*rrFROM) * npes); 
    linterpart = malloc(sizeof(*linterpart) * myrowsCnt);
    setintzero(rCnts, npes);
#ifdef NA_DBG
    MPI_Barrier(MPI_COMM_WORLD);
    na_log(dbgfp, "hello from partition_rows_rand: after initial allocs\n");
#endif

    for (i = 0; i < t->nnz; ++i) {
        /* decide which processor */
        //tp = (t->inds[i * nmodes + mode] / rows_pp >= npes ? 0 :t->inds[i * nmodes + mode] / rows_pp );
        idx_t ind = t->inds[i*nmodes+mode];
        tp = get_pid_from_ind(ind, rows_pp, rows_ppR);
        assert(tp < npes && "ERROR: target processor is larger than npes ??");
        if (pflags[ind] == 0) {
            rCnts[tp]++; 
            pflags[ind] = 1;
        }
    }
    /* allocate an array for all */
    for (i = 0; i < npes; ++i) {
        if(rCnts[i] > 0)
            rrTO[i] = malloc(sizeof(*rrTO[i]) * rCnts[i]);
    }
#ifdef NA_DBG
    MPI_Barrier(MPI_COMM_WORLD);
    na_log(dbgfp, "\t rCnts ready, rrTO allocated\n");
#endif
    setintzero(rCnts_all, npes);
    rCnts_all[mype] = rCnts[mype];
    MPI_Alltoall(rCnts, 1, MPI_IDX_T, rCnts_all, 1, MPI_IDX_T, MPI_COMM_WORLD);

#ifdef NA_DBG
    MPI_Barrier(MPI_COMM_WORLD);
    na_log(dbgfp, "\t after alltot-all\n");
#endif
    /* allocate recv buffers */
    for (i = 0; i < npes; ++i) {
        if (i != mype && rCnts_all[i] > 0) {
            rrFROM[i] = malloc(sizeof(*rrFROM[i]) * rCnts_all[i]);
        }
    }
#ifdef NA_DBG
    MPI_Barrier(MPI_COMM_WORLD);
    na_log(dbgfp, "\t recv buffers allocted \n");
#endif
    /* fill sent buffers */
    setintzero(tcnts, tcntsSize);
    memset(pflags, 0, sizeof(*pflags)*dimSize);
    for (i = 0; i < t->nnz; ++i) {
        idx_t ind = t->inds[i*nmodes+mode];
        tp = get_pid_from_ind(ind, rows_pp, rows_ppR);
        assert(tp < npes && "ERROR: target processor is larger than npes ??");
        assert(rCnts[tp] > 0 && "ERROR: rCnts[tp] should be larger than 0");
        assert(tcnts[tp] <= rCnts[tp]);
        if(pflags[ind] == 0){
            rrTO[tp][tcnts[tp]++] = ind;
            pflags[ind] = 1;
        }
    }
#ifdef NA_DBG
    MPI_Barrier(MPI_COMM_WORLD);
    na_log(dbgfp, "\t  send buffers filled\n");
#endif

    MPI_Request req; 
    for (i = 0; i < npes; ++i) {
        if(i != mype && rCnts[i] > 0){
            MPI_Isend(rrTO[i], rCnts[i], MPI_IDX_T, i, 66, MPI_COMM_WORLD, &req);
        }
    }
#ifdef NA_DBG
    MPI_Barrier(MPI_COMM_WORLD);
    na_log(dbgfp, "\t Isends issued\n");
#endif

    for (i = 0; i < npes; ++i) {
        if(i != mype && rCnts_all[i] > 0){
            MPI_Recv(rrFROM[i], rCnts_all[i], MPI_IDX_T, i, 66, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
#ifdef NA_DBG
    MPI_Barrier(MPI_COMM_WORLD);
    na_log(dbgfp, "\t recvs done, rrFROM is ready\n");
#endif

    /* setup indsmap */
    indsmap = malloc(sizeof(*indsmap) * dimSize);
    for (i = 0; i < dimSize; ++i) {
        indsmap[i] = IDX_T_MAX;
    }
    rindsmap = malloc(sizeof(*rindsmap) * myrowsCnt);
    for (i = 0; i < myrowsCnt; ++i) {
        assert(ss < dimSize && "ERROR: starting offset should be less than dimSize");
        indsmap[ss + i] = i;
    }
#ifdef NA_DBG
    MPI_Barrier(MPI_COMM_WORLD);
    na_log(dbgfp, "\tindsmaps are ready\n");
#endif

    /* count how many processors update each row */
    rows_p_cnt = malloc(sizeof(*rows_p_cnt) * myrowsCnt);
    setintzero(rows_p_cnt, myrowsCnt);
    for (i = 0; i < npes; ++i) {
        if (i != mype) {
            for (j = 0; j < rCnts_all[i]; ++j) {
                assert(rrFROM[i][j] < dimSize);
#ifdef NA_DBG
                if(indsmap[rrFROM[i][j]] >= myrowsCnt)
                    na_log(dbgfp, "\tindsmaps[rrFROM[%zu][%zu]] = %zu while myrowsCnt = %zu, growID= %zu my ss=%zu , se=%zu\n", i, j, indsmap[rrFROM[i][j]], myrowsCnt, rrFROM[i][j], ss, se);
#endif
                assert(indsmap[rrFROM[i][j]] < myrowsCnt);
                rows_p_cnt[indsmap[rrFROM[i][j]]]++;
            }
        }
        else{
            for (j = 0; j < rCnts[mype]; ++j) {
                rows_p_cnt[indsmap[rrTO[i][j]]]++;
            }
        }
    }
#ifdef NA_DBG
    MPI_Barrier(MPI_COMM_WORLD);
    na_log(dbgfp, "\trows_p_cnt is ready\n");
#endif

    /* allocate a seperate list for each row */
    rows_p = malloc(sizeof(*rows_p) * myrowsCnt);

    for (i = 0; i < myrowsCnt; ++i) {
        if (rows_p_cnt[i] != 0) {
            rows_p[i] = malloc(sizeof(*rows_p[i]) * rows_p_cnt[i]); 
        }
    }

    /* populate the list of each row  */
    setintzero(tcnts, tcntsSize);
    idx_t tv;
    for (i = 0; i < npes; ++i) {
        if (i != mype) {
            for (j = 0; j < rCnts_all[i]; ++j) {
                tv = indsmap[rrFROM[i][j]];
                assert(tv < myrowsCnt);
                rows_p[tv][tcnts[tv]++] = i;
            }
        }
        else{
            for (j = 0; j < rCnts[mype]; ++j) {
                tv = indsmap[rrTO[i][j]];
                assert(tv < myrowsCnt);
                rows_p[tv][tcnts[tv]++] = i;
            }
        }
    }
#ifdef NA_DBG
    MPI_Barrier(MPI_COMM_WORLD);
    na_log(dbgfp, "\trows_p are alloated and populated\n");
/*     for (i = 0; i < myrowsCnt; ++i) {
 *         na_log(dbgfp, "\nrows_p[%zu] => ", i);
 *         for (j = 0; j < rows_p_cnt[i]; ++j) {
 *             na_log(dbgfp, "p%zu ", rows_p[i][j]);
 *         }
 * 
 *     }
 */
#endif

    /* now the actual assignment */

    /* first, assign all rows that belong to only one part */
    for (i = 0; i < myrowsCnt; ++i) {
        if(rows_p_cnt[i] == 1)
            linterpart[i] = rows_p[i][0];
    }
#ifdef NA_DBG
    MPI_Barrier(MPI_COMM_WORLD);
    na_log(dbgfp, "\tassign all rows that belong to only one part DONE\n");
#endif

    /* randomly pick a processor from the list and assign row i to it */ 
    for (i = 0; i < myrowsCnt; ++i) {
        if (rows_p_cnt[i] > 1) {
            srand(time(NULL));
            idx_t rn = rand() % rows_p_cnt[i];    
            assert(rn < npes);
            linterpart[i] = rows_p[i][rn];
        }
    }
#ifdef NA_DBG
    MPI_Barrier(MPI_COMM_WORLD);
    na_log(dbgfp, "\tassignments are done\n");
#endif

    /* now all_gatherv */

    int *cnts, *disps;
    cnts = malloc(sizeof(*cnts) * npes);
    disps = malloc(sizeof(*disps) * npes);
    for (i = 0; i < npes; ++i) {
        disps[i] = (int) (i < rows_ppR ? i  * (rows_pp+1) : (rows_ppR * (rows_pp+1))+((i-rows_ppR )* rows_pp)) ;
        cnts[i] = (int) (i < rows_ppR ? rows_pp+1 : rows_pp);
    }
    MPI_Allgatherv(linterpart, myrowsCnt, MPI_IDX_T, interpart, cnts, disps, MPI_IDX_T, MPI_COMM_WORLD);
#ifdef NA_DBG
    MPI_Barrier(MPI_COMM_WORLD);
    na_log(dbgfp, "\tAllgatherv is done\n");
    for (i = 0; i < dimSize; ++i) {
        assert(interpart[i] < npes && "invalid interpart assignment");
    }
#endif
    for (i = 0; i < myrowsCnt; ++i) {
        assert(linterpart[i] == interpart[ss+i]);
    }

    /* cleanup */
    for (i = 0; i < npes; ++i) {
        if (rCnts[i] > 0) 
            free(rrTO[i]);

        if(rCnts_all[i] > 0 && i != mype)
            free(rrFROM[i]);
    }
    for (i = 0; i < myrowsCnt; ++i) {
        if (rows_p_cnt[i] > 0) {
            free(rows_p[i]);    
        }
    }
    free(cnts); free(disps); free(rCnts); free(rCnts_all); free(rows_p_cnt); free(tcnts); free(indsmap); free(rindsmap); free(linterpart);
    free(rows_p);
    free(rrTO); free(rrFROM);
}
