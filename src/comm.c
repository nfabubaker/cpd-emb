#include "comm.h"
#include "stat.h"
#include "tensor.h"
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

void init_comm(struct genst *gs) {

    idx_t nmodes = gs->nmodes;
    gs->comm = (struct comm *)malloc(sizeof(struct comm));
    struct comm *co = gs->comm;

    co->nrecvwho = (idx_t *)malloc(sizeof(*co->nrecvwho) * nmodes);
    co->recvwho = (idx_t **)malloc(sizeof(*co->recvwho) * nmodes);
    co->xrecvind = (idx_t **)malloc(sizeof(*co->xrecvind) * nmodes);
    co->recvind = (idx_t **)malloc(sizeof(*co->recvind) * nmodes);

    co->nsendwho = (idx_t *)malloc(sizeof(*co->nsendwho) * nmodes);
    co->sendwho = (idx_t **)malloc(sizeof(*co->sendwho) * nmodes);
    co->xsendind = (idx_t **)malloc(sizeof(*co->xsendind) * nmodes);
    co->sendind = (idx_t **)malloc(sizeof(*co->sendind) * nmodes);

    co->buffer = (real_t *)malloc(sizeof(*co->buffer) * nmodes);

    setintzero(co->nrecvwho, nmodes);
    setintzero(co->nsendwho, nmodes);
}


void setup_fg_communication(struct genst *gs, struct tensor *t,
        struct stats *st) {
    idx_t i, j, k, gdim, count, *mark, ind, *inds, ptr, lnnz, p,
    *map, myrows, *cnts,maxbufsize ;
    idx_t nmodes, mype, size, *interpart  ;
    struct comm *co;

    nmodes = gs->nmodes;
    mype = gs->mype;
    size = gs->npes;

    //init_comm(gs);
    co = gs->comm;
/*           {
 *               volatile idx_t tt = 0;
 *               printf("PID %d on %d ready for attach\n", (int)mype,  getpid());
 *               fflush(stdout);
 *               while (0 == tt)
 *                   sleep(5);
 *           }
 */



    gs->indmap = malloc(nmodes * sizeof(*gs->indmap));
    gs->ldims = malloc(nmodes * sizeof(*gs->ldims));

    maxbufsize = 0;
    for (i = 0; i < nmodes; i++) {
        gdim = gs->gdims[i];
        interpart = gs->interpart[i];

        // then indicate the number of rows to be received for mode i
        mark = (idx_t *)malloc(sizeof(*mark) * gdim);
        setintzero(mark, gdim);

        co->xrecvind[i] = (idx_t *)malloc(sizeof(*co->xrecvind[i]) * (size + 2));
        setintzero(co->xrecvind[i], size + 2);

        inds = t->inds;
        lnnz = t->nnz;
        ptr = i;
        for (k = 0; k < lnnz; k++) {
            ind = inds[ptr];
            p = interpart[ind];
            if (p != mype)
                if (mark[ind] == 0) {
                    co->xrecvind[i][p + 2]++;
                    mark[ind] = 1;
                }
            ptr += nmodes;
        }

        // count local rows
        cnts = (idx_t *)malloc((size + 2) * sizeof(*cnts));
        setintzero(cnts, size + 2);

        for (j = 0; j < gdim; j++) {
            p = interpart[j];
            if (p != mype) {
                if (mark[j] == 1)
                    cnts[p + 2]++;
            } else
                cnts[p + 2]++;
        }

        myrows = cnts[mype + 2];

        // prefix sum
        for (j = 2; j <= size + 1; j++)
            cnts[j] += cnts[j - 1];

        st->row[i] = myrows;

        // relabel
        gs->indmap[i] = (idx_t *)malloc(gdim * sizeof(*gs->indmap[i]));
        map = gs->indmap[i];
        for (j = 0; j < gdim; j++) {
            p = interpart[j];
            if (p != mype) {
                if (mark[j] == 1)
                    map[j] = cnts[p + 1]++;
            } else
                map[j] = cnts[p + 1]++;

            mark[j] = 0;
        }
        gs->ldims[i] = cnts[size + 1];

        // go back to receive stuff
        for (j = 2; j <= size + 1; j++)
            co->xrecvind[i][j] += co->xrecvind[i][j - 1];

        co->recvind[i] = (idx_t *)malloc(sizeof(*co->recvind[i]) * co->xrecvind[i][size + 1]);

        ptr = i;
        for (k = 0; k < lnnz; k++) {
            ind = inds[ptr];
            p = interpart[ind];
            if (p != mype)
                if (mark[ind] == 0) {
                    co->recvind[i][co->xrecvind[i][p + 1]++] = ind;
                    mark[ind] = 1;
                }
            ptr += nmodes;
        }
        free(mark);

        co->xsendind[i] = (idx_t *)malloc(sizeof(*co->xsendind[i]) * (size + 2));
        setintzero(co->xsendind[i], size + 2);

        MPI_Request req[size];
        for (j = 0; j < size; j++)
            if (j != mype)
                MPI_Irecv(&(co->xsendind[i][j + 1]), 1, MPI_IDX_T, (int) j, 1, MPI_COMM_WORLD,
                        j < mype ? &req[j] : &req[j - 1]);

        for (j = 0; j < size; j++) {
            if (j != mype) {
                idx_t nrecv = co->xrecvind[i][j + 1] - co->xrecvind[i][j];
                MPI_Send(&nrecv, 1, MPI_IDX_T, (int) j, 1, MPI_COMM_WORLD);
            }
        }

        MPI_Status sta[size];
        MPI_Waitall(size - 1, req, sta);

        for (j = 1; j < size + 1; j++)
            co->xsendind[i][j] += co->xsendind[i][j - 1];

        co->sendind[i] = (idx_t *)malloc(sizeof(*co->sendind[i]) * co->xsendind[i][size]);

        co->recvwho[i] = (idx_t *)malloc(sizeof(*co->recvind[i]) * size);
        setintzero(co->recvwho[i], size);

        for (j = 0; j < size; j++)
            if (co->xrecvind[i][j + 1] - co->xrecvind[i][j] > 0)
                co->recvwho[i][co->nrecvwho[i]++] = j;

        co->sendwho[i] = (idx_t *)malloc(sizeof(*co->sendwho[i]) * size);
        setintzero(co->sendwho[i], size);

        for (j = 0; j < size; j++)
            if (co->xsendind[i][j + 1] - co->xsendind[i][j] > 0)
                co->sendwho[i][co->nsendwho[i]++] = j;

        for (j = 0; j < co->nsendwho[i]; j++) {
            idx_t p = co->sendwho[i][j];
            MPI_Irecv(&(co->sendind[i][co->xsendind[i][p]]),
                    co->xsendind[i][p + 1] - co->xsendind[i][p], MPI_IDX_T, p, 2,
                    MPI_COMM_WORLD, &req[j]);
        }

        for (j = 0; j < co->nrecvwho[i]; j++) {
            idx_t p = co->recvwho[i][j];
            MPI_Send(&(co->recvind[i][co->xrecvind[i][p]]),
                    co->xrecvind[i][p + 1] - co->xrecvind[i][p], MPI_IDX_T, p, 2,
                    MPI_COMM_WORLD);
        }
        MPI_Waitall(co->nsendwho[i], req, sta);

        st->recvvol[i] = co->xrecvind[i][size];
        st->sendvol[i] = co->xsendind[i][size];
        st->recvmsg[i] = co->nrecvwho[i];
        st->sendmsg[i] = co->nsendwho[i];

        free(cnts);
    }
    st->nnz = t->nnz;
}
void setup_fg_communication_post(genst *gs,struct tensor *t, comm *co, stats *st){
    idx_t i,j, lnnz, ptr, myrows, maxbufsize = 0; 
    idx_t mype, size;
    mype = gs->mype;
    size = gs->npes;
    lnnz = t->nnz;
    for (i = 0; i < gs->nmodes; ++i) {
        myrows = st->row[i];
        // we dont need recvind indices anymore, do we?
        for (j = mype; j < size; j++)
            co->xrecvind[i][j + 1] += myrows;

        // localize indices in tensor
        ptr = i;
        for (j = 0; j < lnnz; j++) {
            t->inds[ptr] = gs->indmap[i][t->inds[ptr]];
            ptr += gs->nmodes;
        }

        // localize indices in sendind
        for (j = 0; j < co->xsendind[i][size]; j++)
            co->sendind[i][j] = gs->indmap[i][co->sendind[i][j]];


        if (co->xrecvind[i][size] > maxbufsize)
            maxbufsize = co->xrecvind[i][size];
        if (co->xsendind[i][size] > maxbufsize)
            maxbufsize = co->xsendind[i][size];
    }

    co->buffer = (real_t *)malloc(sizeof(*co->buffer) * maxbufsize * gs->cprank);
}

void init_emb_things(genst *gs){
    comm *co = gs->comm; 
    idx_t i,j, ndims;
    ndims = (idx_t) log2((double)gs->npes);
    co->ec = malloc(sizeof(ecomm *) * gs->nmodes * 2);
    
    /* for each mode, setup two hypercubes*/
    for (i = 0; i < gs->nmodes; ++i) {
        /* expand with lambda */
        co->ec[i*2+1] = ecomm_setup(co->nsendwho[i], co->sendwho[i], co->xsendind[i], co->sendind[i], co->nrecvwho[i], co->recvwho[i], co->xrecvind[i], co->recvind[i], ndims, gs->mat[i], gs->indmap[i], gs->cprank , gs->cprank, co->hypercube_imap[i], 0);
        /* fold with cTc */
        co->ec[i*2] = ecomm_setup( co->nrecvwho[i], co->recvwho[i], co->xrecvind[i], co->recvind[i],co->nsendwho[i], co->sendwho[i], co->xsendind[i], co->sendind[i], ndims, gs->matm, gs->indmap[i], gs->cprank , gs->cprank * gs->cprank, co->hypercube_imap[i], 1);
    }
}

void read_imap_and_reassign_partvec(genst  *gs)
{
    idx_t i, j;
    comm *co;
    idx_t *HM, *HI;
/*    HM = malloc(sizeof(*HM) * gs->npes);
 */

    co = gs->comm;
    co->hypercube_imap = malloc(sizeof(*co->hypercube_imap) * gs->nmodes);
    for (i = 0; i < gs->nmodes; ++i) {
        co->hypercube_imap[i] = malloc(sizeof(*co->hypercube_imap[i]) * gs->npes);
    }
    if (gs->use_hc_imap) {
        read_hc_imap(gs->hc_imap_FN, gs->nmodes, gs->npes, co->hypercube_imap);
        /* re-assign partvec values */
/*         for (i = 0; i < gs->nmodes; ++i) {
 *             for (j = 0; j < gs->npes; ++j) {
 *                 HM[co->hypercube_imap[i][j]] = j;
 *             }
 *             HI = co->hypercube_imap[i];
 *             for (j = 0; j < gs->gdims[i]; ++j) {
 *                 gs->interpart[i][j] = HM[gs->interpart[i][j]];
 *             }
 *         }
 */

    }
    else {
        for (i = 0; i < gs->nmodes; ++i) {
            for (j = 0; j < gs->npes; ++j) {
               co->hypercube_imap[i][j] = j; 
            }
        }
    }
/*     free(HM);
 */
}

void setup_comm(genst *gs, tensor *t, stats *st){

#ifdef NA_DBG
    na_log(dbgfp, "\thello from setup_comm\n");
#endif
    idx_t maxldim, i;
    init_comm(gs);
#ifdef NA_DBG
    na_log(dbgfp, "\tafter init_comm\n");
#endif

    if(gs->comm_type == EMB)
        read_imap_and_reassign_partvec(gs);
#ifdef NA_DBG
    MPI_Barrier(MPI_COMM_WORLD);
    na_log(dbgfp, "\tafter real_imap\n");
#endif

    setup_fg_communication(gs, t, st);
#ifdef NA_DBG
    MPI_Barrier(MPI_COMM_WORLD);
    na_log(dbgfp, "\tafter setup_fg_communication\n");
#endif

    init_matrices(gs);
#ifdef NA_DBG
    MPI_Barrier(MPI_COMM_WORLD);
    na_log(dbgfp, "\tafter init_matrices\n");
#endif
    
    maxldim = gs->ldims[0];
    for(i = 1; i < gs->nmodes; i++)
        if(gs->ldims[i] > maxldim)
            maxldim = gs->ldims[i];

    gs->matm = (real_t *)malloc(maxldim*gs->cprank*sizeof(*gs->matm));
#ifdef NA_DBG
    MPI_Barrier(MPI_COMM_WORLD);
    na_log(dbgfp, "\tafter matm allocation\n");
#endif

    if(gs->comm_type == EMB){
        init_emb_things(gs);
#ifdef NA_DBG
    MPI_Barrier(MPI_COMM_WORLD);
    na_log(dbgfp, "\tafter init_emb_things\n");
#endif
    }
    setup_fg_communication_post(gs, t , gs->comm, st);
#ifdef NA_DBG
    MPI_Barrier(MPI_COMM_WORLD);
    na_log(dbgfp, "\tafter setup_fg_communication_post\n");
#endif
    if(gs->comm_type == EMB)
        free_comm(gs->comm, gs->nmodes);
}


void receive_partial_products_fg(struct genst *t, idx_t mode, real_t *matm) {

#ifdef NA_DBG
    na_log(dbgfp, "dbg p4.0.1 hello from recv partial products\n");
#endif
    idx_t i, j, iwrite, iread, nrecvwho, nsendwho, *recvwho, *sendwho, *xrecvind,
        *xsendind, *recvind, cprank, size, *sendcnt, *recvcnt, *senddisp,
        *recvdisp;
    int who, srsize;
    real_t *recvbuf;
    struct comm *co;

    size = t->npes;
    cprank = t->cprank;
    co = t->comm;

    nrecvwho = co->nsendwho[mode];
    recvwho = co->sendwho[mode];
    xrecvind = co->xsendind[mode];

    nsendwho = co->nrecvwho[mode];
    sendwho = co->recvwho[mode];
    xsendind = co->xrecvind[mode];

    recvind = co->sendind[mode];
    recvbuf = co->buffer;

        if (t->alltoall) {
            sendcnt = (idx_t *)malloc(size * sizeof(idx_t));
            setintzero(sendcnt, size);
            recvcnt = (idx_t *)malloc(size * sizeof(idx_t));
            setintzero(recvcnt, size);
            senddisp = (idx_t *)malloc(size * sizeof(idx_t));
            setintzero(senddisp, size);
            recvdisp = (idx_t *)malloc(size * sizeof(idx_t));
            setintzero(recvdisp, size);

            for (i = 0; i < nsendwho; i++) {
                who = sendwho[i];
                sendcnt[who] = (xsendind[who + 1] - xsendind[who]) * cprank;
                senddisp[who] = xsendind[who] * cprank;
            }

            for (i = 0; i < nrecvwho; i++) {
                who = recvwho[i];
                recvcnt[who] = (xrecvind[who + 1] - xrecvind[who]) * cprank;
                recvdisp[who] = xrecvind[who] * cprank;
            }

            MPI_Alltoallv(matm, sendcnt, senddisp, MPI_REAL_T, recvbuf, recvcnt,
                    recvdisp, MPI_REAL_T, MPI_COMM_WORLD);

            free(sendcnt);
            free(recvcnt);
            free(senddisp);
            free(recvdisp);

        } else {
            MPI_Request req[nrecvwho];

            for (i = 0; i < nrecvwho; i++) {
                who = (int) recvwho[i];
                srsize = (int) ((xrecvind[who + 1] - xrecvind[who]) * cprank);
                MPI_Irecv(&recvbuf[xrecvind[who] * cprank],
                        srsize, MPI_REAL_T, who,
                        3, MPI_COMM_WORLD, &req[i]);
            }
            for (i = 0; i < nsendwho; i++) {
                who = (int) sendwho[i];
                srsize = (int) ((xsendind[who + 1] - xsendind[who]) * cprank);
                MPI_Send(&matm[xsendind[who] * cprank],
                        srsize, MPI_REAL_T, who,
                        3, MPI_COMM_WORLD);
            }

            MPI_Status sta[nrecvwho];
            MPI_Waitall(nrecvwho, req, sta);
        }
    iread = 0;
    for (i = 0; i < xrecvind[size]; i++) {
        iwrite = recvind[i] * cprank;
        for (j = 0; j < cprank; j++)
            matm[iwrite++] += recvbuf[iread++];
    }
}

void send_updated_rows_fg(struct genst *gs, idx_t mode) {

    idx_t i, j, ptr, nrecvwho, nsendwho, size, cprank, *recvwho, *sendwho,
        *xsendind, *xrecvind, *sendind, *recvind, *map, start, end, *sendcnt,
        *recvcnt, *senddisp, *recvdisp;
    int who, srsize;
    real_t *sendbuf, *mat;
    struct comm *co;

    size = gs->npes;
    co = gs->comm;
    cprank = gs->cprank;

    mat = gs->mat[mode];

    nrecvwho = co->nrecvwho[mode];
    nsendwho = co->nsendwho[mode];
    recvwho = co->recvwho[mode];
    sendwho = co->sendwho[mode];

    xrecvind = co->xrecvind[mode];
    xsendind = co->xsendind[mode];

    recvind = co->recvind[mode];
    sendind = co->sendind[mode];

    // send computed rows
    sendbuf = co->buffer;
    ptr = 0;
    for (i = 0; i < nsendwho; i++) {
        who = (int) sendwho[i];
        start = xsendind[who];
        end = xsendind[who + 1];
        for (j = start; j < end; j++) {
            memcpy(&sendbuf[ptr], &mat[sendind[j]], sizeof(real_t) * cprank);
            ptr += cprank;
        }
    }

        if (gs->alltoall) {
            sendcnt = (idx_t *)malloc(size * sizeof(idx_t));
            setintzero(sendcnt, size);
            recvcnt = (idx_t *)malloc(size * sizeof(idx_t));
            setintzero(recvcnt, size);
            senddisp = (idx_t *)malloc(size * sizeof(idx_t));
            setintzero(senddisp, size);
            recvdisp = (idx_t *)malloc(size * sizeof(idx_t));
            setintzero(recvdisp, size);

            for (i = 0; i < nrecvwho; i++) {
                who = (int) recvwho[i];
                recvcnt[who] = (xrecvind[who + 1] - xrecvind[who]) * cprank;
                recvdisp[who] = xrecvind[who] * cprank;
            }

            for (i = 0; i < nsendwho; i++) {
                who = sendwho[i];
                sendcnt[who] = (xsendind[who + 1] - xsendind[who]) * cprank;
                senddisp[who] = xsendind[who] * cprank;
            }

            MPI_Alltoallv(sendbuf, sendcnt, senddisp, MPI_REAL_T, mat, recvcnt,
                    recvdisp, MPI_REAL_T, MPI_COMM_WORLD);

            free(sendcnt);
            free(recvcnt);
            free(senddisp);
            free(recvdisp);

        } else {
            // issue Irecvs
            MPI_Request req[nrecvwho];

            for (i = 0; i < nrecvwho; i++) {
                who = (int) recvwho[i];
                srsize = (int) ((xrecvind[who + 1] - xrecvind[who]) * cprank);
                MPI_Irecv(&mat[xrecvind[who] * cprank],
                        srsize, MPI_REAL_T, who,
                        4, MPI_COMM_WORLD, &req[i]);
            }

            // send computed rows
            for (i = 0; i < nsendwho; i++) {
                who = (int) sendwho[i];
                srsize = (int) ((xsendind[who + 1] - xsendind[who]) * cprank);
                MPI_Send(&sendbuf[xsendind[who] * cprank],
                        srsize, MPI_REAL_T, who,
                        4, MPI_COMM_WORLD);
            }

            MPI_Status sta[nrecvwho];
            MPI_Waitall(nrecvwho, req, sta);
        }
}


void free_comm(struct comm *c, idx_t nmodes) {
    idx_t i;

    if (c->nrecvwho != NULL)
        free(c->nrecvwho);

    if (c->recvwho != NULL) {
        for (i = 0; i < nmodes; i++)
            free(c->recvwho[i]);
        free(c->recvwho);
    }

    if (c->xrecvind != NULL) {
        for (i = 0; i < nmodes; i++)
            free(c->xrecvind[i]);
        free(c->xrecvind);
    }

    if (c->recvind != NULL) {
        for (i = 0; i < nmodes; i++)
            free(c->recvind[i]);
        free(c->recvind);
    }

    if (c->nsendwho != NULL)
        free(c->nsendwho);

    if (c->sendwho != NULL) {
        for (i = 0; i < nmodes; i++)
            free(c->sendwho[i]);
        free(c->sendwho);
    }

    if (c->xsendind != NULL) {
        for (i = 0; i < nmodes; i++)
            free(c->xsendind[i]);
        free(c->xsendind);
    }

    if (c->sendind != NULL) {
        for (i = 0; i < nmodes; i++)
            free(c->sendind[i]);
        free(c->sendind);
    }

    if (c->buffer != NULL)
        free(c->buffer);

    //free(c);
}
