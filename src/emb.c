
/* Use P2P comm structures to setup Hypercube for All-reduce*/
#include <mpi.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "ecomm.h"
#include "util.h"
#include <math.h>

void emb_get_stats(ecomm *ec, idx_t *maxSendVol, idx_t *maxRecvVol, idx_t *totalVol , idx_t *maxSendMsgs, idx_t *maxRecvMsgs, idx_t *totalMsgs)
{
    idx_t i;

    /* these include header */
    idx_t *vstot = malloc(sizeof(*vstot) * ec->ndims);
    idx_t *vrtot = malloc(sizeof(*vrtot) * ec->ndims);
    idx_t *vsmax = malloc(sizeof(*vsmax) * ec->ndims);
    idx_t *vrmax = malloc(sizeof(*vrmax) * ec->ndims);
    idx_t *mstot = malloc(sizeof(*mstot) * ec->ndims);
    idx_t *mrtot = malloc(sizeof(*mrtot) * ec->ndims);
    idx_t *msmax = malloc(sizeof(*msmax) * ec->ndims);
    idx_t *mrmax = malloc(sizeof(*mrmax) * ec->ndims);

    idx_t vsend, vrecv, msend, mrecv, vboth;
    for (i = 0; i < ec->ndims; ++i) {
        vsend = ec->xsendptrs[i][CT_CNT] -1; 
        vrecv = ec->xrecvptrs[i][CT_CNT] -1;
        msend = 1;
        mrecv = 1;
        vboth = vsend + vrecv;

        MPI_Reduce(&vsend, &vstot[i], 1, MPI_IDX_T, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&vboth, &vsmax[i], 1, MPI_IDX_T, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&vrecv, &vrtot[i], 1, MPI_IDX_T, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&vboth, &vrmax[i], 1, MPI_IDX_T, MPI_MAX, 0, MPI_COMM_WORLD);

        MPI_Reduce(&msend, &mstot[i], 1, MPI_IDX_T, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&msend, &msmax[i], 1, MPI_IDX_T, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&mrecv, &mrtot[i], 1, MPI_IDX_T, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&mrecv, &mrmax[i], 1, MPI_IDX_T, MPI_MAX, 0, MPI_COMM_WORLD);

    }

    *totalVol = 0;  *maxSendVol= 0; *maxRecvVol = 0;
    *totalMsgs = 0; *maxSendMsgs = 0; *maxRecvMsgs= 0;

    for (i = 0; i < ec->ndims; ++i) {
        *totalVol += vstot[i] + vrtot[i];
        *maxSendVol += vsmax[i];
        *maxRecvVol += vrmax[i];
        *totalMsgs += mstot[i]+ mrtot[i];
        *maxSendMsgs += msmax[i];
        *maxRecvMsgs += mrmax[i];
    }

    idx_t use_maxofsum = 0;



    if (use_maxofsum) {
        vsend = vrecv = msend = mrecv = 0;
        for (i = 0; i < ec->ndims; ++i) {
            vsend += ec->xsendptrs[i][CT_CNT] -1; 
            vrecv += ec->xrecvptrs[i][CT_CNT] -1;
            msend += 1;
            mrecv += 1;
        }   
        idx_t vsend_max, vrecv_max, msend_max, mrecv_max;
        MPI_Reduce(&vsend, &vsend_max, 1, MPI_IDX_T, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&vrecv, &vrecv_max, 1, MPI_IDX_T, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&msend, &msend_max, 1, MPI_IDX_T, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&mrecv, &mrecv_max, 1, MPI_IDX_T, MPI_MAX, 0, MPI_COMM_WORLD);

        *maxSendVol = vsend_max;
        *maxRecvVol = vrecv_max; 
        *maxSendMsgs = msend_max;
        *maxRecvMsgs = mrecv_max;
    }

    free(vstot);
    free(vrtot);
    free(vsmax);
    free(vrmax);
    free(mstot);
    free(mrtot);
    free(msmax);
    free(mrmax);

}
void get_neighbors(idx_t mypid, idx_t *sendto, idx_t ndims, idx_t dir){
    idx_t i;

    for (i = 0; i < ndims; ++i) {
        sendto[i] = mypid ^ (1 << i);
    }

}

idx_t get_comm_dim(idx_t mypid, idx_t dst, idx_t currDim, idx_t ndims, idx_t dir)
{
    idx_t pos=currDim+1;
    idx_t mask = (mypid ^ dst);

#ifdef NA_DBG_L3
    na_log(dbgfp, "\t\t in get_comm_dim, mypid=%d dst=%d mask = %d pos=%d\n", mypid, dst, mask, pos);
#endif

    if (dir == 0) {
        mask >>= currDim+1;
        while(!(mask & 1)){
            pos++;
            mask >>= 1;
        }
        /* 	pos++;
         * 	mask <<= 1;
         * 	while(!(mask & 1)){
         * 		pos++;
         * 		mask <<= 1;
         * 	}
         */
    }
    else{
        idx_t tcnt = 0;
        //fprintf(stderr, "hello I'm stuck %d\n", tcnt++);
        while(!(mask & (1 <<  pos))){
            pos--;
        }
        /* 	pos++;
         * 	mask <<= 1;
         * 	while(!(mask & 1)){
         * 		pos++;
         * 		mask <<= 1;
         * 	}
         */
    }

    return pos;
}

void * remove_from_list(msg *head){
    msg tmsg = *head;
    *head = (*head)->next;
    return tmsg;
}

msg add_to_list(msg head, idx_t src, idx_t dst, idx_t size, idx_t recv_id, idx_t recv_dim, idx_t send_id, idx_t send_dim, idx_t *indsptr){
    msg tm, p;
    tm = malloc(sizeof(*tm));
    tm->src = src; tm->dst = dst; tm->size = size;
    tm->recv_dim = recv_dim; tm->send_dim = send_dim;
    tm->recv_id = recv_id; tm->send_id = send_id;
    tm->inds = malloc(sizeof(*tm->inds) * size);
    memcpy(tm->inds, indsptr, sizeof(*indsptr) * size);
    tm->next = NULL;

    tm->next = head;
    head = tm;
    /*     if(head == NULL){
     *         head = tm;
     *     }
     *     else{
     *         p = head;
     *         while(NULL != p->next)
     *             p = p->next;
     *         p->next = tm;
     *     }
     */
    return head;

}

ecomm *ecomm_setup(idx_t nsendwho, idx_t *sendwho, idx_t *xsendind, idx_t *sendind, idx_t nrecvwho, idx_t *recvwho, idx_t *xrecvind, idx_t *recvind, idx_t ndims, real_t *data, idx_t *indsmap,  idx_t embDataUnitSize, idx_t origDataUnitSize, idx_t *hypercube_imap, idx_t dir)
{
    /* HI: hypercube rank to MPI rank inverse map
     * HM: MPI rank to hypercube map
     * SH: sendwho
     * SHV: sendwho virtual */
    idx_t i,j, currN, mask, mypid, npes, f, maxSendSize = 0, maxRecvSize = 0, SH, SHV;
    idx_t sendSize[3] = {0}, recvSize[3] = {0}, nmsgs;
    ecomm *ec = malloc(sizeof(*ec));
    idx_t *recvBuff, *sendBuff, *stbuff_sizes, **st_recv_offsets, **st_send_offsets, *HI, *HM;
    msg *head, *sndList;
    char * sendtag;
    msg tmsg;

    /* sendtag: just to make sure each msg is sent once
     * stbuff_sizes: ??
     * st_recv_offsets: offsets per msg in the recv buffer 
     * st_send_offsets: same
     * sndList: msgs that are sent by me to be forwarded
     * head: msgs that are recvd and to be forwarded */
    sendtag = calloc(nsendwho, sizeof(*sendtag));
    stbuff_sizes = calloc(ndims, sizeof(*stbuff_sizes));
    st_recv_offsets = malloc(sizeof(*st_recv_offsets) * ndims);
    st_send_offsets = malloc(sizeof(*st_send_offsets) * ndims);
    head = malloc(sizeof(*head) * (ndims ));
    sndList = malloc(sizeof(*sndList) * (ndims ));
    for (i = 0; i < ndims; ++i) {
        head[i] = NULL;
        sndList[i] = NULL;
    }

    f = embDataUnitSize;
#ifdef NA_DBG
    na_log(dbgfp, "%s\n", "\tin emb_setup" );
#endif
    int mypid2, npes2;
    MPI_Comm_rank(MPI_COMM_WORLD, &mypid2);
    MPI_Comm_size(MPI_COMM_WORLD, &npes2);
    mypid = (idx_t) mypid2;
    npes = (idx_t) npes2;
    ec->dir = dir;

    /* prepare HI and HM */
    ec->HI = malloc(sizeof(*ec->HI) * npes);
    for (i = 0; i < npes; ++i) {
        ec->HI[i] = hypercube_imap[i];
    }
    HI = ec->HI;
    HM = malloc(sizeof(*HM) * npes);
    for (i = 0; i < npes; ++i) {
        HM[HI[i]] = i; 
    }

    ec->ndims = ndims;
    ec->neighbor = malloc(sizeof(*ec->neighbor) * ndims);
    ec->xsendptrs = malloc(sizeof(*ec->xsendptrs) * ndims);
    ec->xrecvptrs = malloc(sizeof(*ec->xrecvptrs) * ndims);
    ec->sendptrs = malloc(sizeof(*ec->sendptrs) * ndims);
    ec->recvptrs = malloc(sizeof(*ec->recvptrs) * ndims);
    ec->store_buff = malloc(sizeof(*ec->store_buff) * ndims);

    /* get neighbor per HC dim according to my HC location */
    get_neighbors(HM[mypid] ,ec->neighbor, ndims, dir);
#ifdef NA_DBG
    na_log(dbgfp, "%s%d\n", "\t\tafter get_neighbors, ndims=",ndims);
    for (i = 0; i < ndims; ++i) {
        na_log(dbgfp, "\t\t\tmypid=%d - hcID = %d- neighbor[%d]=%d\n", mypid, HM[mypid], i, ec->neighbor[i]);
    }
#endif
    /* prepare msgs to be send by me. sendSize[0] contains msgs to be recvd by my 
     * neighbor while sendSize[1] contain msgs to be forwarded*/
    idx_t dd, dim, cond;
    for (dd = 0; dd < ndims; ++dd) {
        dim = (dir == 0 )? dd : ndims-1-dd;
        /* for each dim, send/recv the amount of comm*/
        currN = ec->neighbor[dim];
        mask = currN & (1 << dim);
        nmsgs = 0;
        sendSize[0] = 0; sendSize[1] = 0; sendSize[2] = 0;
        for (i = 0; i < nsendwho; ++i) {
            SHV = HM[sendwho[i]]; /* HC location of sendwho */
            SH = sendwho[i];        /* actual pid of sendwho */
            if(SHV == currN && !sendtag[i] ){
                sendSize[0] += (xsendind[SH+1] - xsendind[SH]);
            }
            else if (!sendtag[i] && !((SHV ^ mask) & (1 << dim))) {
                sendSize[1] += (xsendind[SH+1] - xsendind[SH]);
                nmsgs++;
            }
        }
#ifdef NA_DBG
        na_log(dbgfp, "%s\n", "\t  just after initia count 1" );
#endif

        if ( NULL != head[dim]) {
#ifdef NA_DBG
            if(dim==0)
                na_log(dbgfp, "%s\n", "\tI shouldn't be here!\n");
#endif
            tmsg = head[dim];
            i = 0;
            while(NULL != tmsg){
                if(tmsg->dst == currN || !((tmsg->dst ^ mask) & (1<<dim))) {
                    sendSize[1] += tmsg->size;
                    nmsgs++;
                }
#ifdef NA_DBG_L3
                na_log(dbgfp, "\thead[%d] msg: src=%d dst=%d size=%d snd_dim=%d rcv_dim=%d snd_id=%d rcv_id=%d add=%p next=%p\n",dim, tmsg->src, tmsg->dst, tmsg->size, tmsg->send_dim, tmsg->recv_dim, tmsg->send_id, tmsg->recv_id, tmsg, tmsg->next);
#endif
                tmsg = tmsg->next;
            }
        }
#ifdef NA_DBG
        na_log(dbgfp, "%s\n", "\t  just after initia count 2" );
#endif
        sendSize[2] = nmsgs;
        MPI_Send(sendSize, 3, MPI_IDX_T, HI[currN], 1, MPI_COMM_WORLD);
        MPI_Recv(recvSize, 3, MPI_IDX_T, HI[currN], 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        /*         MPI_Sendrecv(&sendSize, 3, MPI_IDX_T, currN, 1, &recvSize, 3, MPI_IDX_T, mypid, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        */
#ifdef NA_DBG
        MPI_Barrier(MPI_COMM_WORLD);
        na_log(dbgfp, "%s\n", "\tafter sendrecv"); 
#endif
        maxSendSize = (sendSize[0] + sendSize[1] > maxSendSize) ? sendSize[0]+sendSize[1] : maxSendSize;
        maxRecvSize = (recvSize[0] + recvSize[1] > maxRecvSize) ? recvSize[0]+recvSize[1] : maxRecvSize;

#ifdef NA_DBG
        MPI_Barrier(MPI_COMM_WORLD);
        na_log(dbgfp, "%s %d - recv=%d\n", "\tattempting to realloc, size = ", sendSize[0] + sendSize[1]  + nmsgs * 3, recvSize[0] + recvSize[1] + recvSize[2] * 3); 
#endif
        sendBuff = malloc(sizeof(*sendBuff) * (sendSize[0] + sendSize[1]  + nmsgs * 3));
        recvBuff = malloc(sizeof(*recvBuff) * (recvSize[0] + recvSize[1] + recvSize[2] * 3));

        st_send_offsets[dim] = calloc(sendSize[2]+1, sizeof(*st_send_offsets[dim]));
        st_recv_offsets[dim] = calloc(recvSize[2]+1, sizeof(*st_recv_offsets[dim]));


#ifdef NA_DBG
        MPI_Barrier(MPI_COMM_WORLD);
        na_log(dbgfp, "%s\n", "\tbuffers allocated, now communicate entries"); 
        na_log(dbgfp,"\tI'm %d and I will recv %d from %d and will send %d\n", mypid, recvSize[0]+recvSize[1]+recvSize[2]*3, currN, sendSize[0]+sendSize[1]+nmsgs*3); 
#endif
        MPI_Request req;
        MPI_Irecv(recvBuff, recvSize[0]+recvSize[1] + recvSize[2]*3, MPI_IDX_T, HI[currN], 2, MPI_COMM_WORLD, &req );
        idx_t *tptr = sendBuff;
        /* fitst, copy the data intended for neighbor */
        for (i = 0; i < nsendwho; ++i) {
            SH = sendwho[i];
            SHV = HM[sendwho[i]];
            if( SHV == currN && !sendtag[i]){
                //memcpy(tptr, &data[indsmap[sendind[j]]], f * sizeof(real_t));	
                memcpy(tptr, &sendind[xsendind[SH]], sizeof(int) * (xsendind[SH+1] - xsendind[SH]));
                tptr += (xsendind[SH+1] - xsendind[SH]) ;
                sendtag[i] = 1;
            }
        }
#ifdef NA_DBG
        MPI_Barrier(MPI_COMM_WORLD);
        na_log(dbgfp, "%s\n", "\trecv rqsts issued, data copied to send");
#endif

#ifdef NA_DBG
        MPI_Barrier(MPI_COMM_WORLD);
        na_log(dbgfp, "%s\n", "\trecv rqsts issued, data2 copied to send");
#endif
        /* then, copy the data to be stored. Add a header (src, dst, size) for each msg */
        idx_t tt = 1;
        for (i = 0; i < nsendwho; ++i) {
            SH = sendwho[i];
            SHV = HM[sendwho[i]];
            if ((SHV != currN ) && !sendtag[i] && !(( SHV^mask) & (1 << dim))){
                *(tptr++) = HM[mypid];
                *(tptr++) = SHV;
                *(tptr++) = (xsendind[SH+1] - xsendind[SH]);
                memcpy(tptr, &sendind[xsendind[SH]], sizeof(int) * (xsendind[SH+1] - xsendind[SH]));
                tptr += (xsendind[SH+1] - xsendind[SH]) ;
                st_send_offsets[dim][tt] = (xsendind[SH+1] - xsendind[SH]);
                sendtag[i] = 1;
#ifdef NA_DBG_L2
                na_log(dbgfp, "adding msg to sndList, src=%d dst=%d size=%d dim=%d", mypid, SH,(xsendind[SH+1] - xsendind[SH]) , dim);

#endif
                sndList[dim] = add_to_list(sndList[dim], HM[mypid], SHV,(xsendind[SH+1] - xsendind[SH]) , 0, 0, tt-1, dim, &sendind[xsendind[SH]]);
                tt++;
            }
        }
#ifdef NA_DBG
        MPI_Barrier(MPI_COMM_WORLD);
        na_log(dbgfp, "%s\n", "\trecv rqsts issued, data3 copied to send");
#endif
        if(head[dim] != NULL){
#ifdef NA_DBG
            if(dim==0)
                na_log(dbgfp, "%s\n", "\tI shouldn't be here!\n");
#endif
            msg tm = head[dim];
            while(tm != NULL){	
                if (tm->dst == currN || !((tm->dst ^ mask) & (1 << dim))){
                    *(tptr++) = tm->src;
                    *(tptr++) = tm->dst;
                    *(tptr++) = tm->size;
                    memcpy(tptr, tm->inds, sizeof(int) * tm->size);
                    tptr += tm->size ;
                    st_send_offsets[dim][tt] = tm->size;
                    tm->send_dim = dim; tm->send_id = tt-1;
                    tt++;
                }
                tm = tm->next;
            }
        }
#ifdef NA_DBG
        MPI_Barrier(MPI_COMM_WORLD);
        na_log(dbgfp, "%s\n", "\trecv rqsts issued, data4 copied to send");
#endif
        MPI_Send(sendBuff, sendSize[0] + sendSize[1] + nmsgs * 3, MPI_IDX_T, HI[currN], 2, MPI_COMM_WORLD);
        MPI_Wait(&req, MPI_STATUS_IGNORE);

#ifdef NA_DBG
        na_log(dbgfp, "%s\n", "\tsend done, now copy data");
#endif
        /* prefix sum for emb send offsets*/
        for (i = 1; i < tt; ++i) {
            st_send_offsets[dim][i] += st_send_offsets[dim][i-1];
        }
        /* Now setup the sendptrs */
        ec->xsendptrs[dim] = malloc(sizeof(*ec->xsendptrs[dim]) * (CT_CNT+1));
        ec->xsendptrs[dim][0] = 0;
        ec->xsendptrs[dim][1] = 1;
        ec->xsendptrs[dim][2] = sendSize[0];
        ec->xsendptrs[dim][3] = sendSize[1];
        for (i = 1; i <= CT_CNT; ++i) {
            ec->xsendptrs[dim][i]+= ec->xsendptrs[dim][i-1];	
        }
        ec->sendptrs[dim] = malloc(sizeof(*ec->sendptrs[dim]) * (1 + sendSize[0]+sendSize[1])); /*  the 1 here for the pointer to orig element to be reduced */
#ifdef NA_DBG
        MPI_Barrier(MPI_COMM_WORLD);
        na_log(dbgfp, "\tnow poidx_t sentptrs to data\n");
#ifdef NA_DBG_L3
        for (i = 0; i < sendSize[0]+sendSize[1]+ 3*sendSize[2]; ++i) {
            na_log(dbgfp, "\t\tpid %d sendbuf[%d] = %d\n", mypid, i, sendBuff[i]);
        }
#endif
#endif
        tptr = sendBuff;
        for (i = ec->xsendptrs[dim][1]; i < ec->xsendptrs[dim][2]; ++i) {
            ec->sendptrs[dim][i] = &data[indsmap[*(tptr)++] * embDataUnitSize];
        }
        /* now assign the indices to be recvd by me to recv */
        ec->xrecvptrs[dim] = malloc(sizeof(*ec->xrecvptrs[dim]) * (CT_CNT+1));
        ec->xrecvptrs[dim][0] = 0;
        ec->xrecvptrs[dim][1] = 1;
        ec->xrecvptrs[dim][2] = recvSize[0];
        ec->xrecvptrs[dim][3] = recvSize[1];
        for (i = 1; i <= CT_CNT; ++i) {
            ec->xrecvptrs[dim][i]+= ec->xrecvptrs[dim][i-1];	
        }
        ec->recvptrs[dim] = malloc(sizeof(*ec->recvptrs[dim]) * (1 + recvSize[0]+recvSize[1])); /*  the 1 here for the pointer to orig element to be reduced */
        tptr = recvBuff;
        for (i = ec->xrecvptrs[dim][1]; i < ec->xrecvptrs[dim][2]; ++i) {
            ec->recvptrs[dim][i] = &data[indsmap[*(tptr)++] * embDataUnitSize];
        }

#ifdef NA_DBG
        MPI_Barrier(MPI_COMM_WORLD);
        na_log(dbgfp, "%s\n", "\tdetermine what's to be forwarded");
#endif
        /* now process the elements to be forwarded and add them to their proper list*/
        tptr = recvBuff+recvSize[0];
        idx_t src, dst, size;
        tt=1;
        for (i = 0; i < recvSize[2]; i++) {
            src = *(tptr++);
            dst = *(tptr++);
            size = *(tptr++);
#ifdef NA_DBG_L3
            na_log(dbgfp, "\tmypid=%d packet: src=%d dst=%d size=%d, content: \n", mypid, src, dst, size);
            for (j = 0; j < size; ++j) {
                na_log(dbgfp, "\t%d ", (*(tptr+j)));
            }
#endif
            idx_t cdim; 
            if(dst == HM[mypid])
                cdim = dim;
            else{
                cdim = get_comm_dim(HM[mypid], dst, dim , ndims, dir);

#ifdef NA_DBG_L3
                na_log(dbgfp, "\tjust b4 add to list, mypid=%d src=%d dst=%d dim=%d cdim=%d\n", mypid, src, dst, dim, cdim);
#endif
                if (dir) {
                    assert(cdim < dim);
                }
                else
                    assert(cdim > dim);
                stbuff_sizes[cdim] += (size * embDataUnitSize);
            }
            /* #ifdef NA_DBG
             *                 na_log(dbgfp, "\tjust b4 add to list, tm=%p head[%d]=%p next=%p null=%p\n", tm, cdim, head[cdim], tm->next, NULL);
             * #endif
             */

            head[cdim] =  add_to_list(head[cdim], src, dst, size, i, dim, 0, cdim, tptr);
            tptr += size;
            /* #ifdef NA_DBG
             *                 na_log(dbgfp, "\tjust after add to list, tm=%p head[%d]=%p next=%p null=%p\n", tm, cdim, head[cdim], tm->next, NULL);
             * #endif
             */
            st_recv_offsets[dim][tt++] = (size);

        }
#ifdef NA_DBG
        MPI_Barrier(MPI_COMM_WORLD);
        na_log(dbgfp, "%s\n", "\tdone adding forward data to linked lists");
#endif
        for (i = 1; i < tt; ++i) {
            st_recv_offsets[dim][i] += st_recv_offsets[dim][i-1];
        }
#ifdef NA_DBG
        MPI_Barrier(MPI_COMM_WORLD);
        na_log(dbgfp, "\tdone with dim%d\n", dim);
#endif
        free(sendBuff); free(recvBuff);
    }

    /* now prepare the pointers to the store-and-forward data for dim+1*/
    for (i = 0; i < ndims; ++i) {
        if(stbuff_sizes[i] > 0){
#ifdef NA_DBG
            na_log(dbgfp, "\tstoreBuff[%d] size=%d actual size=%d\n", dim, stbuff_sizes[i]/embDataUnitSize, stbuff_sizes[i]);
#endif
            ec->store_buff[i] = malloc(sizeof(*ec->store_buff[i]) * stbuff_sizes[i]);
        }
        else
            ec->store_buff[i] = NULL;
#ifdef NA_DBG_L3
        tmsg = head[i];
        while(tmsg != NULL){
            na_log(dbgfp, "\thead[%d] msg: src=%d dst=%d size=%d snd_dim=%d rcv_dim=%d snd_id=%d rcv_id=%d\n",i, tmsg->src, tmsg->dst, tmsg->size, tmsg->send_dim, tmsg->recv_dim, tmsg->send_id, tmsg->recv_id);
            tmsg = tmsg->next;
        }
#endif
    }
#ifdef NA_DBG
    na_log(dbgfp, "%s\n", "\tnow link actual data");
#endif
    for (dd = 0; dd < ndims; ++dd) {
        dim = (dir == 0) ? dd : ndims-1-dd;
        /* first, add my msgs to be forwarded */
        idx_t rcnt = 0, scnt = 0, toffset = 0;
        if(sndList[dim] != NULL){
            tmsg = sndList[dim];
            while (tmsg != NULL) {
#ifdef NA_DBG_L2
                na_log(dbgfp, "\t[MS]linking msg dim=%d src=%d dst=%d size=%d\n",dim, tmsg->src, tmsg->dst, tmsg->size);
#ifdef NA_DBG_L3
                for (i = 0; i < tmsg->size; ++i) {

                    na_log(dbgfp, "\t%d\n",tmsg->inds[i]);
                }
#endif
#endif
                for (i = 0; i < tmsg->size; ++i) 
                    ec->sendptrs[ dim ][ ec->xsendptrs[dim][2] + st_send_offsets [ dim ][ tmsg->send_id] + i ] = &data[indsmap[tmsg->inds[i]] * embDataUnitSize];
                tmsg = tmsg->next;
            }
        }
        if(head[dim] != NULL){
            msg tmsg = head[dim];
            while (tmsg != NULL) {
                /* if it's for me, then link recv only*/
                if (tmsg->dst ==HM[mypid]) {
#ifdef NA_DBG_L2
                    na_log(dbgfp, "\t[MR]linking msg dim=%d src=%d dst=%d size=%d snd_dim=%d rcv_dim=%d snd_id=%d rcv_id=%d\n",dim, tmsg->src, tmsg->dst, tmsg->size, tmsg->send_dim, tmsg->recv_dim, tmsg->send_id, tmsg->recv_id);
#endif

                    for (i = 0; i < tmsg->size; ++i) 
                        ec->recvptrs[tmsg->recv_dim][ec->xrecvptrs[tmsg->recv_dim][2] + st_recv_offsets[tmsg->recv_dim][tmsg->recv_id] + i] = &data[indsmap[tmsg->inds[i]] * embDataUnitSize]; 
                }

                else { /* if not, link send and recv to the same buff */
#ifdef NA_DBG_L2
                    na_log(dbgfp, "\t[SF]linking msg dim=%d src=%d dst=%d size=%d snd_dim=%d rcv_dim=%d snd_id=%d rcv_id=%d\n",dim, tmsg->src, tmsg->dst, tmsg->size, tmsg->send_dim, tmsg->recv_dim, tmsg->send_id, tmsg->recv_id);
#ifdef NA_DBG_L3
                    for (i = 0; i < tmsg->size; ++i) {

                        na_log(dbgfp, "\t%d\n",tmsg->inds[i]);
                    }
#endif
#endif
                    for (i = 0; i < tmsg->size; ++i){ 
                        ec->recvptrs[tmsg->recv_dim][ec->xrecvptrs[tmsg->recv_dim][2] + st_recv_offsets[tmsg->recv_dim][tmsg->recv_id] + i] = &(ec->store_buff[dim][(rcnt)*embDataUnitSize]); 
                        ec->sendptrs[ dim ][ ec->xsendptrs[dim][2] + st_send_offsets[ dim ][ tmsg->send_id ] + i ] = &(ec->store_buff[dim][(rcnt++)*embDataUnitSize]); 
                    }
                }
                tmsg = tmsg->next;
            }

        }
    }

    /* finally allocate send and recv buffers*/

    ec->sendbuff = malloc(sizeof(*ec->sendbuff) * (maxSendSize * embDataUnitSize + origDataUnitSize));
    ec->recvbuff = malloc(sizeof(*ec->recvbuff) * (maxRecvSize * embDataUnitSize + origDataUnitSize));

#ifdef NA_DBG
    MPI_Barrier(MPI_COMM_WORLD);
    na_log(dbgfp, "%s\n", "\tdone, now cleanup");
#endif
    /* cleanup */

    for (i = 0; i < ndims; ++i) {
        free(st_recv_offsets[i]); free(st_send_offsets[i]);
        while (head[i] != NULL) {
            tmsg = remove_from_list(&head[i]);
            free(tmsg->inds);
        }
        while (sndList[i] != NULL) {
            tmsg = remove_from_list(&sndList[i]);
            free(tmsg->inds);
        }

    }
#ifdef NA_DBG
    MPI_Barrier(MPI_COMM_WORLD);
    na_log(dbgfp, "%s\n", "\tdone, cleanup1");
#endif
    free(st_recv_offsets); free(st_send_offsets); free(stbuff_sizes);
    free(sendtag);
    free(head); free(sndList); free(HM);
    return ec;
}

/* hypercub-type communication (all-reduce) of ecomm struct */
void ecomm_communicate_allreduce(ecomm *ec, real_t *orig_inp, real_t *orig_out, idx_t embDataUnitSize, idx_t origDataUnitSize){

    idx_t * HI = ec->HI;

#ifdef NA_DBG
    int mype;
    MPI_Comm_rank(MPI_COMM_WORLD, &mype);
    na_log(dbgfp, "\twelcome to ecomm comm all_reduce\n");
#endif
    memcpy(orig_out, orig_inp, sizeof(*orig_inp) * origDataUnitSize);

    idx_t i,j, dim, cond, ndims = ec->ndims, dd, dir;
    dir = ec->dir;
    MPI_Request rqst;
    for (dd = 0; dd < ndims; ++dd) {
        dim = (dir == 0) ? dd : ndims -1 - dd;
        /* issue Irecv */
        MPI_Irecv(ec->recvbuff, (ec->xrecvptrs[dim][CT_CNT]-1) * embDataUnitSize + origDataUnitSize , MPI_REAL_T, HI[ec->neighbor[dim]], 33+dim, MPI_COMM_WORLD, &rqst);

#ifdef NA_DBG
        MPI_Barrier(MPI_COMM_WORLD);
        na_log(dbgfp, "\tIrecv issued, now copy orig data\n");
#endif
        /* prepare the sendbuff */
        /* first, copy the the origData TODO FIXME */
        memcpy(ec->sendbuff, orig_out, sizeof(*orig_inp) * origDataUnitSize);
#ifdef NA_DBG
        MPI_Barrier(MPI_COMM_WORLD);
        na_log(dbgfp, "\torigData copied, now copy emb data\n");
#endif
        /* then copy the embedded data */
        real_t *tptr = ec->sendbuff+origDataUnitSize;
        for (i = ec->xsendptrs[dim][1]; i < ec->xsendptrs[dim][CT_CNT]; ++i) {
            memcpy(tptr, ec->sendptrs[dim][i], sizeof(*ec->sendbuff) * embDataUnitSize);
#ifdef NA_DBG_L3
            na_log(dbgfp, "\tdim=%d i=%d xsendptrs[dim][CT_CNT]=%d data to be sent:\n", dim, i, ec->xsendptrs[dim][CT_CNT]);
            idx_t k;
            for (k = 0; k < embDataUnitSize; ++k) {
                na_log(dbgfp, "\t\t%0.2f ", *ec->sendptrs[dim][i]);
            }
            na_log(dbgfp, "\t\t\n");
#endif
            tptr+= embDataUnitSize;
        }

#ifdef NA_DBG
        MPI_Barrier(MPI_COMM_WORLD);
        na_log(dbgfp, "\temb data copied, now send\n");
#endif
        MPI_Send(ec->sendbuff, (ec->xsendptrs[dim][CT_CNT]-1)*embDataUnitSize + origDataUnitSize, MPI_REAL_T, HI[ec->neighbor[dim]], 33+dim, MPI_COMM_WORLD);

        MPI_Wait(&rqst, MPI_STATUS_IGNORE);

#ifdef NA_DBG
        MPI_Barrier(MPI_COMM_WORLD);
        na_log(dbgfp, "\tsending done now copy orig rcvd data\n");
#endif
        /* copy the recvd data*/
        /* firt, copy to the origData for reduce TODO FIXME */
        for (i = 0; i < origDataUnitSize; ++i) {
            orig_out[i] += ec->recvbuff[i];
        }
#ifdef NA_DBG
        MPI_Barrier(MPI_COMM_WORLD);
        na_log(dbgfp, "\tcopy emb recvd data\n");
        /*           if(dim==1){
         *               volatile idx_t tt = 0;
         *               printf("PID %d on %d ready for attach\n", mype,  getpid());
         *               fflush(stdout);
         *               while (0 == tt)
         *                   sleep(5);
         *           }
         */

#ifdef NA_DBG_L3
        tptr = ec->recvbuff + origDataUnitSize;
        for (i = ec->xrecvptrs[dim][1]; i < ec->xrecvptrs[dim][CT_CNT]; ++i) {
            na_log(dbgfp, "\tdim=%d i=%d xsendptrs[dim][CT_CNT]=%d data to be sent:\n", dim, i, ec->xsendptrs[dim][CT_CNT]);
            idx_t k;
            for (k = 0; k < embDataUnitSize; ++k) {
                na_log(dbgfp, "\t\t%0.2f ", *ec->sendptrs[dim][i]);
                tptr+= embDataUnitSize;
            }
            na_log(dbgfp, "\t\t\n");
        }
#endif
#endif
        /*then, copy emb data */
        tptr = ec->recvbuff + origDataUnitSize;
        for (i = ec->xrecvptrs[dim][1]; i < ec->xrecvptrs[dim][CT_CNT]; ++i) {
            for (j = 0; j < embDataUnitSize; ++j) {
                ec->recvptrs[dim][i][j] = tptr[j];
            }
            //memcpy(ec->recvptrs[dim][i], tptr, sizeof(*ec->recvbuff) * embDataUnitSize);
            tptr+= embDataUnitSize;
        }
#ifdef NA_DBG
        na_log(dbgfp, "\tcomm dim %d done\n", dim);
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    }
}

void free_ecomm(ecomm *ec){
    idx_t i;
    for (i = 0; i < ec->ndims; ++i) {
        if(ec->sendptrs[i] != NULL)
            free(ec->sendptrs[i]);   
        if(ec->recvptrs[i] != NULL)
            free(ec->recvptrs[i]);   
        if(ec->xsendptrs[i] != NULL)
            free(ec->xsendptrs[i]);   
        if(ec->xrecvptrs[i] != NULL)
            free(ec->xrecvptrs[i]);   
        if(ec->store_buff[i] != NULL)
            free(ec->store_buff[i]);   
    }
    free(ec->recvptrs); free(ec->sendptrs); free(ec->xsendptrs); free(ec->xrecvptrs);
    free(ec->store_buff); free(ec->neighbor); free(ec->sendbuff); free(ec->recvbuff);
}
