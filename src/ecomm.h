
/*
 *
 *
 * =====================================================================================
 *
 *       Filename:  ecomm.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  07-06-2021 13:23:13
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#include "basic.h"
#define CT_CNT 3 //comm type count (currently 3: orig, emb mine, emb fw)
typedef struct _ecomm{
    idx_t ndims;
    idx_t *neighbor; //send neighbors for each dim
    idx_t **xsendptrs; // CSR-like starting and ending indices of message i
    idx_t **xrecvptrs;
    real_t ***sendptrs; // points to send items  
    real_t ***recvptrs;  // points to recv items
    real_t **store_buff;
    real_t *sendbuff, *recvbuff;
    idx_t *HI;
    idx_t dir;

} ecomm;

struct _msg{
    /* recv dim records the dimension at which the item is recvd, while recv_id stores the location in recv buffer at that stage */
        idx_t src, dst, size, send_dim, send_id, recv_dim, recv_id;
	    idx_t *inds;
        struct _msg *next;
};

typedef struct _msg *msg; 

ecomm *ecomm_setup(idx_t nsendwho, idx_t *sendwho, idx_t *xsendind, idx_t *sendind, idx_t nrecvwho, idx_t *recvwho, idx_t *xrecvind, idx_t *recvind, idx_t ndims, real_t *data, idx_t *indsmap,  idx_t embDataUnitSize, idx_t origDataUnitSize, idx_t * hypercube_imap, idx_t dir);

void ecomm_communicate_allreduce(ecomm *ec, real_t *orig_inp, real_t *orig_out, idx_t embDataUnitSize, idx_t origDataUnitSize);
void free_ecomm(ecomm *ec);
void emb_get_stats(ecomm *ec, idx_t *maxSendVol, idx_t *maxRecvVol, idx_t *totalVol , idx_t *maxSendMsgs, idx_t *maxRecvMsgs, idx_t *totalMsgs);
