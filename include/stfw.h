/**
 * @author      : Nabil Abubaker (nabil.abubaker@bilkent.edu.tr)
 * @file        : stfw
 * @created     : Perşembe Tem 16, 2020 14:06:55 +03
 */

#ifndef LIBSTFW_H

#define LIBSTFW_H

#define RATIONAL double

#define MPI_RATIONAL MPI_DOUBLE

/*  early initialization, just to know # of instances and if there will be dual communication (like expand-fold) */
void STFW_init(int _nInstances, unsigned char dual_enabled);

/******************************************************************************
 * Function: STFW_init_instance
 * Description: 
 * @args:
 *   _instance_id   :   id of VPT instance
 *   vpt_ndims      :   number of VPT dims
 *   vpt_dsizes     :   sizes of each dim
 *   npsend         :   #processors in my send list
 *   sendlist       :   processor IDs in my send list
 *   nprecv         :   #prcs in my recv list
 *   recvlist       : procs IDs in my recv list
 *   ssend          : size of message to be sent to sendlist[i]
 *   srecv          : size of message to be recvd from recvlist[i]
 *   sendp          : of size npsend, contains pointers to send buffers for each p in sendlist
 *   recvp          : of size nprecv, contains pointer to recv buffers for p in recvlist
 *   Where:
 *   Return:
 *   Error:
 *****************************************************************************/
void STFW_init_instance(int _instance_id, int vpt_ndims, int *vpt_dsizes,
                        int npsend, int *sendlist, int nprecv, int *recvlist,
                        int *ssend, RATIONAL **sendp, int *srecv,
                        RATIONAL **recvp);

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  STFW__Comm
 *  Description: Performs the actual P2P communication for instance X, according to the
 *  initialized communication struture 
 * =====================================================================================
 */
void STFW_Comm(int _instance_id);


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  STFW_dual_Comm
 *  Description: Performs dual P2P communication of instance X. In this function send
 *  and recv buffered, as well as VPT execution order, are switched 
 * =====================================================================================
 */
void STFW_dual_Comm(int _instance_id);
void STFW_finalize();


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  STFW_stats
 *  Description:  returns stats of the VPT, like max/totla volume/ messages ..etc
 *  @args:
 *  [IN] _intance_id: ID of the VPT instance to get stats from.
 *  [OUT] *maxSendVol: maximum send volume of all VPT stages
 *  [OUT] *maxRecvVol: maximum send volume of all VPT stages
 *  [OUT] *totalVol: total volume of all VPT stages
 *  [OUT] *maxSendMsgs: maximum send messages of all VPT stages
 *  [OUT] *maxRecvMsgs: maximum recv messages of all VPT stages
 *  [OUT] *totalMsgs: total messages of all VPT stages
 *  [IN] factor: factor by which comm volume is multiplied by, for instance
 * =====================================================================================
 */
void STFW_stats(int _instance_id, int *maxSendVol, int *maxRecvVol, int *totalVol , int *maxSendMsgs, int *maxRecvMsgs, int *totalMsgs, int factor);

#endif /* end of include guard LIBSTFW_H */

