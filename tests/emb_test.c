/**
 * @author      : nabeelooo (nabeelooo@$HOSTNAME)
 * @file        : emb_test
 * @created     : Saturday Jun 19, 2021 11:00:48 +03
 */

#include "../src/ecomm.h"
#include "../src/util.h"
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <sys/stat.h>

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int i, mypid, npes, nsendwho,  *sendwho,  *xsendind,  *sendind,  nrecvwho,  *recvwho,  *xrecvind,  *recvind,  ndims, *indsmap,  embDataUnitSize,  origDataUnitSize;  
    float *data = malloc(sizeof(float) * 4 * 100);
    float origD[2] = {1.0, 1.0};

    MPI_Comm_rank(MPI_COMM_WORLD, &mypid);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);

    for (i = 0; i < 400; ++i) {
        data[i] = ((int)((i/40)) == mypid ? (float)mypid/10:i) ;
    }
#ifdef NA_DBG
    struct stat st = {0};

    if (stat("./dbg_logs", &st) == -1) {
        mkdir("./dbg_logs", 0700);
    }
    sprintf(dbg_fn, "./dbg_logs/outfile-%d-%d", mypid, npes);
    dbgfp = fopen(dbg_fn, "w");
#endif 

    switch (mypid) {
        case 0:
            nsendwho = 1; nrecvwho = 2;
            sendwho = malloc(sizeof(int) * nsendwho); sendwho[0] = 2;
            recvwho = malloc(sizeof(int) * nrecvwho); recvwho[0] = 1; recvwho[1] = 3;
            xsendind = malloc(sizeof(int) * nsendwho+1); xsendind[0] = 0; xsendind[1] = 3;
            xrecvind = malloc(sizeof(int) * nrecvwho+1); xrecvind[0] = 0; xrecvind[1] = 3; xrecvind[2] = 6;
            sendind = malloc(sizeof(int) * 3); sendind[0] = 1; sendind[1] = 2; sendind[2] = 3; 
            recvind = malloc(sizeof(int) * 6); recvind[0] = 11; recvind[1] = 12; recvind[2] = 13; recvind[3] = 31; recvind[4] = 32; recvind[5] = 33;
            break;

        case 1:
            nrecvwho = 1; nsendwho = 3;
            recvwho = malloc(sizeof(int) * nrecvwho); recvwho[0] = 2;
            sendwho = malloc(sizeof(int) * nsendwho); sendwho[0] = 0; sendwho[1] = 2; sendwho[2] = 3;
            xrecvind = malloc(sizeof(int) * nrecvwho+1); xrecvind[0] = 0; xrecvind[1] = 3;
            xsendind = malloc(sizeof(int) * nsendwho+1); xsendind[0] = 0; xsendind[1] = 3; xsendind[2] = 6; xsendind[3] = 9;
            recvind = malloc(sizeof(int) * 3); recvind[0] = 21; recvind[1] = 22; recvind[2] = 23; 
            sendind = malloc(sizeof(int) * 9); sendind[0] = 11; sendind[1] = 12; sendind[2] = 13; sendind[3] = 14; sendind[4] = 15; sendind[5] = 16;sendind[6] = 17; sendind[7] = 18;sendind[8] = 19;

            break;
        case 2: 
            nsendwho = 1; nrecvwho = 2;
            sendwho = malloc(sizeof(int) * nsendwho); sendwho[0] = 1;
            recvwho = malloc(sizeof(int) * nrecvwho); recvwho[0] = 0; recvwho[1] = 1;
            xsendind = malloc(sizeof(int) * nsendwho+1); xsendind[0] = 0; xsendind[1] = 3;
            xrecvind = malloc(sizeof(int) * nrecvwho+1); xrecvind[0] = 0; xrecvind[1] = 3; xrecvind[2] = 6;
            sendind = malloc(sizeof(int) * 3); sendind[0] = 21; sendind[1] = 22; sendind[2] = 23; 
            recvind = malloc(sizeof(int) * 6); recvind[0] = 1; recvind[1] = 2; recvind[2] = 3; recvind[3] = 14; recvind[4] = 15; recvind[5] = 16;
            break;
        case 3:
            nsendwho = 1; nrecvwho = 1;
            sendwho = malloc(sizeof(int) * nsendwho); sendwho[0] = 0;
            recvwho = malloc(sizeof(int) * nrecvwho); recvwho[0] = 1;
            xsendind = malloc(sizeof(int) * nsendwho+1); xsendind[0] = 0; xsendind[1] = 3;
            xrecvind = malloc(sizeof(int) * nrecvwho+1); xrecvind[0] = 0; xrecvind[1] = 3;
            sendind = malloc(sizeof(int) * 3); sendind[0] = 31; sendind[1] = 32; sendind[2] = 33; 
            recvind = malloc(sizeof(int) * 6); recvind[0] = 17; recvind[1] = 18; recvind[2] = 19;
            break;
        default:
            break;

    }
    na_log(dbgfp, "%s\n", "done init");
    embDataUnitSize = 4; origDataUnitSize = 2;
    indsmap = malloc(sizeof(int) * 40);
    for (i = 0; i < 40; ++i) {
        indsmap[i] = i; /* 1-1 mappping */
    }
    int * HI = malloc(sizeof(*HI) * npes);
    for (i = 0; i < npes; ++i) {
       //HI[i] = npes-i; 
       HI[i] = i;
    }
    ecomm *ec = ecomm_setup( nsendwho,  sendwho,  xsendind,  sendind,  nrecvwho,  recvwho,  xrecvind,  recvind,  2, data,  indsmap,  embDataUnitSize,  origDataUnitSize, HI, 0);
    na_log(dbgfp, "%s\n", "done setup");
    /*     {
     *         volatile int tt = 0;
     *         printf("PID %d on %d ready for attach\n", mypid,  getpid());
     *         fflush(stdout);
     *         while (0 == tt)
     *             sleep(5);
     *     }
     */


    int dim, ct_cnt;
    for (dim = 0; dim < ec->ndims; ++dim) {
        for (ct_cnt = 1; ct_cnt < CT_CNT; ++ct_cnt) {
            for (i = ec->xsendptrs[dim][ct_cnt]; i < ec->xsendptrs[dim][ct_cnt+1]; ++i) {
                na_log(dbgfp, "\t\t*sendptr[%d][%d] = %f", dim, i , *ec->sendptrs[dim][i]);	
            }	
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    float origD_out[2];
    ecomm_communicate_allreduce(ec, origD, origD_out, embDataUnitSize, origDataUnitSize, 0);

            na_log(dbgfp, "origData_out: %f %f ",origD_out[0], origD_out[1]); 
    int j;
    if(ec->store_buff[1] != NULL)
        for (i = 0; i < 3; ++i) {
            na_log(dbgfp, "st_buff[1][%d]=%f\n",i, ec->store_buff[1][i]);
        }
    for (i = 0; i < 100; ++i) {
        na_log(dbgfp, "[%d] ", i);
        for (j = 0; j < 4; ++j) {
            na_log(dbgfp, "%0.2f ", data[(i*4)+j]);
        }
        na_log(dbgfp, "\n");
    }
    na_log(dbgfp, "%s\n", "done comm");

    MPI_Finalize();
    return 0;
}
