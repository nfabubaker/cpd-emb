#include "comm.h"
#include <sys/stat.h>
#include "cpd.h"
#include "csf.h"
#include "fibertensor.h"
#include "genst.h"
#include "io.h"
#include "mttkrp.h"
#include "stat.h"
#include "tensor.h"
#include "util.h"
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <mpi.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define min(x, y) (((x) < (y)) ? (x) : (y))

void printusage(char *exec) {
    printf("usage: %s [options] tensorfile\n", exec);
    printf("options:\n");
    printf("\t-p partitionfile: (char *) in the partition file, indices numbered "
            "in the mode order\n");
    printf("\t-r rank: (int) rank of CP decomposition. default: 16\n");
    printf("\t-i number of CP-ALS iterations (max). default: 10\n");
    printf("\t-c communication type, can be one of the following:\n"
            "\t\t 0: Point-to-point communication (default), you can specify -a option with this type\n"
            "\t\t 2: Embedded communication (hypercube), use -d and -b option with this type\n");
    printf(
            "\t-a all-to-all communication (0:disable or 1: enabled). default: 0\n");
    printf("\t-f tensor storage option: 0: COO format, 1:CSR-like format 2: CSF "
            "format. default: 1\n");
    printf("\t-b use hypercube imap file for embedded communication, a file name should be provided\n");
    exit(1);
}


idx_t init_param(int argc, char *argv[], char tensorfile[], char partfile[],
        char meshstr[], struct genst *gs, idx_t *niters, idx_t *endian) {
    // set default values
    gs->comm_type = 0; //p2p comm
    gs->cprank = 16;
    gs->fiber = 1;
    gs->alltoall = 0;
    gs->use_hc_imap = 0;
    *niters = 10;
    *endian = 0;
    strcpy(meshstr, "auto");
    strcpy(partfile, "");

    idx_t c;
    while ((c = getopt(argc, argv, "c:p:m:r:f:a:i:e:s:d:b:")) != -1) {
        switch (c) {
            case 'c':
                gs->comm_type = atoi(optarg);
                break;
            case 'p':
                strcpy(partfile, optarg);
                break;
            case 'm':
                strcpy(meshstr, optarg);
                break;
            case 'r':
                gs->cprank = atoi(optarg);
                break;
            case 'f':
                gs->fiber = atoi(optarg);
                break;
            case 'a':
                gs->alltoall = atoi(optarg);
                break;
            case 'i':
                *niters = atoi(optarg);
                break;
            case 'e':
                *endian = atoi(optarg);
                break;
            case 'b':
                gs->use_hc_imap = 1;
                strcpy(gs->hc_imap_FN, optarg);
                break;
        }
    }

    if (argc <= optind)
        printusage(argv[0]);
    if (strcmp(partfile, "") == 0)
        printf("A partition file must be provided\n");

    sprintf(tensorfile, "%s", argv[optind]);
}

void print_stats_v2(char tensorfile[], struct stats *st, double cptime, double *mttkrptime, double *comm1time, double *comm2time, double *mmtime, double *otherstime, genst *gs){

    idx_t nStats = 8, nstfw_s = 2, nstfw_m=4;
    idx_t maxmf, maxme, maxvf, maxve, totm, totv, maxr, totr, i, j;
    double setupT, foldT, expandT, mttkrpT, totT, othersT, mmT, perMT_IN[5], perMT_out[5];
    idx_t **pmStats, **stfw_m, **stfw_s;
    idx_t stfw_maxmf, stfw_maxme, stfw_maxvf, stfw_maxve, stfw_totm, stfw_totv;
    char ftname[1024], tname[1024];

    pmStats = malloc(sizeof(*pmStats)*gs->nmodes);
    if( gs->comm_type == EMB){
        stfw_s = malloc(sizeof(*stfw_s) * gs->nmodes);
        stfw_m = malloc(sizeof(*stfw_m) * gs->nmodes);
    }

    for (i = 0; i < gs->nmodes; ++i) {
        pmStats[i] = calloc(nStats, sizeof(**pmStats));
        if ( gs->comm_type == EMB) {
            stfw_m[i] = calloc(nstfw_m, sizeof(**stfw_m));
            stfw_s[i] = calloc(nstfw_s, sizeof(**stfw_s));
        }   
    }

    if (gs->mype == 0) {
        substring(tensorfile, ftname);
        substring_b(tname, ftname);
        printf("%s %d", tname, gs->npes);
    }

    /* first pridx_t total statss */
    for (i = 0; i < gs->nmodes; ++i) {
        if(gs->comm_type == EMB){
            emb_get_stats(gs->comm->ec[i*2+1],&stfw_m[i][0], &stfw_m[i][1], &stfw_s[i][0], &stfw_m[i][2] , &stfw_m[i][3], &stfw_s[i][1]);
        }
        MPI_Reduce(&st->sendmsg[i], &pmStats[i][0], 1, MPI_IDX_T, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&st->recvmsg[i], &pmStats[i][1], 1, MPI_IDX_T, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&st->sendmsg[i], &pmStats[i][2], 1, MPI_IDX_T, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&st->sendvol[i], &pmStats[i][3], 1, MPI_IDX_T, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&st->recvvol[i], &pmStats[i][4], 1, MPI_IDX_T, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&st->sendvol[i], &pmStats[i][5], 1, MPI_IDX_T, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&st->row[i], &pmStats[i][6], 1, MPI_IDX_T, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&st->row[i], &pmStats[i][7], 1, MPI_IDX_T, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    foldT = expandT = mttkrpT = totT = mmT = othersT = 0.0;
    for (i = 0; i < gs->nmodes; ++i) {
        perMT_IN[0] = mttkrptime[i]; perMT_IN[1] = comm1time[i]; perMT_IN[2] = comm2time[i]; perMT_IN[3] = mmtime[i];
        perMT_IN[4] = otherstime[i];
        MPI_Reduce(perMT_IN, perMT_out, 5, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        foldT   += perMT_out[1];
        expandT += perMT_out[2];
        mttkrpT += perMT_out[0];
        mmT     += perMT_out[3];
        othersT += perMT_out[4];
    }
    if(gs->mype == 0){
        /* pridx_t total statss */
        maxmf = maxme = totm = maxvf = maxve = totv= maxr = totr = 0;
        stfw_maxme = stfw_maxmf = stfw_maxve = stfw_maxvf = stfw_totm = stfw_totv = 0;
        for (i = 0; i < gs->nmodes; ++i) {
            maxmf   += pmStats[i][0];
            maxme   += pmStats[i][1];
            totm    += pmStats[i][2];
            maxvf   += pmStats[i][3];
            maxve   += pmStats[i][4];
            totv    += pmStats[i][5];
            maxr    += pmStats[i][6];
            totr    += pmStats[i][7];
            if ( gs->comm_type == EMB) {
                stfw_maxvf += stfw_m[i][0];
                stfw_maxve += stfw_m[i][1];
                stfw_maxmf += stfw_m[i][2];
                stfw_maxme += stfw_m[i][3];
                stfw_totv  += stfw_s[i][0];
                stfw_totm  += stfw_s[i][1];
            }
        }
        /* pridx_t statss without stfw */
        printf(" %d %d %d %d %d %d", maxvf, maxve, totv, maxmf, maxme, totm);
        /* pridx_t stfw-based statss */
        if( gs->comm_type == EMB)
            printf(" %d %d %d %d %d %d", stfw_maxvf, stfw_maxve, stfw_totv, stfw_maxmf, stfw_maxme, stfw_totm);
        /* pridx_t other statss */
        double mult = 1e-6;
        printf(" %d %d %f %f %f %f %f %f\n", maxr, totr, mttkrpT*mult, foldT*mult, expandT*mult, mmT*mult, othersT*mult, cptime*mult);

        /* Now per-mode statss */
        for (i = 0; i < gs->nmodes; ++i) {
            idx_t *st = pmStats[i], *sst_m, *sst_s;
            if( gs->comm_type == EMB){
                sst_m = stfw_m[i];
                sst_s = stfw_s[i];
            }
            printf("%s %d %d %d %d %d %d %d",tname, gs->npes, st[3], st[4], st[5], st[0], st[1], st[2]);
            /* pridx_t stfw-based statss */
            if( gs->comm_type == EMB)
                printf(" %d %d %d %d %d %d", sst_m[0], sst_m[1], sst_s[0], sst_m[2], sst_m[3], sst_s[1]);
            /* pridx_t other statss */
            printf(" %d %d %f %f %f %f\n", st[6], st[7], mult*mttkrptime[i], mult*comm1time[i], mult*comm2time[i], 0.0);
        }
    }

}

idx_t main(idx_t argc, char *argv[]) {
    idx_t mype, npes, niters, i, endian;
    double readtime, setuptime, cptime, cptimewb, totaltime, ltime[3], gtime[3];

    char tensorfile[1024], partfile[1024], meshstr[1024];
    struct tensor *t;
    struct stats *st;
    struct genst *gs;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &mype);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);

    t = (struct tensor *)malloc(sizeof(struct tensor));
    gs = (struct genst *)malloc(sizeof(struct genst));
    gs->mype = mype;
    gs->npes = npes;
#ifdef NA_DBG
    struct stat stt = {0};

    if (stat("./dbg_logs", &stt) == -1) {
        mkdir("./dbg_logs", 0700);
    }
    sprintf(dbg_fn, "./dbg_logs/outfile-%d-%d", mype, npes);
    dbgfp = fopen(dbg_fn, "w");
    
    na_log(dbgfp, "dbg p0.1: after mpi init\n");
#endif
    init_tensor(t);
    init_genst(gs);
    init_param(argc, argv, tensorfile, partfile, meshstr, gs, &niters, &endian);

#ifdef NOF_DBG
    fprintf(stderr, "dbg p0.2: after param init\n");
#endif
#ifdef NA_DBG
    na_log(dbgfp, "dbg p0.2: after param init\n");
#endif
    // read tensor
    read_fg_tensor(tensorfile, partfile, t, gs, endian);

    MPI_Barrier(MPI_COMM_WORLD);

    st = (struct stats *)malloc(sizeof(struct stats));
    init_stats(st, gs->nmodes);

#ifdef NOF_DBG
    fprintf(stderr, "dbg p1: after init\n");
#endif
#ifdef NA_DBG
    na_log(dbgfp, "dbg p1: after init\n");
#endif
    // setup the environment
    setup_comm(gs, t, st);
    struct fibertensor *ft = NULL;
    struct csftensor *csftns = NULL;
    if (gs->fiber == 1) {
        ft = (struct fibertensor *)malloc(sizeof(struct fibertensor));
        init_fibertensor(ft);

        get_fibertensor(gs, t, ft);

        free_tensor(t);
    } else if (gs->fiber == 2) {
        csftns = csf_alloc(gs, t);
#ifdef NOF_DBG
        MPI_Barrier(MPI_COMM_WORLD);
    fprintf(stderr, "dbg p3: after csf init\n");
#endif
#ifdef NA_DBG
        na_log(dbgfp, "dbg p3: after csf init\n");
#endif
        free_tensor(t);
    }

#ifdef NA_DBG
    MPI_Barrier(MPI_COMM_WORLD);
    na_log(dbgfp, "after cp-als\n");
#endif

    double *comm1time, *comm2time, *mttkrptime, *otherstime, *mmtime;
    comm1time = (double *)calloc(gs->nmodes, sizeof(double));
    comm2time = (double *)calloc(gs->nmodes, sizeof(double));
    mttkrptime = (double *)calloc(gs->nmodes, sizeof(double));
    otherstime = (double *)calloc(gs->nmodes, sizeof(double));
    mmtime = (double *)calloc(gs->nmodes, sizeof(double));


    idx_t *cnt_st = (idx_t *)calloc(gs->nmodes, sizeof(int));
    double cptime2;
    if(gs->comm_type == EMB){
        cp_als_fg_emb(gs, t, ft, csftns, niters, &cptime);
#ifdef NOF_DBG
        MPI_Barrier(MPI_COMM_WORLD);
    fprintf(stderr, "after cp-als-emb\n");
#endif
#ifdef NA_DBG
        MPI_Barrier(MPI_COMM_WORLD);
    na_log(dbgfp, "after cp-als-emb\n");
#endif
        cp_als_fg_emb_time(gs,t, ft, csftns, niters, &cptime2, mmtime, otherstime, mttkrptime, comm1time, comm2time);

#ifdef NA_DBG
        MPI_Barrier(MPI_COMM_WORLD);
    na_log(dbgfp, "after cp-als-emb_time\n");
#endif
    }
    else{
        cp_als_fg(gs, t, ft, csftns, niters, &cptime);
        cp_als_fg_time(gs,t, ft, csftns, niters, &cptime2, mmtime, otherstime, mttkrptime, comm1time, comm2time);
#ifdef NA_DBG
    na_log(dbgfp, "after cp-als-statss\n");
#endif
    }
    double gcptime;
    MPI_Reduce(&cptime, &gcptime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    print_stats_v2(tensorfile, st, gcptime, mttkrptime, comm1time, comm2time, mmtime, otherstime, gs);
#ifdef NOF_DBG
        MPI_Barrier(MPI_COMM_WORLD);
    fprintf(stderr, "dbg: after pridx_t stat\n");
#endif
    free(comm1time); free(comm2time);free(mttkrptime); free(mmtime); free(otherstime);
    free(cnt_st);
/*     if (gs->fiber == 1)
 *         free_fibertensor(ft, gs->nmodes);
 *     else if (gs->fiber == 2)
 *         free_csf(csftns);
 *     else
 *         free_tensor(t);
 *     free_genst(gs);
 *     free_stats(st);
 */
    MPI_Finalize();

    return 0;
}
