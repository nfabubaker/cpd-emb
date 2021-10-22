#include "comm.h"
#include "cpd.h"
#include "csf.h"
#include "fibertensor.h"
#include "genst.h"
#include "io.h"
#include "mttkrp.h"
#include "stats.h"
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
  printf("\t-m meshstring: (char *) number of processors in each dimension. "
         "e.g.: 4x2x2\n");
  printf("\t-r rank: (int) rank of CP decomposition. default: 16\n");
  printf("\t-i number of CP-ALS iterations (max). default: 10\n");
  printf(
      "\t-a all-to-all communication (0:disable or 1: enabled). default: 0\n");
  printf("\t-f tensor storage option: 0: COO format, 1:CSR-like format 2: CSF "
         "format. default: 1\n");
  printf("\t-s=<vpt string> use virtual process topology with store-and-forward.\n"
         "VPT string structure: ndims-d1-d2-d3-..., where d1xd2x... = num processors\n"
         "VPT string example: 3-4-4-4 for T_3 VPT with 64 processors; default: 2-x-y");

  printf("\t-s use dual communication with each store-and-forward instance (0 "
         "or 1). default: 1\n");
  exit(1);
}

idx_t parse_mesh_str(struct genst *t, char meshstr[1024]) {
  idx_t nmodes = 0, ptr = 0, i;
  char *str;

  str = strtok(meshstr, "x");
  while (str != NULL) {
    nmodes++;
    str = strtok(NULL, "x");
  }

  if (nmodes != t->nmodes) {
    if (t->mype == 0)
      printf("Number of modes of the tensor in the file does not match the "
             "number of modes of the parsed mesh dimensionality string\n");

    MPI_Finalize();
    exit(1);
  }

  t->meshdims = (idx_t *)malloc(sizeof(int) * nmodes);
  t->meshsize = 1;

  for (i = 0; i < nmodes; i++ && str != NULL) {
    str = strtok(&meshstr[ptr], "x");
    t->meshdims[i] = atoi(str);
    t->meshsize *= t->meshdims[i];
    ptr += strlen(str) + 1;
  }
}

idx_t init_param(idx_t argc, char *argv[], char tensorfile[], char partfile[],
               char meshstr[], struct genst *gs, idx_t *niters, idx_t *endian) {
  // set default values
  gs->ckbd = 2;
  gs->cprank = 16;
  gs->fiber = 1;
  gs->alltoall = 0;
  gs->use_stfw = 1;
  gs->use_dual = 0;
  *niters = 10;
  *endian = 0;
  strcpy(meshstr, "auto");
  strcpy(partfile, "");

  idx_t c;
  while ((c = getopt(argc, argv, "c:p:m:r:f:a:i:e:s:d:")) != -1) {
    switch (c) {
    case 'c':
      gs->ckbd = atoi(optarg);
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
    case 's':
      gs->use_stfw = 1;
      strcpy(gs->vptstr, optarg);
      break;
    case 'd':
      gs->use_dual = atoi(optarg);
      break;
    }
  }

  if (argc <= optind)
    printusage(argv[0]);
  if (strcmp(partfile, "") == 0)
    printf("A partition file must be provided\n");

  sprintf(tensorfile, "%s", argv[optind]);
}

void print_stats(char tensorfile[], idx_t *meshdims, struct stats *st, idx_t nmodes,
                idx_t mype, idx_t npes, real_t cptime, real_t *mttkrptime,
                real_t *comm1time, real_t *comm2time, idx_t gnnz, idx_t ckbd,
                idx_t *cnt_st) {
  idx_t i, max, modest[5], maxreduced[5], totreduced[5];
  real_t ltime[3], gtime[3], gcptime, comptime = 0, commtime = 0;
  idx_t myvol = 0, mymsg = 0, myrows = 0, maxrows, maxvol, totvol, maxmsg, totmsg,
      sumrows;

  if (mype == 0) {
    char tname[1024];
    substring(tensorfile, tname);
    printf("%s", tname);

    if (ckbd != 2) {
      printf("_%d", meshdims[0]);
      for (i = 1; i < nmodes; i++)
        printf("x%d", meshdims[i]);
    }
    printf(" ");
  }

  for (i = 0; i < nmodes; i++) {
    myvol += 2 * (st->recvvol[i] + st->sendvol[i]);
    mymsg += 2 * (st->recvmsg[i] + st->sendmsg[i]);
  }

  myrows = 0;
  for (i = 0; i < nmodes; i++)
    myrows += st->row[i];

  MPI_Reduce(&myvol, &maxvol, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&myvol, &totvol, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Reduce(&mymsg, &maxmsg, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&mymsg, &totmsg, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Reduce(&myrows, &sumrows, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&myrows, &maxrows, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

  for (i = 0; i < nmodes; i++) {
    ltime[0] = mttkrptime[i];
    ltime[1] = comm1time[i];
    ltime[2] = comm2time[i];

    MPI_Reduce(ltime, gtime, 3, MPI_REAL_T, MPI_MAX, 0, MPI_COMM_WORLD);
    if (mype == 0) {
      comptime += gtime[0];
      commtime += gtime[1] + gtime[2];
    }
  }

  MPI_Reduce(&cptime, &gcptime, 1, MPI_REAL_T, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&(st->nnz), &max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

  if (mype == 0)
    printf("%lf %d %d %d %d %d %d %lf %lf %lf\n",
           (real_t)max / (gnnz / npes) - 1, maxrows, sumrows / npes, maxvol,
           totvol / npes, maxmsg, totmsg / npes, comptime, commtime, gcptime);
  // printf("%lf %lf %lf\n", comptime, commtime, gcptime);

  /*if(mype == 0)
  {
          for(i = 0; i < nmodes; i++)
                  printf("%d %d - ", cnt_st[i*4+0], cnt_st[i*4+1]);
          printf("\n");
  }*/

  /*if(mype == 0)
    printf("%d %d ", max, gnnz/npes);

  for(i = 0; i < nmodes; i++)
    {
      modest[0] = st->recvvol[i];
      modest[1] = st->sendvol[i];
      modest[2] = st->recvmsg[i];
      modest[3] = st->sendmsg[i];
      modest[4] = st->row[i];

      MPI_Reduce(modest, maxreduced, 5, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(modest, totreduced, 5, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

      if(mype == 0)
        printf("- %d %d %d %d %d %d %d", maxreduced[0], maxreduced[1],
  totreduced[0]/npes, maxreduced[2], maxreduced[3], totreduced[2]/npes,
  maxreduced[4]);

    }

  if(mype == 0)
    printf("\n");
*/
  idx_t maxf = 0, avgf = 0;
  for (i = 0; i < nmodes; i++) {
    ltime[0] = mttkrptime[i];
    ltime[1] = comm1time[i];
    ltime[2] = comm2time[i];

    MPI_Reduce(ltime, gtime, 3, MPI_REAL_T, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&cnt_st[i], &maxf, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&cnt_st[i], &avgf, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (mype == 0)
      printf("%d %d %lf %lf %lf - ", maxf, avgf / npes, gtime[0], gtime[1],
             gtime[2]);
  }
  if (mype == 0)
    printf("\n");
}

void init_stfw_things(struct genst *gs) {
  if (gs->use_stfw) {
    assert(gs->nmodes > 0);
    if (gs->use_dual)
      STFW_init(gs->nmodes, 1);
    else
      STFW_init(gs->nmodes * 2, 0);
  }

  if (gs->use_stfw)
    setup_comm_for_stfw(gs);

#ifdef NA_DBG
  na_log(dbgfp, "dbg p2: after comm init\n");
#endif
  idx_t vpt_ndims, ntokens=0, *vpt_dims;
  char *token = strtok(gs->vptstr, "-");
  vpt_ndims = atoi(token);
  vpt_dims = malloc(sizeof(*vpt_dims) * vpt_ndims);
  while(ntokens < vpt_ndims){
      token = strtok(NULL, "-");
      vpt_dims[ntokens++] = atoi(token);
  }
  
  idx_t pprod = 1, i;
  for (i = 0; i < vpt_ndims; ++i) {
      pprod *= vpt_dims[i];
  }
  assert(pprod == gs->npes);
  struct comm *co = gs->comm;

  if (gs->use_stfw) {
    if (gs->use_dual) {
      for (i = 0; i < gs->nmodes; i++) {
        STFW_init_instance(i, vpt_ndims, vpt_dims, co->nsendwho[i],
                           co->sendwho[i], co->nrecvwho[i], co->recvwho[i],
                           co->ssend[i], co->sendp[i], co->srecv[i],
                           co->recvp[i]);
      }
    } else {
      for (i = 0; i < gs->nmodes * 2; i += 2) {
        STFW_init_instance(
            i, vpt_ndims, vpt_dims, co->nsendwho[i / 2], co->sendwho[i / 2],
            co->nrecvwho[i / 2], co->recvwho[i / 2], co->ssend[i / 2],
            co->sendp[i / 2], co->srecv[i / 2], co->recvp[i / 2]);

#ifdef NA_DBG
        na_log(dbgfp, "dbg p2.1: after stfw init, mode %d\n", i);
#endif
      }
      for (i = 1; i < gs->nmodes * 2; i += 2) {
        STFW_init_instance(
            i, vpt_ndims, vpt_dims, co->nrecvwho[i/2], co->recvwho[i / 2],
            co->nsendwho[i / 2], co->sendwho[i / 2], co->srecv[i / 2],
            co->recvp[i / 2], co->ssend[i / 2], co->sendp[i / 2]);

#ifdef NA_DBG
        na_log(dbgfp, "dbg p2.1: after stfw init 2, mode %d\n", i);
#endif
      }
    }
  }
}

idx_t main(idx_t argc, char *argv[]) {
  idx_t mype, npes, niters, i, endian;
  real_t readtime, setuptime, cptime, cptimewb, totaltime, ltime[3], gtime[3];

  char tensorfile[1024], partfile[1024], meshstr[1024];
  struct tensor *t;
  struct stats *st;
  struct genst *gs;
#ifdef NA_DBG
  na_log(dbgfp, "dbg p0.0: hello there\n");
#endif
  MPI_Init(&argc, &argv);

  totaltime = (real_t)get_wc_time();

  MPI_Comm_rank(MPI_COMM_WORLD, &mype);
  MPI_Comm_size(MPI_COMM_WORLD, &npes);

  t = (struct tensor *)malloc(sizeof(struct tensor));
  gs = (struct genst *)malloc(sizeof(struct genst));
  gs->mype = mype;
  gs->npes = npes;
#ifdef NA_DBG
  na_log(dbgfp, "dbg p0.1: after mpi init\n");
#endif
  init_tensor(t);
  init_genst(gs);
  init_param(argc, argv, tensorfile, partfile, meshstr, gs, &niters, &endian);

#ifdef NA_DBG
  na_log(dbgfp, "dbg p0.2: after param init\n");
#endif
  // read tensor
  readtime = (real_t)get_wc_time();
  read_fg_tensor(tensorfile, partfile, t, gs, endian);

  MPI_Barrier(MPI_COMM_WORLD);
  readtime = ((real_t)get_wc_time() - readtime) / 1000000;

  st = (struct stats *)malloc(sizeof(struct stats));
  init_stats(st, gs->nmodes);

#ifdef NA_DBG
  na_log(dbgfp, "dbg p1: after init\n");
#endif
  // setup the environment
  setuptime = (real_t)get_wc_time();

  setup_fg_communication(gs, t, st);

  init_matrices(gs);

  /* NABIL: this function is added to initialize stfw stuff, others are in comm.c */
  init_stfw_things(gs);

  struct fibertensor *ft = NULL;
  struct csftensor *csftns = NULL;
  if (gs->fiber == 1) {
    ft = (struct fibertensor *)malloc(sizeof(struct fibertensor));
    init_fibertensor(ft);

    get_fibertensor(gs, t, ft);

    free_tensor(t);
  } else if (gs->fiber == 2) {
    csftns = csf_alloc(gs, t);
#ifdef NA_DBG
    na_log(dbgfp, "dbg p3: after csf init\n");
#endif
    /*   idx_t nflops = (gs->nmodes == 3)? csftns->pt->nfibs[1]:
       csftns->pt->nfibs[2]; nflops += gs->nnz; idx_t maxf, avgf;

       MPI_Reduce( &nflops, &maxf, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD );
       MPI_Reduce( &nflops, &avgf, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD );
        if(mype == 0)
          printf("%d %d - ", maxf, avgf/npes);
       if(mype == 0)
          printf("number of flops csf: %d -- %d -- %d -- %d\n",
       gs->nnz,csftns->pt->nfibs[0]+gs->nnz, csftns->pt->nfibs[1]+gs->nnz,
       csftns->pt->nfibs[2]+gs->nnz);
  */
    free_tensor(t);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  setuptime = ((real_t)get_wc_time() - setuptime) / 1000000;

  // cp_als_barrier_fg(t, ft, niters);

  // cp factorization

  cp_als_fg(gs, t, ft, csftns, niters, &cptime);

#ifdef NA_DBG
  na_log(dbgfp, "after cp-als\n");
#endif

  real_t *comm1time, *comm2time, *mttkrptime;
  comm1time = (real_t *)calloc(gs->nmodes, sizeof(real_t));
  comm2time = (real_t *)calloc(gs->nmodes, sizeof(real_t));
  mttkrptime = (real_t *)calloc(gs->nmodes, sizeof(real_t));

  idx_t *cnt_st = (idx_t *)calloc(gs->nmodes, sizeof(int));

  // for comm comp timing purposes

  cp_als_stats_fg(gs, t, ft, csftns, niters, mttkrptime, comm1time, comm2time,
                 cnt_st);
#ifdef NA_DBG
  na_log(dbgfp, "after cp-als-statss\n");
#endif
  print_stats(tensorfile, gs->meshdims, st, gs->nmodes, gs->mype, gs->npes,
             cptime, mttkrptime, comm1time, comm2time, gs->gnnz, gs->ckbd,
             cnt_st);

  free(comm1time);
  free(comm2time);
  free(mttkrptime);
  free(cnt_st);
  if (gs->fiber == 1)
    free_fibertensor(ft, gs->nmodes);
  else if (gs->fiber == 2)
    free_csf(csftns);
  else
    free_tensor(t);
  totaltime = ((real_t)get_wc_time() - totaltime) / 1000000;
  free_genst(gs);
  free_stats(st);
  real_t gcptime;
  MPI_Reduce(&cptime, &gcptime, 1, MPI_REAL_T, MPI_MAX, 0, MPI_COMM_WORLD);
  if (mype == 0)
    printf("%.2f %.2f %.2f %0.2f\n", readtime, setuptime, totaltime, gcptime);
  
  /* NABIL: and don't forget to finalize STFW */
  STFW_finalize();

  MPI_Finalize();

  return 0;
}
