#ifndef TP_COMM_H
#define TP_COMM_H

#include "ecomm.h"
#include "util.h"
#include "stat.h"
#include "genst.h"
#include "tensor.h"
#include "io.h"

enum COMM_TYPE {P2P, STFW, EMB };

typedef struct comm
{
  idx_t *nrecvwho;
  idx_t **recvwho;
  idx_t **xrecvind; // number of factor matrix rows to be received from each comunicated processor at each mode
  // e.g. in a 2X3x4 mesh, processor 0 is going to communicate with 0 2 4 .. 22 for first mode. nrecv[0][2] for proc 0 denotes the amount communicated with proc 4. pairs of (pid and number)
  idx_t **recvind;

  idx_t *nsendwho;
  idx_t **sendwho;
  idx_t **xsendind;
  idx_t **sendind;

  real_t *buffer;

  /* emb stuff */
  ecomm **ec;
  idx_t **hypercube_imap;

} comm;

void init_comm (struct genst *gs);
void setup_comm(struct genst *gs, tensor *t, stats *st);
void setup_fg_communication(struct genst *gs, struct tensor *t, struct stats *st);
void receive_partial_products_fg(struct genst *gs, idx_t mode, real_t *matm);
void send_updated_rows_fg(struct genst *gs, idx_t mode);
void free_comm(struct comm *c, idx_t nmodes);
#endif
