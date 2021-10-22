#ifndef TP_STAT_H
#define TP_STAT_H


typedef struct stats
{
  idx_t *recvvol;
  idx_t *sendvol;

  idx_t *recvmsg;
  idx_t *sendmsg;

  idx_t *row;
  idx_t nnz;

} stats;

void init_stats(stats *st, idx_t nmodes);

void free_stats(stats *st);

#endif
