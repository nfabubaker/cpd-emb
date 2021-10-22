#include <stdlib.h>
#include "stat.h"
#include "util.h"

void init_stats(struct stats *st, idx_t nmodes)
{
  st->recvvol = (idx_t *)malloc(nmodes*sizeof(int));
  st->sendvol = (idx_t *)malloc(nmodes*sizeof(int));

  st->recvmsg = (idx_t *)malloc(nmodes*sizeof(int));
  st->sendmsg = (idx_t *)malloc(nmodes*sizeof(int));

  st->row = (idx_t *)malloc(nmodes*sizeof(int));
	
  setintzero(st->recvvol, nmodes);
  setintzero(st->sendvol, nmodes);
  setintzero(st->recvmsg, nmodes);
  setintzero(st->sendmsg, nmodes);
  setintzero(st->row, nmodes);

}


void free_stats(struct stats *st)
{
  if(st->recvvol != NULL)
    free(st->recvvol);

  if(st->sendvol != NULL)
    free(st->sendvol);

  if(st->recvmsg != NULL)
    free(st->recvmsg);

  if(st->sendmsg != NULL)
    free(st->sendmsg);

  if(st->row != NULL)
    free(st->row);

  free(st);

}
