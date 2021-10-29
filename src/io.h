#ifndef TP_IO_H
#define IP_IO_H


#include "tensor.h"
#include "genst.h"

//idx_t read_dimensions(char* tfile, struct tensor *t);

//idx_t read_dimensions_bin_endian(char* tfile, struct tensor *t);

//idx_t read_dimensions_bin(char* tfile, struct tensor *t);

idx_t read_ckbd_tensor_nonzeros(char tensorfile[], struct tensor *t, struct genst *gs);

 idx_t read_ckbd_tensor_nonzeros(char tensorfile[], struct tensor *t, struct genst *gs);

idx_t read_ckbd_tensor_nonzeros_endian(char tensorfile[], struct tensor *t, struct genst *gs);

idx_t read_fg_partition(char partfile[], struct genst *gs);

tensor * read_fg_tensor(char tensorfile[], char partfile[],  struct genst *gs, idx_t endian);
void read_hc_imap(char filename[], idx_t nmodes, idx_t npes, idx_t **imap_arr);

#endif
