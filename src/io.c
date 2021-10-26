#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "io.h"
#include "mpi.h"
#include <string.h>
#include <assert.h>

//assumes the first line of the tensor starts with # and denotes dimension info
/*idx_t read_dimensions(char* tfile, struct tensor *t){

  FILE *tf = fopen(tfile, "r");
  char line[1024];
  fgets(line, 1024, tf);
  fclose(tf);

  idx_t nmodes = -1, ptr = 1, i;
  char * str;

  str = strtok (line, "\t");
  while (str != NULL){
  nmodes++; 
  str = strtok (NULL, "\t");
  }

  t->gdims = (idx_t *)malloc(sizeof(int)*nmodes);

  for(i = 0; i < nmodes; i++ && str != NULL)
  {
  str = strtok (&line[ptr], "\t");
  t->gdims[i] = atoi(str);
  ptr += strlen(str) + 1;
  }

  str = strtok (&line[ptr], "\t");
  t->gnnz = atoi(str);

  t->nmodes = nmodes;

  return 0;
  }


  idx_t read_dimensions_bin_endian(char* tfile, struct tensor *t){

  idx_t nmodes, j;
  FILE *tf = fopen(tfile, "rb");

  fread(&nmodes, sizeof(int), 1, tf);
  nmodes = convert(nmodes);

  t->gdims = (idx_t *)malloc(sizeof(int)*nmodes);
  fread(t->gdims, sizeof(int), nmodes, tf);
  for(j = 0; j < nmodes+2; j++)
  t->gdims[j] = convert(t->gdims[j]);

  fread(&(t->gnnz), sizeof(int), 1, tf);
  t->gnnz = convert(t->gnnz);
  t->nmodes = nmodes;

  fclose(tf);

  return 0;
  }


//assumes nmodes - dim1, dim2, .. dimnmodes - totalnnz - mynnz
idx_t read_dimensions_bin(char* tfile, struct tensor *t){

idx_t nmodes;
FILE *tf = fopen(tfile, "rb");

fread(&nmodes, sizeof(int), 1, tf);
t->gdims = (idx_t *)malloc(sizeof(int)*nmodes);
fread(t->gdims, sizeof(int), nmodes, tf);
fread(&(t->gnnz), sizeof(int), 1, tf);
t->nmodes = nmodes;

fclose(tf);

return 0;
}*/
idx_t read_ckbd_tensor_nonzeros_large(char tensorfile[], struct tensor *t, struct genst *gs)
{
    typedef int64_t inpf_t;
#define MPI_INPF_T MPI_INT64_T
    idx_t i, j, offset_all,  offset,  nnz, *inds, val, total ;
    inpf_t *tmp, buf[2];
    idx_t nmodes; 
    real_t *vals;
    char line[1024], *str;
    MPI_File fh;
    MPI_Status statsus;

    assert(sizeof(idx_t) == sizeof(real_t));
    idx_t of;

    nmodes = gs->nmodes;
    offset = gs->mype*2*sizeof(inpf_t) + gs->mype * sizeof(int);
    offset_all = gs->npes*(2*sizeof(inpf_t) + sizeof(int));

    //MPI_Barrier(MPI_COMM_WORLD);
    //printf("b4 mpi read \n");
#ifdef NA_DBG
    na_log(dbgfp, "\thello from read_ckbd_tensor_nonzeros: before MPI_File_open\n");
#endif
    int tnmodes;
    MPI_File_open(MPI_COMM_WORLD, tensorfile, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_File_read_at(fh, offset, &tnmodes, 1, MPI_INT, &statsus);
    MPI_File_read_at(fh, offset+sizeof(int), buf, 2, MPI_INPF_T, &statsus);
    nmodes = (idx_t) tnmodes;
    assert(nmodes == gs->nmodes);
#ifdef NA_DBG
    na_log(dbgfp, "\t\tafter MPI_File_read_at\n");
#endif
    nnz = (idx_t) buf[0];
    of = (idx_t) buf[1]*(nmodes+1)*sizeof(inpf_t) + offset_all;
    t->inds =  malloc(nmodes*nnz*sizeof(*t->inds));
    t->vals =  malloc(nnz*sizeof(*t->vals));
    t->nnz = nnz;
    idx_t tmpsize = (nmodes+1)*nnz;
    tmp =  malloc(sizeof(*tmp) * tmpsize);
    //tmp = (idx_t *)malloc((nmodes+1)*nnz*sizeof(int));

#ifdef NA_DBG
    na_log(dbgfp, "\t\tafter allocations\n");
#endif
    MPI_Allreduce(&nnz, &t->gnnz, 1, MPI_IDX_T, MPI_SUM, MPI_COMM_WORLD);
    inds = t->inds;
    vals = t->vals;

#ifdef NA_DBG
    MPI_Barrier(MPI_COMM_WORLD);
    na_log(dbgfp, "\t\tafter all_reduce\n");
#endif
    MPI_File_read_at(fh, of, tmp, (nmodes+1)*nnz, MPI_INPF_T, &statsus);
    MPI_File_close(&fh); 

#ifdef NA_DBG
    MPI_Barrier(MPI_COMM_WORLD);
    na_log(dbgfp, "\t\tafter 2nd read_at\n");
#endif
    for(i = 0; i < nnz; i++)
    {
        //memcpy(&inds[i*nmodes], &tmp[i*(nmodes+1)], sizeof(idx_t)*nmodes); 
        for (j = 0; j < nmodes; ++j) {
            inds[i*nmodes+j] = (idx_t) tmp[i*(nmodes+1) +j];
        }
        vals[i] = (real_t) tmp[i*(nmodes+1)+nmodes];
    }
#ifdef NA_DBG
    MPI_Barrier(MPI_COMM_WORLD);
    na_log(dbgfp, "\t\tafter memcpy\n");
#endif
    return 0;
}

idx_t read_ckbd_tensor_nonzeros(char tensorfile[], struct tensor *t, struct genst *gs)
{
    typedef int inpf_t;
#define MPI_INPF_T MPI_INT
    idx_t i, j, offset,  nnz, *inds, val, total ;
    inpf_t *tmp, *buf;
    idx_t nmodes; 
    real_t *vals;
    char line[1024], *str;
    MPI_File fh;
    MPI_Status statsus;

    assert(sizeof(idx_t) == sizeof(real_t));
    idx_t of;

    nmodes = gs->nmodes;
    offset = gs->mype*3*sizeof(inpf_t);
    buf = malloc((nmodes+1)*sizeof(*buf));

    //MPI_Barrier(MPI_COMM_WORLD);
    //printf("b4 mpi read \n");
#ifdef NA_DBG
    na_log(dbgfp, "\thello from read_ckbd_tensor_nonzeros: before MPI_File_open\n");
#endif

    MPI_File_open(MPI_COMM_WORLD, tensorfile, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_File_read_at(fh, offset, buf, 3, MPI_INPF_T, &statsus);
#ifdef NA_DBG
    na_log(dbgfp, "\t\tafter MPI_File_read_at\n");
#endif
    nnz = (idx_t) buf[1];
    of = (idx_t) buf[2]*(nmodes+1)*sizeof(inpf_t)+gs->npes*3*sizeof(inpf_t);
    t->inds =  malloc(nmodes*nnz*sizeof(*t->inds));
    t->vals =  malloc(nnz*sizeof(*t->vals));
    t->nnz = nnz;
    idx_t tmpsize = (nmodes+1)*nnz;
    tmp =  malloc(sizeof(*tmp) * tmpsize);
    //tmp = (idx_t *)malloc((nmodes+1)*nnz*sizeof(int));

#ifdef NA_DBG
    na_log(dbgfp, "\t\tafter allocations\n");
#endif
    MPI_Allreduce(&nnz, &t->gnnz, 1, MPI_IDX_T, MPI_SUM, MPI_COMM_WORLD);
    inds = t->inds;
    vals = t->vals;

#ifdef NA_DBG
    MPI_Barrier(MPI_COMM_WORLD);
    na_log(dbgfp, "\t\tafter all_reduce\n");
#endif
    MPI_File_read_at(fh, of, tmp, (nmodes+1)*nnz, MPI_INPF_T, &statsus);
    MPI_File_close(&fh); 

#ifdef NA_DBG
    MPI_Barrier(MPI_COMM_WORLD);
    na_log(dbgfp, "\t\tafter 2nd read_at\n");
#endif
    for(i = 0; i < nnz; i++)
    {
        //memcpy(&inds[i*nmodes], &tmp[i*(nmodes+1)], sizeof(idx_t)*nmodes); 
        for (j = 0; j < nmodes; ++j) {
            inds[i*nmodes+j] = (idx_t) tmp[i*(nmodes+1) +j];
        }
        vals[i] = (real_t) tmp[i*(nmodes+1)+nmodes];
    }
#ifdef NA_DBG
    MPI_Barrier(MPI_COMM_WORLD);
    na_log(dbgfp, "\t\tafter memcpy\n");
#endif
    free(buf);

}

idx_t read_ckbd_tensor_nonzeros_endian(char tensorfile[], struct tensor *t, struct genst *gs)
{
    idx_t i, j, nnz, *inds, val, total;
    idx_t nmodes, *buff;
    real_t *vals;
    char line[1024], *str;
    FILE *ftensor;

    nmodes = gs->nmodes;
    buff = malloc((nmodes+2)*sizeof(*buff));

    ftensor = fopen(tensorfile, "rb");    
    fread(buff, sizeof(idx_t), nmodes+2, ftensor);
    for(j = 0; j < nmodes+2; j++)
        buff[j] = convert(buff[j]);

    t->gnnz = buff[nmodes+1];

    fread(&nnz, sizeof(idx_t), 1, ftensor);
    nnz = convert(nnz);

    t->inds = (idx_t *)malloc(nmodes*nnz*sizeof(*t->inds));
    t->vals = (real_t *)malloc(nnz*sizeof(*t->vals));
    t->nnz = nnz;

    inds = t->inds;
    vals = t->vals;

    idx_t b = 0;
    for(i = 0; i < nnz; i++)
    {
        fread(&inds[b], sizeof(idx_t), nmodes, ftensor);
        for(j = 0; j < nmodes; j++)
            inds[b+j] = convert(inds[b+j]);
        b += nmodes;

        fread(&val, sizeof(real_t), 1, ftensor);
        val = convert(val);
        vals[i] = (real_t) val;

        if(vals[i] == 0)
            vals[i] = 1.1;
    }
    fclose(ftensor);
    free(buff);
}

idx_t read_fg_partition(char partfile[], struct genst *gs)
{
    idx_t i, j ;
    int  nmodes, npes;
    idx_t dim;
    FILE *fpart;
    char line[128];


    if (gs->mype == 0) {
        fpart = fopen(partfile, "r");
        fgets(line, 128, fpart);
        sscanf(line, "##%d %d\n", &nmodes, &npes);
        if((idx_t)npes != gs->npes)
        {
            if(gs->mype == 0)
                printf("The partition file is for %d processors!!\n", npes);
            MPI_Finalize();
            exit(1);
        }
    }
    MPI_Bcast(&nmodes,  1, MPI_INT, 0, MPI_COMM_WORLD);
    gs->nmodes = (idx_t) nmodes;
    gs->gdims = (idx_t *)malloc(nmodes*sizeof(*gs->gdims));

    gs->interpart = malloc(sizeof(*gs->interpart)*nmodes);
    for(i = 0; i < gs->nmodes; i++)
    {
        if (gs->mype == 0) {
            fgets(line, 128, fpart);
            sscanf(line, "#%zu\n", &dim);
        }
        MPI_Bcast(&dim,  1, MPI_IDX_T, 0, MPI_COMM_WORLD);
        gs->gdims[i] = (idx_t) dim; 

        gs->interpart[i] = malloc(sizeof(*gs->interpart[i])*gs->gdims[i]);

        int pp;
        if (gs->mype == 0) {
            for(j = 0; j < gs->gdims[i]; j++){
                fgets(line, 128, fpart);
                sscanf(line, "%d\n", &pp);
                gs->interpart[i][j] = (idx_t) pp;
            }
        }
        MPI_Bcast(gs->interpart[i],  gs->gdims[i], MPI_IDX_T, 0, MPI_COMM_WORLD);

    }

    if (gs->mype == 0) {
        fclose(fpart);
    }
}

void read_hc_imap(char filename[], idx_t nmodes, idx_t npes, idx_t **imap_arr)
{
    idx_t i, j;
    int mype;
    FILE *fpart;
    char line[128];
    MPI_Comm_rank(MPI_COMM_WORLD, &mype);
    /*           {
     *               volatile idx_t tt = 0;
     *               printf("PID %d on %d ready for attach\n",mype,  getpid());
     *               fflush(stdout);
     *               while (0 == tt)
     *                   sleep(5);
     *           }
     * 
     */
    if(mype == 0)
        fpart = fopen(filename, "r");

    for(i = 0; i < nmodes; i++)
    {
        if(mype  == 0)
            for(j = 0; j < npes; j++){
                fgets(line, 128, fpart);
                sscanf(line, "%zu\n", &imap_arr[i][j]);
            }
        MPI_Bcast(imap_arr[i], npes, MPI_IDX_T, 0, MPI_COMM_WORLD); 
    }
    if(mype == 0)
        fclose(fpart);
}

idx_t read_fg_tensor(char tensorfile[], char partfile[], struct tensor *t, struct genst *gs ,  idx_t endian)
{


#ifdef NA_DBG
    na_log(dbgfp, "dbg read_fg_tensor:\n");
#endif
    read_fg_partition(partfile, gs);
#ifdef NA_DBG
    na_log(dbgfp, "\tafter read_fg_partition\n");
#endif
    t->nmodes = gs->nmodes; 
    if(endian)
        read_ckbd_tensor_nonzeros_endian(tensorfile, t, gs);
    else{
#if gsize == 64
        read_ckbd_tensor_nonzeros_large(tensorfile, t, gs);
#elif gsize == 32
        read_ckbd_tensor_nonzeros(tensorfile, t, gs);
#endif
    }

#ifdef NA_DBG
    na_log(dbgfp, "\tafter read_ckbd_tensor_nonzeros\n");
#endif
    gs->gnnz = t->gnnz;
    gs->nnz = t->nnz;
}
