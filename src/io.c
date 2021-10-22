#include <stdlib.h>
#include <stdio.h>
#include "io.h"

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

idx_t read_ckbd_tensor_nonzeros(char tensorfile[], struct tensor *t, struct genst *gs)
{
    idx_t i, j, offset,  nmodes, nnz, *inds, val, total, *buf, *tmp;
    real_t *vals;
    char line[1024], *str;
    MPI_File fh;
    MPI_Status statsus;

    long idx_t of;

    nmodes = gs->nmodes;
    offset = gs->mype*3*sizeof(int);
    buf = (idx_t *)malloc((nmodes+1)*sizeof(int));

    //MPI_Barrier(MPI_COMM_WORLD);
    //printf("b4 mpi read \n");

    MPI_File_open(MPI_COMM_WORLD, tensorfile, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_File_read_at(fh, offset, buf, 3, MPI_IDX_T, &statsus);
    nnz = buf[1];
    of = buf[2]*(nmodes+1)*sizeof(int)+gs->npes*3*sizeof(int);
    t->inds = (idx_t *) malloc(nmodes*nnz*sizeof(int));
    t->vals = (real_t *) malloc(nnz*sizeof(real_t));
    t->nnz = nnz;
    idx_t tmpsize = (nmodes+1)*nnz;
    tmp = (idx_t *) malloc(sizeof(int) * tmpsize);
    //tmp = (idx_t *)malloc((nmodes+1)*nnz*sizeof(int));

    MPI_Allreduce(&nnz, &t->gnnz, 1, MPI_IDX_T, MPI_SUM, MPI_COMM_WORLD);
    inds = t->inds;
    vals = t->vals;

    MPI_File_read_at(fh, of, tmp, (nmodes+1)*nnz, MPI_IDX_T, &statsus);
    MPI_File_close(&fh); 

    for(i = 0; i < nnz; i++)
    {
        memcpy(&inds[i*nmodes], &tmp[i*(nmodes+1)], sizeof(int)*nmodes); 
        vals[i] = (real_t) tmp[i*(nmodes+1)+nmodes];
    }
    free(buf);

}

idx_t read_ckbd_tensor_nonzeros_endian(char tensorfile[], struct tensor *t, struct genst *gs)
{
    idx_t i, j, nmodes, nnz, *inds, val, *buff, total;
    real_t *vals;
    char line[1024], *str;
    FILE *ftensor;

    nmodes = gs->nmodes;
    buff = (idx_t *)malloc((nmodes+2)*sizeof(int));

    ftensor = fopen(tensorfile, "rb");    
    fread(buff, sizeof(int), nmodes+2, ftensor);
    for(j = 0; j < nmodes+2; j++)
        buff[j] = convert(buff[j]);

    t->gnnz = buff[nmodes+1];

    fread(&nnz, sizeof(int), 1, ftensor);
    nnz = convert(nnz);

    t->inds = (idx_t *)malloc(nmodes*nnz*sizeof(int));
    t->vals = (real_t *)malloc(nnz*sizeof(real_t));
    t->nnz = nnz;

    inds = t->inds;
    vals = t->vals;

    idx_t b = 0;
    for(i = 0; i < nnz; i++)
    {
        fread(&inds[b], sizeof(int), nmodes, ftensor);
        for(j = 0; j < nmodes; j++)
            inds[b+j] = convert(inds[b+j]);
        b += nmodes;

        fread(&val, sizeof(int), 1, ftensor);
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
    idx_t i, j, nmodes, npes, dim;
    FILE *fpart;
    char line[128];


    if (gs->mype == 0) {
        fpart = fopen(partfile, "r");
        fgets(line, 128, fpart);
        sscanf(line, "##%d %d\n", &nmodes, &npes);
        if(npes != gs->npes)
        {
            if(gs->mype == 0)
                printf("The partition file is for %d processors!!\n", npes);
            MPI_Finalize();
            exit(1);
        }
    }
    MPI_Bcast(&nmodes,  1, MPI_IDX_T, 0, MPI_COMM_WORLD);
    gs->nmodes = nmodes;
    gs->gdims = (idx_t *)malloc(nmodes*sizeof(int));

    gs->interpart = (idx_t **)malloc(sizeof(idx_t *)*nmodes);
    for(i = 0; i < nmodes; i++)
    {
        if (gs->mype == 0) {
            fgets(line, 128, fpart);
            sscanf(line, "#%d\n", &dim);
        }
        MPI_Bcast(&dim,  1, MPI_IDX_T, 0, MPI_COMM_WORLD);
        gs->gdims[i] = dim; 

        gs->interpart[i] = (idx_t *)malloc(sizeof(int)*dim);

        if (gs->mype == 0) {
            for(j = 0; j < dim; j++){
                fgets(line, 128, fpart);
                sscanf(line, "%d\n", &gs->interpart[i][j]);
            }
        }
        MPI_Bcast(gs->interpart[i],  dim, MPI_IDX_T, 0, MPI_COMM_WORLD);

    }

    if (gs->mype == 0) {
        fclose(fpart);
    }
}

void read_hc_imap(char filename[], idx_t nmodes, idx_t npes, idx_t **imap_arr)
{
    idx_t i, j, mype;
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
                sscanf(line, "%d\n", &imap_arr[i][j]);
            }
        MPI_Bcast(imap_arr[i], npes, MPI_IDX_T, 0, MPI_COMM_WORLD); 
    }
    if(mype == 0)
        fclose(fpart);
}

idx_t read_fg_tensor(char tensorfile[], char partfile[], struct tensor *t, struct genst *gs ,  idx_t endian)
{


    read_fg_partition(partfile, gs);
    t->nmodes = gs->nmodes; 
    if(endian)
        read_ckbd_tensor_nonzeros_endian(tensorfile, t, gs);
    else
        read_ckbd_tensor_nonzeros(tensorfile, t, gs);

    gs->gnnz = t->gnnz;
    gs->nnz = t->nnz;
}
