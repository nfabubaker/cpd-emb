#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "genst.h"
#include "io.h"
#include "mpi.h"
#include "tensor.h"
#include "partition.h"
#include <string.h>
#include <assert.h>
#include <sys/types.h>

//assumes the first line of the tensor starts with # and denotes dimension info
idx_t read_dimensions(char* tfile,  struct genst *gs){

    FILE *tf = fopen(tfile, "r");
    char line[1024];
    fgets(line, 1024, tf);
    fclose(tf);

    idx_t nmodes = 0, ptr = 1, i;
    char * str;

    str = strtok (line, "\t ");
    while (str != NULL){
        nmodes++; 
        str = strtok (NULL, "\t ");
    }
    nmodes--;

    gs->gdims = malloc(nmodes * sizeof(*gs->gdims));

    for(i = 0; i < nmodes; i++ )
    {
        str = strtok (&line[ptr], "\t ");
        gs->gdims[i] = (idx_t) strtoull(str, NULL , 10);
        ptr += strlen(str) + 1;
        //gs->gdims[i] = strtoull(str, &str, 10);
    }

    //gs->gnnz = strtoull(str, &str, 10);

    str = strtok (&line[ptr], "\t ");
    gs->gnnz = (idx_t) strtoull(str, NULL, 10);

    gs->nmodes = nmodes;

    return 0;
}


/* idx_t read_dimensions_bin_endian(char* tfile, struct tensor *t){

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
    idx_t i, j, offset_all,  offset,  nnz, *inds, val, total ;
    inpf_t *tmp, buf[2];
    idx_t nmodes; 
    real_t *vals;
    char line[1024], *str;
    MPI_File fh;
    MPI_Status statsus;

    uint64_t of;

    nmodes = gs->nmodes;
    offset = gs->mype*2*sizeof(int64_t) + gs->mype * sizeof(int);
    offset_all = gs->npes*(2*sizeof(int64_t) + sizeof(int));

    //MPI_Barrier(MPI_COMM_WORLD);
    //printf("b4 mpi read \n");
#ifdef NA_DBG
    na_log(dbgfp, "\thello from read_ckbd_tensor_nonzeros: before MPI_File_open\n");
#endif
    int tnmodes;
    MPI_File_open(MPI_COMM_WORLD, tensorfile, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_File_read_at(fh, offset, &tnmodes, 1, MPI_INT, &statsus);
    MPI_File_read_at(fh, offset+sizeof(int), buf, 2, MPI_INT64_T, &statsus);
    nmodes = (idx_t) tnmodes;
#ifdef NA_DBG
    na_log(dbgfp, "\t\tafter MPI_File_read_at\n");
#endif
    nnz = (idx_t) buf[0];
    of = (uint64_t) buf[1]*(nmodes+1)*sizeof(inpf_t) + offset_all;
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
    idx_t i, j, offset,  nnz, *inds, val, total ;
    inpf_t *tmp, *buf;
    idx_t nmodes; 
    real_t *vals;
    char line[1024], *str;
    MPI_File fh;
    MPI_Status statsus;

    uint64_t of;

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
    of = (uint64_t) buf[2]*(nmodes+1)*sizeof(inpf_t)+gs->npes*3*sizeof(inpf_t);
    t->inds =  malloc(nmodes*nnz*sizeof(*t->inds));
    t->vals =  malloc(nnz*sizeof(*t->vals));
    t->nnz = nnz;
    idx_t tmpsize = (nmodes+1)*nnz;
    tmp =  malloc(sizeof(*tmp) * tmpsize);
    //tmp = (idx_t *)malloc((nmodes+1)*nnz*sizeof(int));

#ifdef NA_DBG
    MPI_Barrier(MPI_COMM_WORLD);
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

void random_interpart(tensor *t, genst *gs){

    idx_t i, j, k, dsize, slices_per_p, slices_per_pR;
    gs->interpart = malloc(gs->nmodes * sizeof(*gs->interpart));
    for (i = 0; i < gs->nmodes; ++i) {
        dsize = gs->gdims[i];
        gs->interpart[i] = malloc(dsize * sizeof(*gs->interpart[i]));
        if (gs->rows_assignment == 1) {
#ifdef NA_DBG
            MPI_Barrier(MPI_COMM_WORLD);
            na_log(dbgfp, "\tattempting random-respect-comm assignment\n");
#endif
            partition_rows_rand(t, gs, i, gs->interpart[i]);
#ifdef NA_DBG
            MPI_Barrier(MPI_COMM_WORLD);
            na_log(dbgfp, "\t random-respect-comm done\n");
#endif
        }
        else {
            k = 0;
            for (j = 0; j < dsize; ++j) {
                gs->interpart[i][j] = ( k++ % gs->npes );
            }
        }

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

tensor * alloc_tensor_coo(const genst *gs, const idx_t nnz)
{
    tensor *tt = malloc(sizeof(*tt)); 
    init_tensor(tt);
    tt->gnnz = gs->gnnz;
    tt->nnz = nnz;
    tt->nmodes = gs->nmodes;
    tt->inds = malloc(sizeof(*tt->inds) * gs->nmodes * nnz);
    tt->vals = malloc(sizeof(*tt->vals) * nnz);

    return tt;

}

void read_tensor_coo(FILE *fin, tensor *const t, const genst *gs, idx_t const nnz_to_read)
{
    idx_t i;
    char *line = NULL, *ptr = NULL;
    ssize_t read;
    size_t len = 0;
    idx_t nnzC = 0;
    while(nnzC < nnz_to_read && (read = getline(&line, &len, fin)) != -1 )
    {
        if(read > 1 && line[0] != '#'){
            ptr = line;
            for (i = 0; i < gs->nmodes; ++i) {
                t->inds[nnzC*gs->nmodes+i] = (idx_t) strtoull(ptr, &ptr, 10) - 1; 
            }
            t->vals[nnzC++] = (real_t) strtod(ptr, &ptr);
        }
    }


    /*     idx_t idx ;
     *     while(nnzC < nnz_to_read && (read = getline(&line, &len, fin)) != -1 )
     *     {
     *         if(read > 1 && line[0] != '#'){
     *             ptr = line;
     *             idx = 0;
     *             for(i = 0; i < gs->nmodes; i++ )
     *             {
     *                 ptr = strtok (&line[idx], "\t");
     *                 t->inds[nnzC*gs->nmodes + i] = (idx_t) strtoull(ptr, NULL , 10);
     *                 idx += strlen(ptr) + 1;
     *             }
     *           ptr = strtok (&line[idx], "\t");
     *           t->vals[nnzC++] = (real_t) strtod(ptr, NULL);
     *         }
     *     }
     */

}

tensor * read_tensor_txt(char tensorfile[], struct genst *gs ){
    int i, npes;
    idx_t lnnzC, lnnzR;
    tensor *tt, *t;
    FILE *fin;
#ifdef NOF_DBG
    fprintf(stderr, "dbg : hello from read_tensor_txt\n");
#endif
    if(gs->mype == 0){
        read_dimensions(tensorfile, gs);
#ifdef NOF_DBG
        fprintf(stderr, "\tdbg : after read_dimensions \n");
#endif
    }
    /*     printf("ndims = %zu nnz=%zu\n", gs->nmodes, gs->gnnz);
     *     for (i = 0; i < gs->nmodes; ++i) {
     *         printf("mode %zu = %zu\n", i, gs->gdims[i]);
     *     }
     */

    MPI_Bcast(&gs->nmodes, 1, MPI_IDX_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&gs->gnnz, 1, MPI_IDX_T, 0, MPI_COMM_WORLD);
    if(gs->mype != 0)
        gs->gdims = malloc(gs->nmodes * sizeof(*gs->gdims));
    MPI_Bcast(gs->gdims, gs->nmodes, MPI_IDX_T, 0, MPI_COMM_WORLD);

#ifdef NOF_DBG
    fprintf(stderr, "\tdbg : after initial Bcasts \n");
#endif
    npes = (int) gs->npes;
    lnnzC = gs->gnnz / gs->npes;
    lnnzR = gs->gnnz % gs->npes;
    if (gs->mype == 0) {
        fin = fopen(tensorfile, "r");
        idx_t mynnz = (lnnzR == 0 ? lnnzC : lnnzC+1);
        t = alloc_tensor_coo(gs, mynnz);
        tt = alloc_tensor_coo(gs, mynnz);
        /* read my nonzeros */
        read_tensor_coo(fin, t, gs, mynnz);
#ifdef NOF_DBG
        fprintf(stderr, "\tdbg : read_tensor_coo for p0 done\n");
#endif
        /* now read for others */
        for(i = 1; i < lnnzR  ; ++i) {
            /* read lnnzC+1 nonzeros and send them to processor i+1 */
            read_tensor_coo(fin, tt, gs, lnnzC+1);
#ifdef NOF_DBG
            fprintf(stderr, "\tdbg : [+1] read_tensor_coo for p%d done \n", (i) );
#endif
            MPI_Send(tt->inds, gs->nmodes*(lnnzC+1), MPI_IDX_T, (i), 11+i, MPI_COMM_WORLD);
            MPI_Send(tt->vals, (lnnzC+1), MPI_REAL_T, (i), npes + 11+i, MPI_COMM_WORLD);
#ifdef NOF_DBG
            fprintf(stderr, "\tdbg : [+1] tensor_coo for p%d sent\n", (i) );
#endif
        }
        int ssidx = (lnnzR == 0 ? 1 : (int)lnnzR);
        for(i = ssidx; i < npes; ++i) {
            /* read lnnzC+1 nonzeros and send them to processor i+1 */
            read_tensor_coo(fin, tt, gs, lnnzC);
#ifdef NOF_DBG
            fprintf(stderr, "\tdbg : [r] read_tensor_coo for p%d done \n", (i) );
#endif
            MPI_Send(tt->inds, gs->nmodes*(lnnzC), MPI_IDX_T, (i), 11+i, MPI_COMM_WORLD);
            MPI_Send(tt->vals, (lnnzC), MPI_REAL_T, (i), npes + 11 + i, MPI_COMM_WORLD);
#ifdef NOF_DBG
            fprintf(stderr, "\tdbg : [r] read_tensor_coo for p%d done and sent\n", (i) );
#endif
        }
        free(tt->inds);
        free(tt->vals);
        fclose(fin);
    }
    else{
        idx_t mynnz = (gs->mype < lnnzR ? lnnzC+1 : lnnzC);
        t = alloc_tensor_coo(gs, mynnz);
        MPI_Recv(t->inds, gs->nmodes * mynnz, MPI_IDX_T, 0, 11+gs->mype, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(t->vals, mynnz, MPI_REAL_T, 0, npes +11+gs->mype, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#ifdef NOF_DBG
        fprintf(stderr, "\tdbg : p%d all recvd\n", (int)gs->mype);
#endif
    }

    return t;
}

tensor * read_fg_tensor(char tensorfile[], char partfile[],  struct genst *gs ,  idx_t endian)
{

    tensor *t;

#ifdef NA_DBG
    na_log(dbgfp, "dbg read_fg_tensor:\n");
#endif
    if (gs->use_pfile == 1) {
        t = (struct tensor *) malloc(sizeof(*t));
        init_tensor(t);
        read_fg_partition(partfile, gs);
#ifdef NA_DBG
        na_log(dbgfp, "\tafter read_fg_partition\n");
#endif
        t->nmodes = gs->nmodes; 
        if(endian)
            read_ckbd_tensor_nonzeros_endian(tensorfile, t, gs);
        else{
#if idxsize == 64
            read_ckbd_tensor_nonzeros_large(tensorfile, t, gs);
#elif idxsize == 32
            read_ckbd_tensor_nonzeros(tensorfile, t, gs);
#endif
        }

#ifdef NA_DBG
        na_log(dbgfp, "\tafter read_ckbd_tensor_nonzeros\n");
#endif
        gs->gnnz = t->gnnz;
    }
    else{
        t = read_tensor_txt(tensorfile, gs);
#ifdef NA_DBG
        idx_t i, j, idx;
        for (i = 0; i < t->nnz; ++i) {
            for (j = 0; j < gs->nmodes; ++j) {
                idx = t->inds[i * gs->nmodes + j];
                if(idx >= gs->gdims[j] || idx < 0){
                    fprintf(stderr, "idx %zu is greater than dim %zu of mode %zu nnz %zu of p%zu\n of %zu nonzeros", idx, gs->gdims[j], j, i, gs->mype, t->nnz);
                }
                assert(idx < gs->gdims[j] && idx >= 0 && "error: idx is greater than dim");  
            }
        }
        idx_t tgnnz;
        MPI_Allreduce(&t->nnz, &tgnnz, 1, MPI_IDX_T, MPI_SUM, MPI_COMM_WORLD);

        na_log(dbgfp, "\tactual gnnz = %zu calc tgnnz = %zu\n", t->gnnz, tgnnz);
        assert(tgnnz == t->gnnz);
#endif
        random_interpart(t, gs);
    }
    assert(gs->gnnz == t->gnnz);
    gs->nnz = t->nnz;
    return t;
}
