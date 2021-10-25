#include "util.h"
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <mpi.h>
/*void *myMalloc(size_t size)
  {
  void *p = malloc(size);
  if(p == NULL)
  {
  printf("malloc couldn't allocate %d byte memory\n", size);
  exit(1);
  }
  else
  return p;

  }*/

void na_log(FILE *fp, const char *format, ...) {
    va_list args;
    va_start(args, format);
    vfprintf(fp, format, args);
    fflush(fp);
    va_end(args);
}
void *myMalloc(size_t size){
    return aligned_alloc(8*64, size);
}

void setintzero(idx_t *arr, idx_t size)
{
    idx_t i;
    for(i = 0; i < size; i++)
        arr[i] = 0;
}

/* NA: Just testing things for setreal_tzero, source: https://bytes.com/topic/c/answers/222353-safe-zero-float-array-memset */

static idx_t check_real_t_zero(void){
    real_t d = 0.0;
    unsigned char zero_bytes[sizeof d] = {0};
    return memcmp(&d, zero_bytes, sizeof d);
}
void setreal_tzero_try(real_t *arr, idx_t size)
{
    static idx_t initialized = 0;
    static idx_t mayusememset;

    if(!initialized){
        mayusememset = check_real_t_zero();
        initialized = 1;
        /* NA spoiler alert: it does't work on oceanic:) */
        /*         printf("will use memset ? %d\n", (mayusememset)? 1:0);
        */
    }
    if(mayusememset)
        memset(arr,0,size*sizeof(real_t));
    else{
        idx_t i;
        for(i = 0; i < size; i++)
            arr[i] = 0.0;
    }
}

void setreal_tzero(real_t *arr, idx_t size){
    real_t *first = arr, *last = arr+size;
    for(; first!=last; ++first)
        *first = 0.0;
}

void start_timer(tmr_t *t){                                                                                                                                                               
    clock_gettime(CLOCK_MONOTONIC, &t->ts_beg);                                                                                                                                           
    return;                                                                                                                                                                               
}                                                                                                                                                                                         
void stop_timer(tmr_t *t){                                                                                                                                                                
    clock_gettime(CLOCK_MONOTONIC, &t->ts_end);                                                                                                                                           
    t->elapsed += (1000000000.0 * (double) (t->ts_end.tv_sec - t->ts_beg.tv_sec) + (double) (t->ts_end.tv_nsec - t->ts_beg.tv_nsec));                                                       
    return;                                                                                                                                                                               
}                


/* long get_wc_time ( void )
 * {
 *     static struct timeval twclk ;
 *     gettimeofday(&twclk, NULL) ;
 *     return(twclk.tv_sec*1000000 + twclk.tv_usec) ;
 * }
 */

idx_t convert(idx_t value)
{
    char arr[4];
    arr[0] = (char)(value >> 24);
    arr[1] = (char)(value >> 16);
    arr[2] = (char)(value >> 8);
    arr[3] = (char)(value);

    return arr[3] << 24 | (arr[2] & 0xFF) << 16 | (arr[1] & 0xFF) << 8 | (arr[0] & 0xFF);

}

void substring(char *text, char out[1024])
{
    char *ptr = text;
    char *prevptr = NULL;

    while( (ptr = strstr(ptr,"/")))
    {
        prevptr = ptr++;
    }
    prevptr++;


    idx_t sl = strlen(prevptr)-6;
    strncpy(out, prevptr, sl);
    out[sl] = '\0';

}


void substring_b(char *dst, char *src){
    char *ptr = src;
    char *prevptr = NULL;

    while( (ptr = strstr(ptr, "_"))){
        prevptr = ptr++;
    }

    //prevptr;
    idx_t sl = strlen(src) - strlen(prevptr);
    strncpy(dst, src, sl);
    dst[sl] = '\0';
}

void radixsort(idx_t *inds, real_t *vals, idx_t nnz, idx_t nmodes, idx_t *order, idx_t *dims)
{
#ifdef NA_DBG
        na_log(dbgfp, "\thello from radixsort\n");
#endif

    idx_t i, j, ptr, ptr2, *cnt, *tmpinds, size, dim;
    idx_t mode;
    real_t *tmpvals;

    size = ((idx_t) nmodes)*sizeof(idx_t);

    tmpinds = (idx_t *)malloc(nnz*( (idx_t) nmodes)*sizeof(*tmpinds));
    tmpvals = (real_t *)malloc(nnz*sizeof(real_t));

    for(i = 0; i < nmodes; i++)
    {
        mode = order[nmodes-1-i];
        dim = dims[mode];

#ifdef NA_DBG
        na_log(dbgfp, "\t\tbefore cnt alloc,  mode %zu\n", i);
/*         int mype;
 *          MPI_Comm_rank(MPI_COMM_WORLD, &mype);
 *           {
 *               volatile idx_t tt = 0;
 *               printf("PID %d on %d ready for attach\n", mype,  getpid());
 *               fflush(stdout);
 *               while (0 == tt)
 *                   sleep(5);
 *           }
 */

#endif
        cnt = (idx_t *)malloc((dim+1)*sizeof(*cnt));
        setintzero(cnt, dim+1);

#ifdef NA_DBG
        MPI_Barrier(MPI_COMM_WORLD);
        na_log(dbgfp, "\t\tafter cnt alloc,  mode %zu\n", i);
#endif
        ptr = mode;
        for(j = 0; j < nnz; j++)
        {
            cnt[inds[ptr]+1]++;
            ptr += nmodes;
        }

        dim++;

        for(j = 2; j < dim; j++)
            cnt[j] += cnt[j-1];

#ifdef NA_DBG
        na_log(dbgfp, "\t\tbefore inds and vals memcopy,  mode %zu\n", i);
#endif
        memcpy(tmpinds, inds, nnz*nmodes*sizeof(*tmpinds));
        memcpy(tmpvals, vals, nnz*sizeof(*tmpvals));
#ifdef NA_DBG
        na_log(dbgfp, "\t\tafter inds and vals memcopy,  mode %zu\n", i);
#endif

        ptr = mode;
        for(j = 0; j < nnz; j++)
        {	
            ptr2 = cnt[tmpinds[ptr]]++;
            memcpy(&inds[ptr2*nmodes], &tmpinds[j*nmodes], size);
            vals[ptr2] = tmpvals[j];
            ptr += nmodes;
        }
#ifdef NA_DBG
        na_log(dbgfp, "\t\tafter 2nd memcopy,  mode %zu\n", i);
#endif

        free(cnt);

    }

    free(tmpinds);
    free(tmpvals);

}

idx_t checksort(idx_t *dims, idx_t nnz, idx_t nmodes, idx_t *order)
{
    idx_t i, j, cont;
    for(i = 1; i < nnz; i++)
    {
        cont = 1;
        for(j = 0; cont == 1 && j < nmodes; j++)
            if(dims[i*nmodes+order[j]] > dims[(i-1)*nmodes+order[j]])
                cont = 0;
            else if(dims[i*nmodes+order[j]] == dims[(i-1)*nmodes+order[j]])
                cont = 1;
            else
            {
                cont = 0;
                printf("adasda\n");
            }
    }
}


