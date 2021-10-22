#ifndef TP_UTIL_H
#define TP_UTIL_H

#include <time.h>
#include <stdio.h>
#include "basic.h"
FILE *dbgfp;
char dbg_fn[1024];
void na_log(FILE *fp, const char* format, ...);
typedef struct _tmr_t{                                                                                                                                                                    
    struct timespec ts_beg;
    struct timespec ts_end;
    double elapsed;
} tmr_t; 

    void setintzero(idx_t *arr, idx_t size);
    void setreal_tzero(real_t *arr, idx_t size);
    long get_wc_time ( void );
    idx_t convert(idx_t value);
    void substring(char *text, char out[1024]);
    void substring_b(char *dst, char *src);
    void radixsort(idx_t *inds, real_t *vals, idx_t nnz, idx_t nmodes, idx_t *order, idx_t *dims);
    idx_t checksort(idx_t *dims, idx_t nnz, idx_t nmodes, idx_t *order);
    void stop_timer(tmr_t *t);
    void start_timer(tmr_t *t);

#endif
