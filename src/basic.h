/**
 * @author      : Nabil Abubaker (nabil.abubaker@bilkent.edu.tr)
 * @file        : basic
 * @created     : Tuesday Jun 29, 2021 11:59:07 +03
 */
#include <inttypes.h>
#ifndef BASIC_H

#define BASIC_H
#define idxsize 64 
#define valsize 32
#if idxsize == 32

#define idx_t uint32_t
#define IDX_T_MAX UINT32_MAX 
#define MPI_IDX_T MPI_UINT32_T
typedef int inpf_t;
#define MPI_INPF_T MPI_INT

#elif idxsize == 64

#define idx_t uint64_t
#define IDX_T_MAX UINT64_MAX 
#define MPI_IDX_T MPI_UINT64_T
typedef int inpf_t;
#define MPI_INPF_T MPI_INT
#endif

#if valsize == 32
#define real_t float
#define MPI_REAL_T MPI_FLOAT
#elif valsize == 64
#define MPI_REAL_T MPI_DOUBLE
#define real_t double
#endif

#endif /* end of include guard BASIC_H */

