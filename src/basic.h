/**
 * @author      : Nabil Abubaker (nabil.abubaker@bilkent.edu.tr)
 * @file        : basic
 * @created     : Tuesday Jun 29, 2021 11:59:07 +03
 */
#include <inttypes.h>
#ifndef BASIC_H

#define BASIC_H
#define gsize 64
#if gsize == 32

#define real_t float
#define idx_t uint32_t
#define IDX_T_MAX UINT32_MAX 
#define MPI_REAL_T MPI_FLOAT
#define MPI_IDX_T MPI_UINT32_T

#elif gsize == 64

#define real_t double
#define idx_t uint64_t
#define IDX_T_MAX UINT64_MAX 
#define MPI_REAL_T MPI_DOUBLE
#define MPI_IDX_T MPI_UINT64_T

#endif
#endif /* end of include guard BASIC_H */

