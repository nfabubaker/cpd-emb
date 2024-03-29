#ifndef TP_FT_H
#define TP_FT_H

#include "tensor.h"
#include "genst.h"
struct fibertensor
{
  idx_t lmode;

  idx_t *fibermode;  //the modes of the fibers; nmodes-1 same, one different
  idx_t **order;     //order of the modes from top to the bottom for modes
  idx_t *topmostcnt; //number of nonzero subtensors for the topmost mode
  idx_t ***xfibers;  //hierarchical poidx_ter structure, poidx_ting in the bottommost level either to lfibers or slfibers

  idx_t *lfibers;    // fibers of the longest mode
  idx_t *slfibers;    // fibers of the second longest mode

  real_t *lvals;   //values of the fibers of the longest mode
  real_t *slvals;  //values of the fibers of the second longest modep
};

void init_fibertensor(struct fibertensor *ft);

idx_t get_longest_fibers(const idx_t *inds, const real_t *vals, const idx_t nmodes, const idx_t longestmode, const idx_t nnz, struct fibertensor *ft);

idx_t get_secondlongest_fibers(const idx_t *inds, const real_t *vals, const idx_t nmodes, const idx_t secondlongestmode, const idx_t nnz, struct fibertensor *ft);

idx_t point_fibers(idx_t mype, const idx_t *inds, const idx_t *order, const idx_t nmodes, const idx_t nnz, const idx_t mode, struct fibertensor *ft, const idx_t longest);

void get_sparsity_order(const idx_t *gdims, idx_t *order, const idx_t nmodes);

idx_t get_fibertensor(genst *gs, const tensor *t, struct fibertensor *ft);

void free_fibertensor(struct fibertensor *ft, idx_t nmodes);

#endif


