#ifndef TP_MTTKRP_H
#define TP_MTTKRP_H

#include "tensor.h"
#include "fibertensor.h"
#include "csf.h"
#include "genst.h"
void mttkrp_nnz(struct genst *gs, struct tensor *t, idx_t mode, real_t *matm);

void mttkrp_fiber_3(struct genst *gs, struct fibertensor *ft, idx_t mode, real_t *matm);

void mttkrp_fiber_4(struct genst *gs, struct fibertensor *ft, idx_t mode, real_t *matm);

real_t mttkrp_nnz_stats(struct genst *gs, struct tensor *t, idx_t mode, real_t *matm, idx_t niters);

idx_t  mttkrp_fiber_3_stats(struct genst *gs, struct fibertensor *ft, idx_t mode, real_t *matm);

idx_t mttkrp_fiber_4_stats(struct genst *gs, struct fibertensor *ft, idx_t mode, real_t *matm);

void mttkrp(struct genst *gs, struct tensor *t, struct fibertensor *ft, struct csftensor *csftns,  idx_t mode, real_t *matm);

idx_t mttkrp_stats(struct genst *gs, struct tensor *t, struct fibertensor *ft, struct csftensor *csftns, idx_t mode, real_t *matm);

void mttkrp_csf(struct genst *gs, struct csftensor *ft, idx_t mode, real_t *matm);

void mttkrp_csf_root(struct genst *gs, struct csftensor *ft, idx_t mode, real_t *matm);

void mttkrp_csf_int(struct genst *gs, struct csftensor *ft, idx_t mode, real_t *matm);

void mttkrp_csf_leaf(struct genst *gs, struct csftensor *ft, idx_t mode, real_t *matm);

void mttkrp_csf_root_3m(struct genst *gs, struct csftensor *ft, real_t *matm);

void mttkrp_csf_int_3m(struct genst *gs, struct csftensor *ft, real_t *matm);

void mttkrp_csf_leaf_3m(struct genst *gs, struct csftensor *ft, real_t *matm);

void mttkrp_csf_root_4m(struct genst *gs, struct csftensor *ft, real_t *matm);

void mttkrp_csf_int_4m(struct genst *gs, struct csftensor *ft, real_t *matm, idx_t mode);

void mttkrp_csf_leaf_4m(struct genst *gs, struct csftensor *ft, real_t *matm);


#endif
