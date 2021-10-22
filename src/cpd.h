#ifndef TP_CPD_H
#define TP_CPD_H

#include "genst.h"
#include "tensor.h"
#include "fibertensor.h"
#include "csf.h"


void compute_inverse(struct genst *t, idx_t mode, real_t *inverse);

void matrix_multiply(struct genst *t, idx_t mode, real_t *matm, real_t *inverse);
void normalize2(struct genst *t, idx_t mode, real_t *lambda);
void normalizemax(struct genst *t, idx_t mode, real_t *lambda);
void compute_aTa(struct genst *t, idx_t mode);

real_t compute_input_norm(real_t *vals, idx_t lnnz);

real_t compute_fit(struct genst *t, real_t *matm, real_t inputnorm, idx_t mode, real_t *lambda);

void cp_als_stats_fg(struct genst *gs, struct tensor *t, struct fibertensor *ft, struct csftensor *csftns, int niters, double *mttkrptime, double *comm1time, double * comm2time, idx_t *cnt_st);

void cp_als_fg(struct genst *gs, struct tensor *t, struct fibertensor *ft, struct csftensor *csftns, int niters, double *cptime);
void cp_als_fg_time(struct genst *gs, struct tensor *t, struct fibertensor *ft, struct csftensor *csftns, int niters, double *cptime, double *mmT, double *othersT , double *mttkrptime, double *comm1time, double * comm2time);
void cp_als_fg_emb(struct genst *gs, struct tensor *t, struct fibertensor *ft, struct csftensor *csftns, int niters, double *cptime);
void cp_als_fg_dbg(struct genst *gs, struct tensor *t, struct fibertensor *ft, struct csftensor *csftns, int niters, double *cptime);
void cp_als_fg_emb_time(struct genst *gs, struct tensor *t, struct fibertensor *ft, struct csftensor *csftns, int niters, double *cptime, double *mmT, double *othersT , double *mttkrptime, double *comm1time, double * comm2time);

//idx_t cp_als_barrier_fg(struct tensor *t, struct fibertensor *ft, int niters);

#endif
