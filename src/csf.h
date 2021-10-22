#ifndef TP_CSF_H
#define TP_CSF_H

#include "tensor.h"
#include "genst.h"
struct csfsparsity
{
    idx_t *nfibs;
    idx_t **fptr;
    idx_t **fids;
    real_t *vals;
};

struct csftensor
{
    idx_t nnz;
    idx_t nmodes;
    idx_t *dims;
    idx_t *dim_perm;
    idx_t *dim_iperm;
    idx_t ntiles;
    idx_t ntiled_modes;
    idx_t *tile_dims;

    struct csfsparsity *pt;

};

static inline idx_t csf_mode_to_depth(struct csftensor *csftns, idx_t mode)
{
    return csftns->dim_iperm[mode];
}

static inline idx_t csf_depth_to_mode(struct csftensor *csftns, idx_t lvl)
{
    return csftns->dim_perm[lvl];
}

void csf_get_sparsity_order(idx_t *gdims, idx_t *order, idx_t nmodes);

struct csftensor * csf_alloc( struct genst *gs, struct tensor *tns );

void p_mk_csf(struct genst *gs, struct tensor *tns, struct csftensor *csftns, idx_t mode);

void p_csf_alloc_untiled(struct genst *gs, struct tensor *tns, struct csftensor *csftns);

void p_mk_fptr(struct tensor *tns, struct csftensor *csftns, idx_t tile_id, idx_t * nnztile_ptr, idx_t mode);

void p_mk_outerptr(struct tensor *tns, struct csftensor *csftns, idx_t tile_id, idx_t *nnztile_ptr);

void free_csf(struct csftensor *csftns);


#endif
