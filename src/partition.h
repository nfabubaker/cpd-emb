/**
 * @author      : Nabil Abubaker (nabil.abubaker@bilkent.edu.tr)
 * @file        : partition
 * @created     : Friday Oct 29, 2021 16:27:55 +03
 */

#ifndef PARTITION_H

#define PARTITION_H

#include "genst.h"
#include "tensor.h"
#include "basic.h"

void partition_rows_rand(const tensor *t, const genst *gs, const idx_t mode, idx_t *interpart);

#endif /* end of include guard PARTITION_H */
