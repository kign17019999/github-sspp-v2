#ifndef READ_ELLPACK_H
#define READ_ELLPACK_H

#include <stdio.h>
#include <stdlib.h>

struct ellpack_matrix {
  int rows;
  int cols;
  int nnz;
  int max_nnz_per_row;
  int *col_idx;
  double *val;
};

int read_ellpack_matrix(const char *file_name, struct ellpack_matrix *matrix);

#endif
