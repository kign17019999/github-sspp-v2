#ifndef READ_CSR_H
#define READ_CSR_H

#include <stdio.h>
#include <stdlib.h>

enum MatrixType {
  COORDINATE,
  ARRAY
};

enum MatrixFormat {
  GENERAL,
  SYMMETRIC
};

enum MatrixValueType {
  INT,
  REAL,
  COMPLEX,
  PATTERN
};

struct csr_matrix {
  int rows;
  int cols;
  int nnz;
  int *row_ptr;
  int *col_idx;
  double *val;
};

int read_csr_matrix(const char *file_name, struct csr_matrix *matrix);

#endif
