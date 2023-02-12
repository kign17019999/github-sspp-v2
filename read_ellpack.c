#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"
#include "read_ellpack.h"

// struct ellpack_matrix {
//   int rows, cols, nnz, max_nnz_per_row;
//   int *col_idx;
//   double *val;
// };

int read_ellpack_matrix(const char *file_name, struct ellpack_matrix *matrix) {
  MM_typecode matcode;
  int ret_code;

  FILE *f = fopen(file_name, "r");
  if (!f) {
    return -1;
  }

  if (mm_read_banner(f, &matcode) != 0) {
    fclose(f);
    return -1;
  }

  int rows, cols, nnz;
  if ((ret_code = mm_read_mtx_crd_size(f, &rows, &cols, &nnz)) != 0) {
    fclose(f);
    return ret_code;
  }

  int i, j;
  matrix->rows = rows;
  matrix->cols = cols;
  matrix->nnz = nnz;

  int row, col;
  double val;
  int *row_counts = (int *) calloc(rows, sizeof(int));

  for (i = 0; i < nnz; i++) {
    if (fscanf(f, "%d %d %lf", &row, &col, &val) != 3) {
      fclose(f);
      return -1;
    }
    row--;
    col--;
    row_counts[row]++;
  }

  matrix->max_nnz_per_row = 0;
  for (i = 0; i < rows; i++) {
    if (row_counts[i] > matrix->max_nnz_per_row) {
      matrix->max_nnz_per_row = row_counts[i];
    }
  }

  matrix->col_idx = (int *) malloc(rows * matrix->max_nnz_per_row * sizeof(int));
  matrix->val = (double *) malloc(rows * matrix->max_nnz_per_row * sizeof(double));

  for (i = 0; i < rows; i++) {
    for (j = 0; j < matrix->max_nnz_per_row; j++) {
      matrix->col_idx[i * matrix->max_nnz_per_row + j] = -1;
    }
  }

  rewind(f);

  if (mm_read_banner(f, &matcode) != 0) {
    fclose(f);
    return -1;
  }

  if ((ret_code = mm_read_mtx_crd_size(f, &rows, &cols, &nnz)) != 0) {
    fclose(f);
    return ret_code;
  }

  for (i = 0; i < nnz; i++) {
    if (fscanf(f, "%d %d %lf", &row, &col, &val) != 3) {
      fclose(f);
      return -1;
    }
    row--;
    col--;

    int index = row * matrix->max_nnz_per_row;
    for (j = 0; j < matrix->max_nnz_per_row; j++) {
      if (matrix->col_idx[index + j] == -1) {
        matrix->col_idx[index + j] = col;
        matrix->val[index + j] = val;
        break;
      }
    }
  }

  fclose(f);
  free(row_counts);
  return 0;
}

