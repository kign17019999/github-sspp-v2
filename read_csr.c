#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"
#include "read_csr.h"

int read_csr_matrix(const char *file_name, struct csr_matrix *matrix) {
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

  int i;
  matrix->rows = rows;
  matrix->cols = cols;
  matrix->nnz = nnz;

  matrix->row_ptr = (int *) malloc((rows + 1) * sizeof(int));
  matrix->col_idx = (int *) malloc(nnz * sizeof(int));
  matrix->val = (double *) malloc(nnz * sizeof(double));

  matrix->row_ptr[0] = 0;

  int *row_counts = (int *) calloc(rows, sizeof(int));

  int row, col;
  double val;
  for (i = 0; i < nnz; i++) {
    if (fscanf(f, "%d %d", &row, &col) != 2) {
      fclose(f);
      return -1;
    }
    row--;
    col--;

    if (mm_is_pattern(matcode)) {
      val = 1.0;
    } else {
      if (fscanf(f, "%lf", &val) != 1) {
        fclose(f);
        return -1;
      }
    }

    row_counts[row]++;
  }

  for (i = 0; i < rows; i++) {
    matrix->row_ptr[i + 1] = matrix->row_ptr[i] + row_counts[i];
  }

  for (i = 0; i < rows; i++) {
    row_counts[i] = 0;
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
    if (fscanf(f, "%d %d", &row, &col) != 2) {
      fclose(f);
      return -1;
    }
    row--;
    col--;

    if(mm_is_pattern(matcode)) {
      val = 1.0;
    } else {
      if (fscanf(f, "%lf", &val) != 1) {
        fclose(f);
        return -1;
      }
    }
    int index = matrix->row_ptr[row] + row_counts[row];
    matrix->col_idx[index] = col;
    matrix->val[index] = val;
    row_counts[row]++;
  }

  fclose(f);
  free(row_counts);
  return 0;
}
