#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"

struct csr_matrix {
  int rows, cols, nnz;
  int *row_ptr, *col_idx;
  double *val;
};

enum MatrixType {
  COORDINATE, ARRAY
};

enum MatrixFormat {
  GENERAL, SYMMETRIC
};

enum MatrixDataType {
  INT, REAL, COMPLEX, PATTERN
};

int read_csr_matrix(char *filename, struct csr_matrix *matrix) {
  MM_typecode matcode;
  int ret_code;

  FILE *f = fopen(filename, "r");
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

  enum MatrixType type;
  if (mm_is_coordinate(matcode)) {
    type = COORDINATE;
  } else {
    type = ARRAY;
  }

  enum MatrixFormat format;
  if (mm_is_symmetric(matcode)) {
    format = SYMMETRIC;
  } else {
    format = GENERAL;
  }

  enum MatrixDataType data_type;
  if (mm_is_pattern(matcode)) {
    data_type = PATTERN;
  } else if (mm_is_real(matcode)) {
    data_type = REAL;
  } else if (mm_is_complex(matcode)) {
    data_type = COMPLEX;
  } else if (mm_is_integer(matcode)) {
    data_type = INT;
  }

  int row, col;
  double val;
  int *row_counts = (int *) calloc(rows, sizeof(int));

  for (i = 0; i < nnz; i++) {
    if (type == COORDINATE) {
      if (fscanf(f, "%d %d", &row, &col) != 2) {
        fclose(f);
        return -1;
      }
      row--;
      col--;
      if (format == SYMMETRIC) {
        if (row > col) {
          int tmp = row;
          row = col;
          col = tmp;
        }
      }
    }

    if (data_type == PATTERN) {
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
    if (type == COORDINATE) {
      if (fscanf(f, "%d %d", &row, &col) != 2) {
        fclose(f);
        return -1;
      }
      row--;
      col--;
      if (format == SYMMETRIC) {
        if (row > col) {
          int tmp = row;
          row = col;
          col = tmp;
        }
      }
    }

    if (data_type == PATTERN) {
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