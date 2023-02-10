#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"

struct csr_matrix {
  int rows;
  int cols;
  int nnz;
  int *row_ptr;
  int *col_idx;
  double *val;
};

enum MatrixType {
  COORDINATE,
  ARRAY
};

enum MatrixFormat {
  GENERAL,
  SYMMETRIC
};

enum MatrixDataType {
  INT,
  REAL,
  COMPLEX,
  PATTERN
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

  int i, curr_row = 1;
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
  for (i = 0; i < nnz; i++) {
    if (type == COORDINATE) {
      if (fscanf(f, "%d %d", &row, &col) != 2) {
        fclose(f);
        return -1;
      }
      row--;
      col--;
      if (format == SYMMETRIC) {
        row = col;
      } else {
        if (fscanf(f, "%d", &row) != 1) {
          fclose(f);
          return -1;
        }
        row--;
      }
    }

    matrix->col_idx[i] = col;

    if (data_type == PATTERN) {
      val = 1.0;
    } else {
      if (fscanf(f, "%lf", &val) != 1) {
        fclose(f);
        return -1;
      }
    }
    matrix->val[i] = val;

    if (row != curr_row - 1) {
      int j;
      for (j = curr_row; j <= row; j++) {
        matrix->row_ptr[j] = i;
      }
      curr_row = row + 1;
    }
  }

  matrix->row_ptr[rows] = nnz;

  fclose(f);
  return 0;
}

#ifdef READ_CSR_MAIN
int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: read_csr <matrix_file>\n");
    return -1;
  }

  struct csr_matrix matrix;
  int ret_code = read_csr_matrix(argv[1], &matrix);
  if (ret_code != 0) {
    printf("Failed to read matrix file\n");
    return ret_code;
  }

  printf("Matrix dimensions: %d x %d\n", matrix.rows, matrix.cols);
  printf("Number of non-zero elements: %d\n", matrix.nnz);
  printf("CSR representation:\n");
  printf("row_ptr: ");
  int i;
  for (i = 0; i < matrix.rows + 1; i++) {
    printf("%d ", matrix.row_ptr[i]);
  }
  printf("\ncol_idx: ");
  for (i = 0; i < matrix.nnz; i++) {
    printf("%d ", matrix.col_idx[i]);
  }
  printf("\nval: ");
  for (i = 0; i < matrix.nnz; i++) {
    printf("%.1lf ", matrix.val[i]);
  }
  printf("\n");

  free(matrix.row_ptr);
  free(matrix.col_idx);
  free(matrix.val);
  return 0;
}
#endif
