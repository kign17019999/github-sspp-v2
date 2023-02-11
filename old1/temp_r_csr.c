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
    printf("here fopen \n");
    return -1;
  }

  if (mm_read_banner(f, &matcode) != 0) {
    fclose(f);
    printf("here read banner \n");
    return -1;
  }

  int rows, cols, nnz;
  if ((ret_code = mm_read_mtx_crd_size(f, &rows, &cols, &nnz)) != 0) {
    fclose(f);
    return ret_code;
  }
  ///////////
  printf("rows=%d, cols=%d, nnz=%d \n", rows, cols, nnz);
  ///////////

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
        if (row != col) {
          col = row;
        }
      }
    } else if (type == ARRAY) {
      if (fscanf(f, "%d", &col) != 1) {
        fclose(f);
        return -1;
      }
      col--;
      row = curr_row;
      curr_row++;
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
