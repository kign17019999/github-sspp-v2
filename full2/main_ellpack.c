#include <stdio.h>
#include "read_ellpack.h"

int main(int argc, char *argv[]) {
  char* matrix_file = "matrices/cage4.mtx"; // set default file name
  if (argc > 2) {
    printf("Usage: main <matrix_file>\n");
    return -1;
  } else if (argc == 2) {
    matrix_file = argv[1];
  }

  struct ellpack_matrix matrix;
  int ret_code = read_ellpack_matrix(matrix_file, &matrix);
  if (ret_code != 0) {
    printf("Failed to read matrix file\n");
    return ret_code;
  }

  printf("Matrix dimensions M x N: %d x %d\n", matrix.rows, matrix.cols);
  printf("Number of non-zero elements: %d\n", matrix.nnz);
  printf("ELLPACK representation:\n");
  printf("max_nnz_per_row: %d\n", matrix.max_nnz_per_row);
  printf("col_idx (IA): ");
  int i, j;
  for (i = 0; i < matrix.rows; i++) {
    for (j = 0; j < matrix.max_nnz_per_row; j++) {
      printf("%d ", matrix.col_idx[i * matrix.max_nnz_per_row + j]);
    }
    printf("\n");
  }
  printf("val (A): ");
  for (i = 0; i < matrix.rows; i++) {
    for (j = 0; j < matrix.max_nnz_per_row; j++) {
      printf("%.3lf ", matrix.val[i * matrix.max_nnz_per_row + j]);
    }
    printf("\n");
  }

  free(matrix.col_idx);
  free(matrix.val);
  return 0;
}
