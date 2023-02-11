#include <stdio.h>
#include "read_csr.h"

int main(int argc, char *argv[]) {
  char* matrix_file = "matrices/cage4.mtx"; // set default file name
  if (argc > 2) {
    printf("Usage: main <matrix_file>\n");
    return -1;
  } else if (argc == 2) {
    matrix_file = argv[1];
  }

  struct csr_matrix matrix;
  int ret_code = read_csr_matrix(matrix_file, &matrix);
  if (ret_code != 0) {
    printf("Failed to read matrix file\n");
    return ret_code;
  }

  printf("Matrix dimensions M x N: %d x %d\n", matrix.rows, matrix.cols);
  printf("Number of non-zero elements: %d\n", matrix.nnz);
  printf("CSR representation:\n");
  printf("row_ptr (IRP): ");
  int i;
  for (i = 0; i < matrix.rows + 1; i++) {
    printf("%d ", matrix.row_ptr[i]);
  }
  printf("\ncol_idx (JA): ");
  for (i = 0; i < matrix.nnz; i++) {
    printf("%d ", matrix.col_idx[i]);
  }
  printf("\nval (AZ): ");
  for (i = 0; i < matrix.nnz; i++) {
    printf("%.3lf ", matrix.val[i]);
  }
  printf("\n");

  free(matrix.row_ptr);
  free(matrix.col_idx);
  free(matrix.val);
  return 0;
}