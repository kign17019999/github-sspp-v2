#include <stdio.h>
#include "temp_r_csr.h"
#include <string.h>

//int main(int argc, char *argv[]) {
int main() {
  // if (argc != 2) {
  //   printf("Usage: main <matrix_file>\n");
  //   return -1;
  // }
  //char *file_name[256] = "/local_matrices/cage4.mtx";

  char *file_name[] = {"local_matrices/cage4.mtx", "king"};
  struct csr_matrix matrix;
  int ret_code = read_csr_matrix(file_name[0], &matrix);
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
