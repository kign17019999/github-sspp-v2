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

  printf("CSR representation:\n");
  printf("M: %d\nN: %d\n", matrix.M, matrix.N);
  printf("NNZ: %d\n", matrix.NNZ);
  printf("IRP: ");
  int i;
  for (i = 0; i < matrix.M + 1; i++) {
    printf("%d ", matrix.IRP[i]);
  }
  printf("\nJA: ");
  for (i = 0; i < matrix.NNZ; i++) {
    printf("%d ", matrix.JA[i]);
  }
  printf("\nAZ: ");
  for (i = 0; i < matrix.NNZ; i++) {
    printf("%.3lf ", matrix.AZ[i]);
  }
  printf("\n");

  free(matrix.IRP);
  free(matrix.JA);
  free(matrix.AZ);
  return 0;
}
