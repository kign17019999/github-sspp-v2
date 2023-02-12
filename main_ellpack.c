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
  
  printf("ELLPACK representation:\n");
  printf("M: %d\nN: %d\n", matrix.M, matrix.N);
  printf("NNZ: %d\n", matrix.NNZ);
  printf("MAXNZ: %d\n", matrix.MAXNZ);
  printf("JA: ");
  int i, j;
  for (i = 0; i < matrix.M; i++) {
    for (j = 0; j < matrix.MAXNZ; j++) {
      printf("%d ", matrix.JA[i * matrix.MAXNZ + j]);
    }
    printf("\n");
  }
  printf("AZ: ");
  for (i = 0; i < matrix.M; i++) {
    for (j = 0; j < matrix.MAXNZ; j++) {
      printf("%.3lf ", matrix.AZ[i * matrix.MAXNZ + j]);
    }
    printf("\n");
  }

  free(matrix.JA);
  free(matrix.AZ);
  return 0;
}
