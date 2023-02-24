#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"
#include "read_ellpack.h"

int read_ellpack_matrix(const char *file_name, struct ellpack_matrix *matrix) {
  MM_typecode matcode;
  int ret_code, M, N, NNZ, row, col, *row_counts, i, j = 0;
  double AZ;

  FILE *f = fopen(file_name, "r");
  if (!f || mm_read_banner(f, &matcode) != 0 || (ret_code = mm_read_mtx_crd_size(f, &M, &N, &NNZ)) != 0) {
    if (f) fclose(f);
    return -1;
  }

  matrix->M = M;
  matrix->N = N;
  matrix->NNZ = NNZ;

  row_counts = (int *) calloc(M, sizeof(int));
  matrix->MAXNZ = 0;

  for (i = 0; i < NNZ && fscanf(f, "%d %d", &row, &col) == 2; i++) {
    row--, col--;

    if (!mm_is_pattern(matcode) && fscanf(f, "%lf", &AZ) != 1) {
      fclose(f);
      free(row_counts);
      return -1;
    }

    row_counts[row]++, matrix->MAXNZ = row_counts[row] > matrix->MAXNZ ? row_counts[row] : matrix->MAXNZ;
  }

  matrix->JA = (int *) malloc(M * matrix->MAXNZ * sizeof(int));
  matrix->AZ = (double *) malloc(M * matrix->MAXNZ * sizeof(double));
  for (i = 0; i < M * matrix->MAXNZ; i++) matrix->JA[i] = -1;

  rewind(f);
  if (mm_read_banner(f, &matcode) != 0 || (ret_code = mm_read_mtx_crd_size(f, &M, &N, &NNZ)) != 0) {
    fclose(f);
    free(row_counts);
    return -1;
  }

  for (i = 0; i < NNZ && fscanf(f, "%d %d", &row, &col) == 2; i++) {
    row--, col--;

    if (!mm_is_pattern(matcode) && fscanf(f, "%lf", &AZ) != 1) {
      fclose(f);
      free(row_counts);
      return -1;
    }

    while (matrix->JA[row * matrix->MAXNZ + j] != -1) j++;
    matrix->JA[row * matrix->MAXNZ + j] = col, matrix->AZ[row * matrix->MAXNZ + j] = AZ;
  }

  fclose(f);
  free(row_counts);
  return 0;
}
