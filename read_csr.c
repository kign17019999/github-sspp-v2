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

  int M, N, NNZ;
  if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &NNZ)) != 0) {
    fclose(f);
    return ret_code;
  }

  matrix->M = M;
  matrix->N = N;
  matrix->NNZ = NNZ;

  matrix->IRP = (int *) malloc((M + 1) * sizeof(int));
  matrix->JA = (int *) malloc(NNZ * sizeof(int));
  matrix->AZ = (double *) malloc(NNZ * sizeof(double));

  int i;
  for (i = 0; i <= M; i++) {
    matrix->IRP[i] = 0;
  }

  int row_counts = 0;
  int last_row = -1;
  int row, col;
  double AZ;
  for (i = 0; i < NNZ; i++) {
    if (fscanf(f, "%d %d", &row, &col) != 2) {
      fclose(f);
      return -1;
    }
    row--;
    col--;

    if (row != last_row) {
      for (; last_row < row; last_row++) {
        matrix->IRP[last_row + 1] = row_counts;
      }
    }

    if (mm_is_pattern(matcode)) {
      AZ = 1.0;
    } else {
      if (fscanf(f, "%lf", &AZ) != 1) {
        fclose(f);
        return -1;
      }
    }

    matrix->JA[row_counts] = col;
    matrix->AZ[row_counts] = AZ;
    row_counts++;
  }

  for (; last_row < M; last_row++) {
    matrix->IRP[last_row + 1] = row_counts;
  }

  fclose(f);
  return 0;
}

