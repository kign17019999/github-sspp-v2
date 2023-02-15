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

  int i;
  matrix->M = M;
  matrix->N = N;
  matrix->NNZ = NNZ;

  matrix->IRP = (int *) malloc((M + 1) * sizeof(int));
  matrix->JA = (int *) malloc(NNZ * sizeof(int));
  matrix->AZ = (double *) malloc(NNZ * sizeof(double));

  matrix->IRP[0] = 0;

  int *row_counts = (int *) calloc(M, sizeof(int));

  int row, col;
  double AZ;
  for (i = 0; i < NNZ; i++) {
    if (fscanf(f, "%d %d", &row, &col) != 2) {
      fclose(f);
      return -1;
    }
    row--;
    col--;

    if (mm_is_pattern(matcode)) {
      AZ = 1.0;
    } else {
      if (fscanf(f, "%lf", &AZ) != 1) {
        fclose(f);
        return -1;
      }
    }

    row_counts[row]++;
  }

  for (i = 0; i < M; i++) {
    matrix->IRP[i + 1] = matrix->IRP[i] + row_counts[i];
  }

  for (i = 0; i < M; i++) {
    row_counts[i] = 0;
  }

  rewind(f);

  if (mm_read_banner(f, &matcode) != 0) {
    fclose(f);
    return -1;
  }

  if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &NNZ)) != 0) {
    fclose(f);
    return ret_code;
  }

  for (i = 0; i < NNZ; i++) {
    if (fscanf(f, "%d %d", &row, &col) != 2) {
      fclose(f);
      return -1;
    }
    row--;
    col--;

    if(mm_is_pattern(matcode)) {
      AZ = 1.0;
    } else {
      if (fscanf(f, "%lf", &AZ) != 1) {
        fclose(f);
        return -1;
      }
    }
    int index = matrix->IRP[row] + row_counts[row];
    matrix->JA[index] = col;
    matrix->AZ[index] = AZ;
    row_counts[row]++;
  }

  fclose(f);
  free(row_counts);
  return 0;
}
