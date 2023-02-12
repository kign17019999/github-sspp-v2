#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"
#include "read_ellpack.h"

// struct ellpack_matrix {
//   int M, N, NNZ, MAXNZ;
//   int *JA;
//   double *AZ;
// };

int read_ellpack_matrix(const char *file_name, struct ellpack_matrix *matrix) {
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

  int i, j;
  matrix->M = M;
  matrix->N = N;
  matrix->NNZ = NNZ;

  int row, col;
  double AZ;
  int *row_counts = (int *) calloc(M, sizeof(int));

  for (i = 0; i < NNZ; i++) {
    if (fscanf(f, "%d %d %lf", &row, &col, &AZ) != 3) {
      fclose(f);
      return -1;
    }
    row--;
    col--;
    row_counts[row]++;
  }

  matrix->MAXNZ = 0;
  for (i = 0; i < M; i++) {
    if (row_counts[i] > matrix->MAXNZ) {
      matrix->MAXNZ = row_counts[i];
    }
  }

  matrix->JA = (int *) malloc(M * matrix->MAXNZ * sizeof(int));
  matrix->AZ = (double *) malloc(M * matrix->MAXNZ * sizeof(double));

  for (i = 0; i < M; i++) {
    for (j = 0; j < matrix->MAXNZ; j++) {
      matrix->JA[i * matrix->MAXNZ + j] = -1;
    }
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
    if (fscanf(f, "%d %d %lf", &row, &col, &AZ) != 3) {
      fclose(f);
      return -1;
    }
    row--;
    col--;

    int index = row * matrix->MAXNZ;
    for (j = 0; j < matrix->MAXNZ; j++) {
      if (matrix->JA[index + j] == -1) {
        matrix->JA[index + j] = col;
        matrix->AZ[index + j] = AZ;
        break;
      }
    }
  }

  fclose(f);
  free(row_counts);
  return 0;
}

