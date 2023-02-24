#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"
#include "read_ellpack.h"

int read_ellpack_matrix(const char *file_name, struct ellpack_matrix *matrix) {
  MM_typecode matcode;

  FILE *f = fopen(file_name, "r");
  if (!f || mm_read_banner(f, &matcode) != 0 || mm_read_mtx_crd_size(f, &matrix->M, &matrix->N, &matrix->NNZ) != 0) {
    if (f) fclose(f);
    return -1;
  }

  int *row_counts = (int *) calloc(matrix->M, sizeof(int));
  for (int i = 0, row, col; i < matrix->NNZ; i++) {
    if (fscanf(f, "%d %d", &row, &col) != 2) goto failure;
    if (!mm_is_pattern(matcode) && fscanf(f, "%lf", &matrix->AZ[i]) != 1) goto failure;
    matrix->JA[i] = col - 1;
    row_counts[row - 1]++;
  }

  matrix->MAXNZ = 0;
  for (int i = 0; i < matrix->M; i++) {
    if (row_counts[i] > matrix->MAXNZ) matrix->MAXNZ = row_counts[i];
  }

  matrix->JA = (int *) realloc(matrix->JA, matrix->M * matrix->MAXNZ * sizeof(int));
  matrix->AZ = (double *) realloc(matrix->AZ, matrix->M * matrix->MAXNZ * sizeof(double));
  for (int i = matrix->NNZ; i < matrix->M * matrix->MAXNZ; i++) matrix->JA[i] = -1;

  rewind(f);

  if (mm_read_banner(f, &matcode) != 0 || mm_read_mtx_crd_size(f, &matrix->M, &matrix->N, &matrix->NNZ) != 0) goto failure;

  for (int i = 0, row, col, index; i < matrix->NNZ; i++) {
    if (fscanf(f, "%d %d", &row, &col) != 2) goto failure;
    if (!mm_is_pattern(matcode) && fscanf(f, "%lf", &matrix->AZ[i]) != 1) goto failure;
    row--; col--;
    index = row * matrix->MAXNZ;
    for (int j = 0; j < matrix->MAXNZ; j++) {
      if (matrix->JA[index + j] == -1) {
        matrix->JA[index + j] = col;
        break;
      }
    }
  }

  free(row_counts);
  fclose(f);
  return 0;

failure:
  free(row_counts);
  fclose(f);
  return -1;
}


