#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"
#include "read_csr.h"

int read_csr_matrix(const char *file_name, struct csr_matrix *matrix) {
  MM_typecode matcode;
  int ret_code;

  FILE *f = fopen(file_name, "r");
  if (!f || mm_read_banner(f, &matcode) != 0 || mm_read_mtx_crd_size(f, &matrix->M, &matrix->N, &matrix->NNZ) != 0) {
    if (f) fclose(f);
    return -1;
  }

  matrix->IRP = (int *) malloc((matrix->M + 1) * sizeof(int));
  matrix->JA = (int *) malloc(matrix->NNZ * sizeof(int));
  matrix->AZ = (double *) malloc(matrix->NNZ * sizeof(double));
  matrix->IRP[0] = 0;

  int *row_counts = (int *) calloc(matrix->M, sizeof(int));

  int row, col;
  double AZ;
  for (int i = 0; i < matrix->NNZ; i++) {
    if (fscanf(f, "%d %d", &row, &col) != 2) goto failure;
    row--; col--;
    if (!mm_is_pattern(matcode) && fscanf(f, "%lf", &AZ) != 1) goto failure;
    row_counts[row]++;
  }

  for (int i = 0; i < matrix->M; i++) matrix->IRP[i + 1] = matrix->IRP[i] + row_counts[i];
  for (int i = 0; i < matrix->M; i++) row_counts[i] = 0;
  rewind(f);

  if (mm_read_banner(f, &matcode) != 0 || mm_read_mtx_crd_size(f, &matrix->M, &matrix->N, &matrix->NNZ) != 0) goto failure;

  for (int i = 0; i < matrix->NNZ; i++) {
    if (fscanf(f, "%d %d", &row, &col) != 2) goto failure;
    row--; col--;
    if (!mm_is_pattern(matcode) && fscanf(f, "%lf", &AZ) != 1) goto failure;
    int index = matrix->IRP[row] + row_counts[row]++;
    matrix->JA[index] = col;
    matrix->AZ[index] = AZ;
  }

  free(row_counts);
  fclose(f);
  return 0;

failure:
  free(row_counts);
  fclose(f);
  return -1;
}
