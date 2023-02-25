#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"
#include "read_csr.h"
#include <string.h>

void make_csr_matrix_symmetric(struct csr_matrix *matrix);

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

  double **dense_matrix;
  int i, j, k;

  // Allocate memory for dense matrix
  dense_matrix = (double **)malloc(M * sizeof(double *));
  for (i = 0; i < M; i++) {
    dense_matrix[i] = (double *)calloc(N, sizeof(double));
  }

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

    dense_matrix[row][col] = AZ;

    if (mm_is_symmetric(matcode) && row != col) {
      // If matrix is symmetric and not on the diagonal, also store in the transposed position
      dense_matrix[col][row] = AZ;
    }
  }

  fclose(f);

  if (mm_is_symmetric(matcode)) {
    // If matrix is symmetric, make it explicitly symmetric by copying lower triangle to upper triangle or vice versa
    for (i = 0; i < M; i++) {
      for (j = i + 1; j < N; j++) {
        dense_matrix[j][i] = dense_matrix[i][j];
      }
    }
  }

  // Convert dense matrix to CSR format
  int NNZ_CSR = 0;
  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j++) {
      if (dense_matrix[i][j] != 0.0) {
        NNZ_CSR++;
      }
    }
  }

  matrix->M = M;
  matrix->N = N;
  matrix->NNZ = NNZ_CSR;

  matrix->IRP = (int *)malloc((M + 1) * sizeof(int));
  matrix->JA = (int *)malloc(NNZ_CSR * sizeof(int));
  matrix->AZ = (double *)malloc(NNZ_CSR * sizeof(double));

  k = 0;
  matrix->IRP[0] = 0;
  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j++) {
      if (dense_matrix[i][j] != 0.0) {
        matrix->AZ[k] = dense_matrix[i][j];
    matrix->JA[k] = j;
    k++;
  }
}
matrix->IRP[i+1] = k;
}

// Free memory for dense matrix
for (i = 0; i < M; i++) {
free(dense_matrix[i]);
}
free(dense_matrix);

return 0;
}
