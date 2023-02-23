#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"
#include "read_ellpack_2dt.h"

int read_ellpack_matrix_2dt(const char *file_name, struct ellpack_matrix_2dt *matrix) {
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
  int *col_counts = (int *) calloc(N, sizeof(int));

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

    col_counts[col]++;
  }  

  matrix->MAXNZ = 0;
  for (j = 0; j < N; j++) {
    if (col_counts[j] > matrix->MAXNZ) {
      matrix->MAXNZ = col_counts[j];
    }
  }

  matrix->JA = (int **) malloc(N * sizeof(int*));
  matrix->AZ = (double **) malloc(N * sizeof(double*));

  for (j = 0; j < N; j++) {
    matrix->JA[j] = (int *) malloc(matrix->MAXNZ * sizeof(int));
    matrix->AZ[j] = (double *) malloc(matrix->MAXNZ * sizeof(double));
  }

  for (j = 0; j < N; j++) {
    for (i = 0; i < matrix->MAXNZ; i++) {
      matrix->JA[j][i] = -1;
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
        fclose;
        return -1;
      }
    }
    int temp = row;
    row = col;
    col = temp;
    for (j = 0; j < matrix->MAXNZ; j++) {
      if (matrix->JA[row][j] == -1) {
        matrix->JA[row][j] = col;
        matrix->AZ[row][j] = AZ;
        break;
      }
    }
  }

  fclose(f);
  free(col_counts);
  return 0;
}