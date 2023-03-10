#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"
#include "read_ellpack_2d.h"

int read_ellpack_matrix_2d(const char *file_name, struct ellpack_matrix_2d *matrix) {
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

  matrix->MAXNZ = 0;
  for (i = 0; i < M; i++) {
    if (row_counts[i] > matrix->MAXNZ) {
      matrix->MAXNZ = row_counts[i];
    }
  }

  matrix->JA = (int **) malloc(M * sizeof(int*));
  matrix->AZ = (double **) malloc(M * sizeof(double*));

  for (i = 0; i < M; i++) {
    matrix->JA[i] = (int *) malloc(matrix->MAXNZ * sizeof(int));
    matrix->AZ[i] = (double *) malloc(matrix->MAXNZ * sizeof(double));
  }

  for (i = 0; i < M; i++) {
    for (j = 0; j < matrix->MAXNZ; j++) {
      matrix->JA[i][j] = -1;
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
        fclose(f);
        return -1;
      }    
    }
    for (j = 0; j < matrix->MAXNZ; j++) {
      if (matrix->JA[row][j] == -1) {
        matrix->JA[row][j] = col;
        matrix->AZ[row][j] = AZ;
        break;
      }
    }
  }

  fclose(f);
  free(row_counts);
  return 0;
}