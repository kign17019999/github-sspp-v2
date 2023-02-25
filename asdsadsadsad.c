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
  int new_NNZ = NNZ;

  int row, col;
  double AZ;
  int *row_counts = (int *) calloc(M, sizeof(int));
  int *row_counts_0 = (int *) calloc(M, sizeof(int));

  for (i = 0; i < NNZ; i++) {
    if(mm_is_pattern(matcode)){
        if (fscanf(f, "%d %d", &row, &col) != 2) {
            fclose(f);
            return -1;
        }
    }else{
        if (fscanf(f, "%d %d %lf", &row, &col, &AZ) != 3) {
            fclose(f);
            return -1;
        }
    }
    row--;
    col--;

    row_counts[row]++;
    if (mm_is_symmetric(matcode) && row != col) {
        row_counts[col]++;
        new_NNZ++;
    }
  }  

  int MAXNZ = 0;
  for (i = 0; i < M; i++) {
    if (row_counts[i] > MAXNZ) {
      MAXNZ = row_counts[i];
    }
  }
  
  matrix->NNZ = new_NNZ;
  matrix->MAXNZ = MAXNZ;
  matrix->JA = (int **) malloc(M * sizeof(int*));
  matrix->AZ = (double **) malloc(M * sizeof(double*));

  for (i = 0; i < M; i++) {
    matrix->JA[i] = (int *) calloc(matrix->MAXNZ, sizeof(int));
    matrix->AZ[i] = (double *) malloc(matrix->MAXNZ * sizeof(double));
  }

  rewind(f);
  mm_read_banner(f, &matcode);
  mm_read_mtx_crd_size(f, &M, &N, &NNZ);
  
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

    int free_col = row_counts_0[row];
    matrix->JA[row][free_col] = col;
    matrix->AZ[row][free_col] = AZ;
    row_counts_0[row]++;
  
    if(mm_is_symmetric(matcode) && row != col){
      free_col = row_counts_0[col];
      matrix->JA[col][free_col] = row;
      matrix->AZ[col][free_col] = AZ;
      row_counts_0[col]++;
    }
  }

  fclose(f);
  free(row_counts);
  return 0;
}