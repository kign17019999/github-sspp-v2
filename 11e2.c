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

  matrix->M = M;
  matrix->N = N;
  matrix->NNZ = NNZ;

  int *row_counts = (int *) calloc(M, sizeof(int));
  int i, j;
  int row, col;
  double AZ;

  // Determine number of non-zero elements per row
  int max_nz = 0;
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
    if (mm_is_symmetric(matcode) && row != col) {
  row_counts[col]++; // Add symmetric element
}
if (row_counts[row] > max_nz) {
  max_nz = row_counts[row];
}
}

// Allocate memory for dense matrix
double **dense = (double **) malloc(M * sizeof(double *));
for (i = 0; i < M; i++) {
dense[i] = (double *) calloc(N, sizeof(double));
}

// Read matrix data and fill in dense matrix
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

dense[row][col] = AZ;
if (mm_is_symmetric(matcode) && row != col) {
  dense[col][row] = AZ; // Add symmetric element
}
}

// Allocate memory for ELLPACK-R format arrays
matrix->MAXNZ = max_nz;
matrix->JA = (int **) malloc(M * sizeof(int *));
for (i = 0; i < M; i++) {
matrix->JA[i] = (int *) malloc(max_nz * sizeof(int));
}

matrix->AZ = (double **) malloc(M * sizeof(double *));
for (i = 0; i < M; i++) {
matrix->AZ[i] = (double *) malloc(max_nz * sizeof(double));
}

// Convert dense matrix to ELLPACK-R format
for (i = 0; i < M; i++) {
int index = 0;
for (j = 0; j < N; j++) {
if (dense[i][j] != 0.0) {
matrix->JA[i][index] = j;
matrix->AZ[i][index] = dense[i][j];
index++;
}
}
for (; index < max_nz; index++) {
matrix->JA[i][index] = -1;
matrix->AZ[i][index] = 0.0;
}
}

// Free memory for dense matrix
for (i = 0; i < M; i++) {
free(dense[i]);
}
free(dense);

// Free memory for row_counts
free(row_counts);

// Close file and return success
fclose(f);
return 0;
}
