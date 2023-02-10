#include "mmio.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
  MM_typecode matcode;
  int M, N, nz;
  int i, *I, *J;
  double *val;

  // Read the matrix from file
  FILE *f = fopen("matrix.mtx", "r");
  if (mm_read_banner(f, &matcode) != 0) {
    printf("Could not process Matrix Market banner.\n");
    exit(1);
  }

  // Check if the matrix is real (not complex)
  if (!mm_is_real(matcode)) {
    printf("Sorry, this example does not support complex matrices.\n");
    exit(1);
  }

  // Check if the matrix is sparse
  if (!mm_is_matrix(matcode)) {
    printf("Sorry, this example only supports sparse matrices.\n");
    exit(1);
  }

  // Get the size of the matrix
  if (mm_read_mtx_crd_size(f, &M, &N, &nz) !=0) {
    printf("Could not read matrix size.\n");
    exit(1);
  }

  // Allocate memory for the arrays to store the matrix
  I = (int *) malloc(nz * sizeof(int));
  J = (int *) malloc(nz * sizeof(int));
  val = (double *) malloc(nz * sizeof(double));

  // Read the matrix from the file into the arrays
  for (i=0; i<nz; i++) {
    fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
    I[i]--;  /* adjust from 1-based to 0-based */
    J[i]--;
  }

  // Close the file
  if (f !=NULL) fclose(f);

  // Print the matrix to verify that it was read correctly
  printf("Matrix read from file:\n");
  for (i=0; i<nz; i++)
    printf("%d %d %g\n", I[i], J[i], val[i]);

  // Convert the matrix to CSR format
  int *row_ptr = (int *) malloc((M+1) * sizeof(int));
  int *col_ind = (int *) malloc(nz * sizeof(int));
  double *csr_val = (double *) malloc(nz * sizeof(double));
  int idx = 0;
  row_ptr[0] = 0;
  for (i=0; i<M; i++) {
    int row_nnz = 0;
    while (I[idx] == i) {
      row_nnz++;
      idx++;
    }
    row_ptr[i+1] = row_ptr[i] + row_nnz;
  }
  idx = 0;
  for (i=0; i<M; i++) {
    int row_start = row_ptr[i];
    int row_end = row_ptr[i+1];
    for (int j=row_start; j<row_end; j++) {
      col_ind[j] = J[idx];
      csr_val[j] = val[idx];
      idx++;
    }
  }

  // Print the CSR representation of the matrix to verify
  printf("Matrix stored in CSR format:\n");
  printf("row_ptr: ");
  for (i=0; i<=M; i++)
    printf("%d ", row_ptr[i]);
  printf("\ncol_ind: ");
  for (i=0; i<nz; i++)
    printf("%d ", col_ind[i]);
  printf("\ncsr_val: ");
  for (i=0; i<nz; i++)
    printf("%g ", csr_val[i]);
  printf("\n");

  // Free memory
  free(I);
  free(J);
  free(val);
  free(row_ptr);
  free(col_ind);
  free(csr_val);

  return 0;
}
