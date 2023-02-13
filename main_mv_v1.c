// Author: Salvatore Filippone salvatore.filippone@cranfield.ac.uk
//

// Computes matrix-vector product. Matrix A is in row-major order
// i.e. A[i, j] is stored in i * COLS + j element of the vector.

#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "wtime.h"
#include "read_csr.h"


void MatrixVector(int M, int N, const int* IRP, const int* JA, const double* AZ, const double* x, double* restrict y);

int main(int argc, char** argv) 
{
  char* matrix_file = "matrices/cage4.mtx"; // set default file name
  if (argc > 2) {
    printf("Usage: main <matrix_file>\n");
    return -1;
  } else if (argc == 2) {
    matrix_file = argv[1];
  }
  struct csr_matrix matrix;
  int ret_code = read_csr_matrix(matrix_file, &matrix);
  if (ret_code != 0) {
    printf("Failed to read matrix file\n");
    return ret_code;
  }
  printCSR(matrix.M, matrix.N, matrix.NNZ, matrix.IRP, matrix.JA, matrix.AZ);
  
  double* x = (double*) malloc(sizeof(double)*matrix.M);
  double* y = (double*) malloc(sizeof(double)*matrix.M);
  
  int row;
  for ( row = 0; row < matrix.M; ++row) {
    x[row] = 100.0f * ((double) rand()) / RAND_MAX;      
  }

  double t1 = wtime();
  MatrixVector(matrix.M, matrix.N, matrix.IRP, matrix.JA, matrix.AZ, x, y);
  double t2 = wtime();
  double tmlt = (t2-t1);
  double mflops = (2.0e-6)*matrix.M*matrix.N/tmlt;
  
  fprintf(stdout,"Matrix-Vector product of size %d x %d with 1 thread: time %lf  MFLOPS %lf \n",
	  matrix.M,matrix.N,tmlt,mflops);
  
  free(matrix.IRP);
  free(matrix.JA);
  free(matrix.AZ);
  free(x);
  free(y);
  return 0;
}

void MatrixVector(int M, int N, const int* IRP, const int* JA, const double* AZ, const double* x, double* restrict y) 
{
  int row, col, idx;
  double t;
  for (row = 0; row < M; row++) {
      double t = 0;
      for (col = IRP[row]; col < IRP[row+1]; col++) {
          t += AZ[col] * x[JA[col]];
      }
      y[row] = t;
  }
}

void printCSR(int M, int N, int NNZ, const int* IRP, const int* JA, const double* AZ){
  printf("CSR representation:\n");
  printf("M: %d\nN: %d\n", M, N);
  printf("NNZ: %d\n", matrix.NNZ);
  printf("IRP: ");
  int i;
  for (i = 0; i < M + 1; i++) {
    printf("%d ", IRP[i]);
    if(i!=M-1 && i==5 && M>11){
      printf("... ");
      i=M-5;
    }
  }
  printf("\nJA: ");
  for (i = 0; i < NNZ; i++) {
    printf("%d ", JA[i]);
    if(i!=NNZ-1 && i==5 && NNZ>11){
      printf("... ");
      i=NNZ-5;
    }
  }
  printf("\nAZ: ");
  for (i = 0; i < NNZ; i++) {
    printf("%.3lf ", AZ[i]);
    if(i!=NNZ-1 && i==5 && NNZ>11){
      printf("... ");
      i=NNZ-5;
    }
  }
  printf("\n");
  /* end print CSR */ 

}