// Author: Salvatore Filippone salvatore.filippone@cranfield.ac.uk
//

// Computes matrix-vector product. Matrix A is in row-major order
// i.e. A[i, j] is stored in i * COLS + j element of the vector.

#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "wtime.h"
#include "read_csr.h"
#include "read_ellpack.h"
#include <omp.h>

void MatrixVectorCSR(int M, int N, const int* IRP, const int* JA,
 const double* AZ, const double* x, double* restrict y);
void printCSR(int M, int N, int NNZ, const int* IRP, const int* JA,
 const double* AZ);
void MatrixVectorELLPACK(int M, int N, int NNZ, int MAXNZ, const int* JA,
 const double* AZ, const double* x, double* restrict y);
void printELLPACK(int M, int N, int NNZ, int MAXNZ, const int* JA,
 const double* AZ);
void MatrixVectorCSRomp1(int M, int N, const int* IRP, const int* JA,
 const double* AZ, const double* x, double* restrict y) 

int main(int argc, char** argv) 
{
  char* matrix_file = "matrices/cage4.mtx"; // set default file name
  if (argc > 2) {
    printf("Usage: main <matrix_file>\n");
    return -1;
  } else if (argc == 2) {
    matrix_file = argv[1];
  }
  /* CSR */
  struct csr_matrix matrix_csr;
  int ret_code;
  ret_code = read_csr_matrix(matrix_file, &matrix_csr);
  if (ret_code != 0) {
    printf("Failed to read matrix file\n");
    return ret_code;
  }
  //printCSR(matrix_csr.M, matrix_csr.N, matrix_csr.NNZ, matrix_csr.IRP, matrix_csr.JA, matrix_csr.AZ);
  
  double* x = (double*) malloc(sizeof(double)*matrix_csr.M);
  double* y = (double*) malloc(sizeof(double)*matrix_csr.M);
  
  int row;
  for ( row = 0; row < matrix_csr.M; ++row) {
    x[row] = 100.0f * ((double) rand()) / RAND_MAX;      
  }

  double t1, t2;
  t1 = wtime();
  MatrixVectorCSR(matrix_csr.M, matrix_csr.N, matrix_csr.IRP, matrix_csr.JA,
   matrix_csr.AZ, x, y);
  t2 = wtime();
  double tmlt_csr_serial = (t2-t1);
  double mflops_csr_serial = (2.0e-6)*matrix_csr.NNZ/tmlt_csr_serial;
  fprintf(stdout,"[CSR] Matrix-Vector product of size %d x %d with 1 thread: time %lf  MFLOPS %lf \n",
	  matrix_csr.M,matrix_csr.N,tmlt_csr_serial,mflops_csr_serial);
  /* END CSR */

  /* ELLPACK */
  struct ellpack_matrix matrix_ellpack;
  ret_code = read_ellpack_matrix(matrix_file, &matrix_ellpack);
  if (ret_code != 0) {
    printf("Failed to read matrix file\n");
    return ret_code;
  }
  //printELLPACK(matrix_ellpack.M, matrix_ellpack.N, matrix_ellpack.NNZ, matrix_ellpack.MAXNZ, matrix_ellpack.JA, matrix_ellpack.AZ);

  t1 = wtime();
  MatrixVectorELLPACK(matrix_ellpack.M, matrix_ellpack.N, matrix_ellpack.NNZ,
   matrix_ellpack.MAXNZ, matrix_ellpack.JA, matrix_ellpack.AZ, x, y);
  t2 = wtime();
  double tmlt_ell_serial = (t2-t1);
  double mflops_ell_serial = (2.0e-6)*matrix_ellpack.NNZ/tmlt_ell_serial;
  fprintf(stdout,"[ELL] Matrix-Vector product of size %d x %d with 1 thread: time %lf  MFLOPS %lf \n",
	  matrix_ellpack.M,matrix_ellpack.N,tmlt_ell_serial,mflops_ell_serial);
  /* END ELLPACK */

  /* CSR omp v1*/
  t1 = wtime();
  MatrixVectorCSRomp1(matrix_csr.M, matrix_csr.N, matrix_csr.IRP,
   matrix_csr.JA, matrix_csr.AZ, x, y);
  t2 = wtime();
  double tmlt_csr_omp1 = (t2-t1);
  double mflops_csr_omp1 = (2.0e-6)*matrix_csr.NNZ/tmlt_csr_omp1;
  fprintf(stdout,"[CSR omp1] Matrix-Vector product of size %d x %d with 1 thread: time %lf  MFLOPS %lf \n",
	  matrix_csr.M,matrix_csr.N,tmlt_csr_omp1,mflops_csr_omp1);
  /* END CSR omp v1*/

  free(matrix_csr.IRP);
  free(matrix_csr.JA);
  free(matrix_csr.AZ);
  free(matrix_ellpack.JA);
  free(matrix_ellpack.AZ);
  free(x);
  free(y);
  return 0;
}

void MatrixVectorCSR(int M, int N, const int* IRP, const int* JA,
 const double* AZ, const double* x, double* restrict y) 
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

void printCSR(int M, int N, int NNZ, const int* IRP, const int* JA,
 const double* AZ){
  printf("CSR representation:\n");
  printf("M: %d\nN: %d\n", M, N);
  printf("NNZ: %d\n", NNZ);
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
}

void MatrixVectorELLPACK(int M, int N, int NNZ, int MAXNZ, const int* JA,
 const double* AZ, const double* x, double* restrict y) 
{
  int row, col, idx;
  double t;
  for (row = 0; row < M; row++) {
      t = 0;
      for (col = 0; col < MAXNZ; col++) {
          if (col >= NNZ || JA[row*MAXNZ+col] == 0) {
            break;
          }
          t += AZ[col] * x[JA[col]];
      }
      y[row] = t;
  }
}

void printELLPACK(int M, int N, int NNZ, int MAXNZ, const int* JA,
 const double* AZ){
  printf("ELLPACK representation:\n");
  printf("M: %d\nN: %d\n", M, N);
  printf("NNZ: %d\n", NNZ);
  printf("MAXNZ: %d\n", MAXNZ);
  printf("JA: \n");
  int i, j;
  for (i = 0; i < M; i++) {
    for (j = 0; j < MAXNZ; j++) {
      printf("%d ", JA[i * MAXNZ + j]);
      if(j!=MAXNZ-1 && j==5 && MAXNZ>11){
        printf("... ");
        j=MAXNZ-5;
      }
    }
    printf("\n");
    if(i!=M-1 && i==5 && M>11){
      printf(" ... \n");
      i=M-5;
    }
  }
  printf("AZ: \n");
  for (i = 0; i < M; i++) {
    for (j = 0; j < MAXNZ; j++) {
      printf("%.3lf ", AZ[i * MAXNZ + j]);
      if(j!=MAXNZ-1 && j==5 && MAXNZ>11){
        printf("... ");
        j=MAXNZ-5;
      }
    }
    printf("\n");
    if(i!=M-1 && i==5 && M>11){
      printf(" ... \n");
      i=M-5;
    }
  }
}

void MatrixVectorCSRomp1(int M, int N, const int* IRP, const int* JA,
 const double* AZ, const double* x, double* restrict y) 
{
  int row, col, idx;
  double t;
  int chunk_size=128/4;
#pragma omp parallel shared(x, y, chunk_size) private(row, col, idx)
{
#pragma omp for schedule(dynamic, chunk_size) nowait
  for (row = 0; row < M; row++) {
      double t = 0;
      for (col = IRP[row]; col < IRP[row+1]; col++) {
          t += AZ[col] * x[JA[col]];
      }
      y[row] = t;
  }
}