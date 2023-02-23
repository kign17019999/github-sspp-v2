#include <stdlib.h>  // Standard input/output library
#include <stdio.h>  // Standard library
#include "read_csr.h" // For import matrix into CSR format
#include "read_ellpack_2d.h"  // For import matrix into ELLPACK format
#include "wtime.h"  // For timing the procress
#include <omp.h>  // For OpenMP programming

const int ntimes = 5;
int nthreads=32;
int chunk_size=256;

void MatrixVectorCSR(int M, int N, const int* IRP, const int* JA,
 const double* AZ, const double* x, double* restrict y);
// void MatrixVectorELLPACK(int M, int N, int NNZ, int MAXNZ, const int* JA,
//  const double* AZ, const double* x, double* restrict y);
void MatrixVectorELLPACK(int M, int N, int NNZ, int MAXNZ, const int** JA,
 const double** AZ, const double* x, double* y);
double check_result(int M, double* restrict y0, double* restrict y);
void save_result_omp(char *program_name, char* matrix_file, int M, int N,
                 int nthreads, int chunk_size,
                 double time_csr_serial, double mflops_csr_serial, double max_diff_csr_serial,
                 double time_ell_serial, double mflops_ell_serial, double max_diff_ell_serial,
                 double time_csr_omp, double mflops_csr_omp, double max_diff_csr_omp,
                 double time_ell_omp, double mflops_ell_omp, double max_diff_ell_omp);

void ompMatrixVectorCSR(int M, int N, const int* IRP, const int* JA,
 const double* AZ, const double* x, double* restrict y) ;
// void ompMatrixVectorELL(int M, int N, int NNZ, int MAXNZ, const int* JA,
//  const double* AZ, const double* x, double* restrict y);
void ompMatrixVectorELL(int M, int N, int NNZ, int MAXNZ, const int** JA,
const double** AZ, const double* x, double* restrict y);

int main(int argc, char** argv) 
{
  char *program_name = argv[0];
  //printf("Run from file %s\n", program_name);
  char* matrix_file;
  if (argc == 2) {
    matrix_file = argv[1];
  } else if (argc == 3) {
    matrix_file = argv[1];
    nthreads = atoi(argv[2]);
  } else if (argc == 4){
    matrix_file = argv[1];
    nthreads = atoi(argv[2]);
    chunk_size = atoi(argv[3]);
  } else {
    printf("Usage: %s matrixFile.mtx [nthreads] [chunk_size]\n", program_name);
    return -1;
  }
  printf("---------------------------------------------------------------------\n");
  printf("Run from file: %s, reading matrix: %s, nthreads: %d, chunk_size: %d\n", program_name, matrix_file, nthreads, chunk_size);
  
  // Set number of thread usage in this program
  omp_set_num_threads(nthreads);

  // ----------------------- Import Matrix Data ----------------------- //

  // Save matrix file into memory in CSR format.
  struct csr_matrix matrix_csr;
  int ret_code;
  ret_code = read_csr_matrix(matrix_file, &matrix_csr);
  if (ret_code != 0) {
    printf(" Failed to read matrix file\n");
    return ret_code;
  }

  // Save matrix file into memory in ELLPACK format.
  struct ellpack_matrix_2d matrix_ellpack;
  ret_code = read_ellpack_matrix_2d(matrix_file, &matrix_ellpack);
  if (ret_code != 0) {
    printf(" Failed to read matrix file\n");
    return ret_code;
  }

  //transpose matrix JA and AZ from ELLPACK format >. to achieve row-wise
  int **JAt = (int **) malloc(matrix_ellpack.MAXNZ * sizeof(int *));
  double **AZt = (double **) malloc(matrix_ellpack.MAXNZ * sizeof(double *));
  for (int i = 0; i < matrix_ellpack.MAXNZ; i++) {
    JAt[i] = (int *) malloc(matrix_ellpack.M * sizeof(int));
    AZt[i] = (double *) malloc(matrix_ellpack.M * sizeof(double));
  }

  for (int i = 0; i < matrix_ellpack.M; i++) {
    for (int j = 0; j < matrix_ellpack.MAXNZ; j++) {
      JAt[j][i] = matrix_ellpack.JA[i][j];
      AZt[j][i] = matrix_ellpack.AZ[i][j];
    }
  }

  // ----------------------- Host memory initialisation ----------------------- //

  double* x = (double*) malloc(sizeof(double)*matrix_csr.N);
  double* y0 = (double*) malloc(sizeof(double)*matrix_csr.M);
  double* y_s_e = (double*) malloc(sizeof(double)*matrix_csr.M);
  double* y_o_c = (double*) malloc(sizeof(double)*matrix_csr.M);
  double* y_o_e = (double*) malloc(sizeof(double)*matrix_csr.M);
  
  int row;
  for ( row = 0; row < matrix_csr.M; ++row) {
    x[row] = 100.0f * ((double) rand()) / RAND_MAX;      
  }
  double t1, t2;
  fprintf(stdout," Matrix-Vector product of %s of size %d x %d\n", matrix_file, matrix_csr.M, matrix_csr.N);

  // ------------------------ Serial Calculations on the CPU ------------------------- //

  // ---- perform serial code in CSR format ---- //
  t1 = wtime();
  for(int tryloop=0; tryloop<ntimes; tryloop++){
    MatrixVectorCSR(matrix_csr.M, matrix_csr.N, matrix_csr.IRP, matrix_csr.JA,
    matrix_csr.AZ, x, y0);
  }
  t2 = wtime();
  double time_csr_serial = (t2-t1)/ntimes; // timing
  double mflops_csr_serial = (2.0e-6)*matrix_csr.NNZ/time_csr_serial; // mflops

  fprintf(stdout," [CSR 1Td] with 1 thread: time %lf  MFLOPS %lf \n",
	  time_csr_serial,mflops_csr_serial);

  // ---- perform serial code in ELLPACK format ---- //
  t1 = wtime();
  for(int tryloop=0; tryloop<ntimes; tryloop++){
    MatrixVectorELLPACK(matrix_ellpack.M, matrix_ellpack.N, matrix_ellpack.NNZ,
    matrix_ellpack.MAXNZ, (const int**) JAt, (const double**) AZt, x, y_s_e);
  }
  t2 = wtime();

  double time_ell_serial = (t2-t1)/ntimes;
  double mflops_ell_serial = (2.0e-6)*matrix_ellpack.NNZ/time_ell_serial;
  double max_diff_ell_serial = check_result(matrix_csr.M, y0, y_s_e);

  fprintf(stdout," [ELL 1Td] with 1 thread: time %lf  MFLOPS %lf max_diff %lf\n",
	  time_ell_serial,mflops_ell_serial, max_diff_ell_serial);

  // ------------------------ Parallel Calculations with OpenMP ------------------------- //

  // ---- perform parallel code in CSR format ---- //
  t1 = wtime();
  for(int tryloop=0; tryloop<ntimes; tryloop++){
    ompMatrixVectorCSR(matrix_csr.M, matrix_csr.N, matrix_csr.IRP,
    matrix_csr.JA, matrix_csr.AZ, x, y_o_c);
  }
  t2 = wtime();

  double time_csr_omp = (t2-t1)/ntimes;
  double mflops_csr_omp = (2.0e-6)*matrix_csr.NNZ/time_csr_omp;
  double max_diff_csr_omp = check_result(matrix_csr.M, y0, y_o_c);

#pragma omp parallel 
{
#pragma omp master
{
  fprintf(stdout," [CSR OMP] with %d thread chunk_size %d: time %lf  MFLOPS %lf max_diff %lf\n",
	  omp_get_num_threads(), chunk_size, time_csr_omp, mflops_csr_omp, max_diff_csr_omp);
}
}

  // ---- perform parallel code in ELLPACK format ---- //
  t1 = wtime();
  for(int tryloop=0; tryloop<ntimes; tryloop++){
    ompMatrixVectorELL(matrix_ellpack.M, matrix_ellpack.N, matrix_ellpack.NNZ, matrix_ellpack.MAXNZ, (const int**) JAt,
     (const double**) AZt, x, y_o_e);
  }
  t2 = wtime();

  double time_ell_omp = (t2-t1)/ntimes;
  double mflops_ell_omp = (2.0e-6)*matrix_ellpack.NNZ/time_ell_omp;
  double max_diff_ell_omp = check_result(matrix_csr.M, y0, y_o_e);

#pragma omp parallel 
{
#pragma omp master
{
  fprintf(stdout," [ELL OMP] with %d thread chunk_size %d: time %lf  MFLOPS %lf max_diff %lf\n",
	  omp_get_num_threads(), chunk_size, time_ell_omp, mflops_ell_omp, max_diff_ell_omp);
}
}

  // ------------------------------- save result into CSV file ------------------------------ //

    save_result_omp(program_name, matrix_file, matrix_csr.M, matrix_csr.N,
                 nthreads, chunk_size,
                 time_csr_serial, mflops_csr_serial, 0,
                 time_ell_serial, mflops_ell_serial, max_diff_ell_serial,
                 time_csr_omp, mflops_csr_omp, max_diff_csr_omp,
                 time_ell_omp, mflops_ell_omp, max_diff_ell_omp);

  // ------------------------------- Cleaning up ------------------------------ //

  free(matrix_csr.IRP);
  free(matrix_csr.JA);
  free(matrix_csr.AZ);
  free(matrix_ellpack.JA);
  free(matrix_ellpack.AZ);
  free(x);
  free(y0);
  free(y_s_e);
  free(y_o_c);
  free(y_o_e);

  return 0;
}

// Simple CPU implementation of matrix_vector product multiplication in CSR format.
void MatrixVectorCSR(int M, int N, const int* IRP, const int* JA,
 const double* AZ, const double* x, double* restrict y) 
{
  int row, col;
  double t;
  for (row = 0; row < M; row++) {
      t = 0;
      for (col = IRP[row]; col < IRP[row+1]; col++) {
          t += AZ[col] * x[JA[col]];
      }
      y[row] = t;
  }
}

// Simple CPU implementation of matrix_vector product in ELLPACK format.
// void MatrixVectorELLPACK(int M, int N, int NNZ, int MAXNZ, const int* JA,
//  const double* AZ, const double* x, double* y) 
// {
//   int row, col;
//   double t;
//   int ja_idx;
//   for (row = 0; row < M; row++) {
//     t = 0;
//     for (col = 0; col < MAXNZ; col++) {
//       ja_idx = row * MAXNZ + col;
//       t += AZ[ja_idx] * x[JA[ja_idx]];
//     }
//     y[row] = t;
//   }
// }

// Simple CPU implementation of matrix_vector product in ELLPACK format.
// void MatrixVectorELLPACK(int M, int N, int NNZ, int MAXNZ, const int** JA,
// const double** AZ, const double* x, double* y)
// {
//   int row, col;
//   double t;
//   for (row = 0; row < M; row++) {
//     t = 0;
//     for (col = 0; col < MAXNZ; col++) {
//       if (col < N) {
//         t += AZ[row][col] * x[JA[row][col]];
//       }
//     }
//     y[row] = t;
//   }
// }

// Simple CPU implementation of matrix_vector product in ELLPACK format.
void MatrixVectorELLPACK(int M, int N, int NNZ, int MAXNZ, const int** JAt,
 const double** AZt, const double* x, double* y)
{
  int row, col;
  double t;
  for (row = 0; row < M; row++) {
    t = 0;
    for (col = 0; col < MAXNZ; col++) {
      if (col < N) {
        t += AZt[col][row] * x[JAt[col][row]];
      }
    }
    y[row] = t;
  }
}

// function to calcucate maximun different of result's element and return the maximun one 
double check_result(int M, double* y0, double* y)
{
  double max_diff = 0;
  double cal_diff = 0;
  for(int i=0; i < M; i++){
    cal_diff = abs(y0[i] - y[i]);
    if(max_diff < cal_diff) max_diff = cal_diff;
  }
  return max_diff;
}

// OpenMP implementation of matrix_vector product in CSR format
void ompMatrixVectorCSR(int M, int N, const int* IRP, const int* JA,
 const double* AZ, const double* x, double* restrict y) 
{
#pragma omp parallel shared(M, N, IRP, JA, AZ, x, y, chunk_size)
{
  double t;
#pragma omp for schedule(dynamic, chunk_size)
  for (int row = 0; row < M; row++) {
      t = 0;
      for (int col = IRP[row]; col < IRP[row+1]; col++) {
          t += AZ[col] * x[JA[col]];
      }
      y[row] = t;
  }
}
}

// OpenMP implementation of matrix_vector product in ELLPACK format
// void ompMatrixVectorELL(int M, int N, int NNZ, int MAXNZ, const int* JA,
//  const double* AZ, const double* x, double* restrict y) 
// {
// #pragma omp parallel shared(M, N, NNZ, MAXNZ, JA, AZ, x, y, chunk_size)
// {
//   double t;
//   int ja_idx;
// #pragma omp for schedule(dynamic, chunk_size)
//   for (int row = 0; row < M; row++) {
//     t = 0;
//     for (int col = 0; col < MAXNZ; col++) {
//       ja_idx = row * MAXNZ + col;
//       t += AZ[ja_idx] * x[JA[ja_idx]];
//     }
//     y[row] = t;
//   }
// }
// }

// OpenMP implementation of matrix_vector product in ELLPACK format
// void ompMatrixVectorELL(int M, int N, int NNZ, int MAXNZ, const int** JA,
// const double** AZ, const double* x, double* restrict y)
// {
// #pragma omp parallel shared(M, N, NNZ, MAXNZ, JA, AZ, x, y, chunk_size)
// {
//   double t;
// #pragma omp for schedule(dynamic, chunk_size)
//   for (int row = 0; row < M; row++) {
//     t = 0;
//     for (int col = 0; col < MAXNZ; col++) {
//       if (col < N) {
//         t += AZ[row][col] * x[JA[row][col]];
//       }
//     }
//     y[row] = t;
//   }
// }
// }

// OpenMP implementation of matrix_vector product in ELLPACK format
// void ompMatrixVectorELL(int M, int N, int NNZ, int MAXNZ, const int** JAt,
// const double** AZt, const double* x, double* restrict y)
// {
// #pragma omp parallel shared(M, N, NNZ, MAXNZ, JAt, AZt, x, y, chunk_size)
// {
//   double t;
// #pragma omp for schedule(dynamic, chunk_size)
//   for (int row = 0; row < M; row++) {
//     t = 0;
//     for (int col = 0; col < MAXNZ; col++) {
//       if (col < N) {
//         t += AZt[col][row] * x[JAt[col][row]];
//       }
//     }
//     y[row] = t;
//   }
// }
// }

// OpenMP implementation of matrix_vector product in ELLPACK format
void ompMatrixVectorELL(int M, int N, int NNZ, int MAXNZ, const int** JAt,
const double** AZt, const double* x, double* restrict y)
{
  const int BLOCK_SIZE = 100;
  double t[BLOCK_SIZE];

#pragma omp parallel shared(M, N, NNZ, MAXNZ, JAt, AZt, x, y, BLOCK_SIZE)
{
#pragma omp for schedule(dynamic)
  for (int row = 0; row < M; row += BLOCK_SIZE) {
    for (int i = 0; i < BLOCK_SIZE; i++) {
      t[i] = 0.0;
    }
    for (int col = 0; col < MAXNZ; col++) {
      for (int i = 0; i < BLOCK_SIZE; i++) {
        int j = row + i;
        if (j < M && col < N) {
          t[i] += AZt[col][j] * x[JAt[col][j]];
        }
      }
    }
    for (int i = 0; i < BLOCK_SIZE; i++) {
      int j = row + i;
      if (j < M) {
        y[j] = t[i];
      }
    }
  }
}
}



// function to save result into CSV file
void save_result_omp(char *program_name, char* matrix_file, int M, int N,
                 int nthreads, int chunk_size,
                 double time_csr_serial, double mflops_csr_serial, double max_diff_csr_serial,
                 double time_ell_serial, double mflops_ell_serial, double max_diff_ell_serial,
                 double time_csr_omp, double mflops_csr_omp, double max_diff_csr_omp,
                 double time_ell_omp, double mflops_ell_omp, double max_diff_ell_omp)
{
  // open file for appending or create new file with header
  FILE *fp;
  char filename[] = "result_omp.csv";  //file name
  fp = fopen(filename, "a+");
  if (fp == NULL) {
    printf("Error opening file.\n");
    exit(1);
  }
  // check if file is empty
  fseek(fp, 0, SEEK_END);
  long file_size = ftell(fp);
  if (file_size == 0) {
    // add header row
    fprintf(fp, "program_name,matrix_file,M,N,nthreads,chunk_size,time_csr_serial,mflops_csr_serial,max_diff_csr_serial,time_ell_serial,mflops_ell_serial,max_diff_ell_serial,time_csr_omp,mflops_csr_omp,max_diff_csr_omp,time_ell_omp,mflops_ell_omp,max_diff_ell_omp\n");
  }

  // write new row to file
  fprintf(fp, "%s,%s,%d,%d,%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
          program_name, matrix_file, M, N,
          nthreads, chunk_size, time_csr_serial, mflops_csr_serial, max_diff_csr_serial,
          time_ell_serial, mflops_ell_serial, max_diff_ell_serial,
          time_csr_omp, mflops_csr_omp, max_diff_csr_omp,
          time_ell_omp, mflops_ell_omp, max_diff_ell_omp);

  // close file
  fclose(fp);
}