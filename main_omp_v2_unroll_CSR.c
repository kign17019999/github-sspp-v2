#include <stdlib.h>  // Standard input/output library
#include <stdio.h>  // Standard library
#include "read_csr.h" // For import matrix into CSR format
#include "read_ellpack.h"  // For import matrix into ELLPACK format store in 1D array.
#include "read_ellpack_2d.h"  // For import matrix into ELLPACK format store in 2D array.
#include "wtime.h"  // For timing the procress
#include <omp.h>  // For OpenMP programming
#include <math.h> // For abs and max
#include <string.h>

char* default_filename = "result_omp.csv";
const int ntimes = 5;
int nthreads=16;
int chunk_size=256;

void MatrixVectorCSR(int M, int N, const int* IRP, const int* JA,
 const double* AZ, const double* x, double* restrict y);
void check_result(int M, double* y_s_c, double* y, double* max_abs_diff, double* max_rel_diff);
void save_result_omp( char *program_name,      char* matrix_file,          int M, int N,                     int NNZ, int MAXNZ,
                      int nthreads,            int chunk_size,
                      double time_csr_serial,  double mflops_csr_serial,   double max_abs_diff_csr_serial,   double max_rel_diff_csr_serial,
                      double time_ell_serial,  double mflops_ell_serial,   double max_abs_diff_ell_serial,   double max_rel_diff_ell_serial,
                      double time_csr_omp,     double mflops_csr_omp,      double max_abs_diff_csr_omp,      double max_rel_diff_csr_omp,
                      double time_ell_1d_omp,  double mflops_ell_1d_omp,   double max_abs_diff_ell_1d_omp,   double max_rel_diff_ell_1d_omp,
                      double time_ell_2d_omp,  double mflops_ell_2d_omp,   double max_abs_diff_ell_2d_omp,   double max_rel_diff_ell_2d_omp, 
                      double time_ell_2dt_omp, double mflops_ell_2dt_omp,  double max_abs_diff_ell_2dt_omp,  double max_rel_diff_ell_2dt_omp);

void ompMatrixVectorCSR(int M, int N, const int* IRP, const int* JA,
 const double* AZ, const double* x, double* restrict y) ;

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
  } else if (argc == 5) {
    matrix_file = argv[1];
    nthreads = atoi(argv[2]);
    chunk_size = atoi(argv[3]);
    default_filename = argv[4];
  } else {
    printf("Usage: %s matrixFile.mtx [nthreads] [chunk_size] [CSV_saving_file_name]\n", program_name);
    return -1;
  }
  printf("---------------------------------------------------------------------\n");
  printf("Run from file: %s, reading matrix: %s, nthreads: %d, chunk_size: %d\n", program_name, matrix_file, nthreads, chunk_size);
  
  // Set number of thread usage in this program
  omp_set_num_threads(nthreads);

  // ======================= Import Matrix Data ======================= //

  // Save matrix file into memory in CSR format.
  struct csr_matrix matrix_csr;
  int ret_code;
  ret_code = read_csr_matrix(matrix_file, &matrix_csr);
  if (ret_code != 0) {
    printf(" Failed to read matrix file\n");
    return ret_code;
  }
  printf("finish loading matrix into CSR format\n");

  // ======================= Host memory initialisation ======================= //

  double* x = (double*) malloc(sizeof(double)*matrix_csr.N);
  double* y_s_c = (double*) malloc(sizeof(double)*matrix_csr.M); // result of serial csr
  double* y_o_c = (double*) malloc(sizeof(double)*matrix_csr.M); // result of omp csr
  
  // random vector element's values
  for (int row = 0; row < matrix_csr.N; ++row) {
    x[row] = 100.0f * ((double) rand()) / RAND_MAX;      
  }
  double t1, t2;
  fprintf(stdout," Matrix-Vector product of %s of size %d x %d\n", matrix_file, matrix_csr.M, matrix_csr.N);

  // ======================= Serial Calculations on the CPU ======================= //

  // ----------------------- perform serial code in CSR format ----------------------- //
  t1 = wtime();
  for(int tryloop=0; tryloop<ntimes; tryloop++){
    MatrixVectorCSR(matrix_csr.M, matrix_csr.N, matrix_csr.IRP, matrix_csr.JA,
    matrix_csr.AZ, x, y_s_c);
  }
  t2 = wtime();
  double time_csr_serial = (t2-t1)/ntimes; // timing
  double mflops_csr_serial = (2.0e-6)*matrix_csr.NNZ/time_csr_serial; // mflops

  fprintf(stdout," [CSR 1Td] with 1 thread: time %lf  MFLOPS %lf \n",
	  time_csr_serial,mflops_csr_serial);

  // ======================= Parallel Calculations with OpenMP ======================= //

  // ----------------------- perform parallel code in CSR format ----------------------- //
  t1 = wtime();
  for(int tryloop=0; tryloop<ntimes; tryloop++){
    ompMatrixVectorCSR(matrix_csr.M, matrix_csr.N, matrix_csr.IRP,
    matrix_csr.JA, matrix_csr.AZ, x, y_o_c);
  }
  t2 = wtime();

  double time_csr_omp = (t2-t1)/ntimes;
  double mflops_csr_omp = (2.0e-6)*matrix_csr.NNZ/time_csr_omp;
  double max_abs_diff_csr_omp, max_rel_diff_csr_omp;
  check_result(matrix_csr.M, y_s_c, y_o_c, &max_abs_diff_csr_omp, &max_rel_diff_csr_omp); // calculate a difference of result

#pragma omp parallel 
{
#pragma omp master
{
  fprintf(stdout," [CSR OMP] with %d thread chunk_size %d: time %lf  MFLOPS %lf max_abs_diff %lf max_rel_diff %lf\n",
	  omp_get_num_threads(), chunk_size, time_csr_omp, mflops_csr_omp, max_abs_diff_csr_omp, max_rel_diff_csr_omp);
}
}

  // ======================= save result into CSV file ======================= //

    save_result_omp(program_name,      matrix_file,        matrix_csr.M, matrix_csr.N, matrix_csr.NNZ, 0,
                    nthreads,          chunk_size,
                    time_csr_serial,   mflops_csr_serial,  0,                        0,
                    0,   0,  0,  0,
                    time_csr_omp,      mflops_csr_omp,     max_abs_diff_csr_omp,     max_rel_diff_csr_omp,
                    0,   0,  0,  0,
                    0,   0,  0,  0,
                    0,  0, 0, 0);


  // ======================= Cleaning up ======================= //

  free(matrix_csr.IRP);
  free(matrix_csr.JA);
  free(matrix_csr.AZ);
  free(x);
  free(y_s_c);
  free(y_o_c);

  return 0;
}

// ******************** Simple CPU implementation of matrix_vector product multiplication in CSR format ******************** //
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

// ******************** function to calculate maximum absolute and relative difference of two arrays ******************** //
void check_result(int M, double* y_s_c, double* y, double* max_abs_diff, double* max_rel_diff)
{
  *max_abs_diff = 0;
  *max_rel_diff = 0;

  for(int i=0; i < M; i++){
    double abs_diff = fabs(y_s_c[i] - y[i]);
    *max_abs_diff = fmax(*max_abs_diff, abs_diff);

    double rel_diff = abs_diff / fmax(fabs(y_s_c[i]), fabs(y[i]));
    *max_rel_diff = fmax(*max_rel_diff, rel_diff);
  }
}

// ******************** OpenMP implementation of matrix_vector product in CSR format ******************** //
void ompMatrixVectorCSR(int M, int N, const int* IRP, const int* JA,
                        const double* AZ, const double* x, double* restrict y)
{
    int i, j, k;
    double t1, t2, t3, t4, t5, t6, t7, t8;
    double sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8;

#pragma omp parallel shared(M, N, IRP, JA, AZ, x, y, chunk_size)
{
#pragma omp for schedule(dynamic, chunk_size)
    for (i = 0; i < M; i++) {
      const int start = IRP[i];
      const int end = IRP[i + 1];
      sum1 = 0.0;
      sum2 = 0.0;
      sum3 = 0.0;
      sum4 = 0.0;
      sum5 = 0.0;
      sum6 = 0.0;
      sum7 = 0.0;
      sum8 = 0.0;

      for (j = start; j < end - 7; j += 8) {
        t1 = AZ[j] * x[JA[j]];
        t2 = AZ[j + 1] * x[JA[j + 1]];
        t3 = AZ[j + 2] * x[JA[j + 2]];
        t4 = AZ[j + 3] * x[JA[j + 3]];
        t5 = AZ[j + 4] * x[JA[j + 4]];
        t6 = AZ[j + 5] * x[JA[j + 5]];
        t7 = AZ[j + 6] * x[JA[j + 6]];
        t8 = AZ[j + 7] * x[JA[j + 7]];
        sum1 += t1;
        sum2 += t2;
        sum3 += t3;
        sum4 += t4;
        sum5 += t5;
        sum6 += t6;
        sum7 += t7;
        sum8 += t8;
      }

      for (k = end - (end - start) % 8; k < end; k++) {
        t1 = AZ[k] * x[JA[k]];
        sum1 += t1;
      }

      y[i] = sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7 + sum8;
    }
}
}

// ******************** function to save result into CSV file ******************** //
void save_result_omp( char *program_name,      char* matrix_file,          int M, int N,                     int NNZ, int MAXNZ,
                      int nthreads,            int chunk_size,
                      double time_csr_serial,  double mflops_csr_serial,   double max_abs_diff_csr_serial,   double max_rel_diff_csr_serial,
                      double time_ell_serial,  double mflops_ell_serial,   double max_abs_diff_ell_serial,   double max_rel_diff_ell_serial,
                      double time_csr_omp,     double mflops_csr_omp,      double max_abs_diff_csr_omp,      double max_rel_diff_csr_omp,
                      double time_ell_1d_omp,  double mflops_ell_1d_omp,   double max_abs_diff_ell_1d_omp,   double max_rel_diff_ell_1d_omp,
                      double time_ell_2d_omp,  double mflops_ell_2d_omp,   double max_abs_diff_ell_2d_omp,   double max_rel_diff_ell_2d_omp, 
                      double time_ell_2dt_omp, double mflops_ell_2dt_omp,  double max_abs_diff_ell_2dt_omp,  double max_rel_diff_ell_2dt_omp)

{
  // open file for appending or create new file with header
  FILE *fp;
  char filename[100];
  strcpy(filename, default_filename);
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
fprintf(fp, "program_name,matrix_file,M,N,NNZ,MAXNZ,");
fprintf(fp, "nthreads,chunk_size,");
fprintf(fp, "time_csr_serial,mflops_csr_serial,max_abs_diff_csr_serial,max_rel_diff_csr_serial,");
fprintf(fp, "time_ell_serial,mflops_ell_serial,max_abs_diff_ell_serial,max_rel_diff_ell_serial,");
fprintf(fp, "time_csr_omp,mflops_csr_omp,max_abs_diff_csr_omp,max_rel_diff_csr_omp,");
fprintf(fp, "time_ell_d1_omp,mflops_ell_d1_omp,max_diff_ell_d1_omp,max_rel_diff_ell_1d_omp,");
fprintf(fp, "time_ell_2d_omp,mflops_ell_2d_omp,max_abs_diff_ell_2d_omp,max_rel_diff_ell_2d_omp,");
fprintf(fp, "time_ell_2dt_omp,mflops_ell_2dt_omp,max_abs_diff_ell_2dt_omp,max_rel_diff_ell_2dt_omp\n");
  }

  // write new row to file
  fprintf(fp, "%s,%s,%d,%d,%d,%d,%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
          program_name,     matrix_file,        M, N,                      NNZ, MAXNZ,
          nthreads,         chunk_size, 
          time_csr_serial,  mflops_csr_serial,  max_abs_diff_csr_serial,  max_rel_diff_csr_serial,
          time_ell_serial,  mflops_ell_serial,  max_abs_diff_ell_serial,  max_rel_diff_ell_serial,
          time_csr_omp,     mflops_csr_omp,     max_abs_diff_csr_omp,     max_rel_diff_csr_omp,
          time_ell_1d_omp,  mflops_ell_1d_omp,  max_abs_diff_ell_1d_omp,  max_rel_diff_ell_1d_omp,
          time_ell_2d_omp,  mflops_ell_2d_omp,  max_abs_diff_ell_2d_omp,  max_rel_diff_ell_2d_omp,
          time_ell_2dt_omp, mflops_ell_2dt_omp, max_abs_diff_ell_2dt_omp, max_rel_diff_ell_2dt_omp);

  // close file
  fclose(fp);
}