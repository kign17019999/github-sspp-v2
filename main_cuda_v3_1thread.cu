#include <cuda_runtime.h>  // For CUDA runtime API
#include <helper_cuda.h>  // For checkCudaError macro
#include <helper_timer.h>  // For CUDA SDK timers

//Simple dimension: define a 1D block structure
#define BD 256

const dim3 BLOCK_DIM(BD);


#include <stdlib.h>
#include <stdio.h>
#include "wtime.h"
#include "read_csr.h"
#include "read_ellpack.h"
inline double dmin ( double a, double b ) { return a < b ? a : b; }

void MatrixVectorCSR(int M, int N, const int* IRP, const int* JA,
 const double* AZ, const double* x, double* y);
void printCSR(int M, int N, int NNZ, const int* IRP, const int* JA,
 const double* AZ);
void MatrixVectorELLPACK(int M, int N, int NNZ, int MAXNZ, const int* JA,
 const double* AZ, const double* x, double* y);
void printELLPACK(int M, int N, int NNZ, int MAXNZ, const int* JA,
 const double* AZ);
int check_result(int M, double* y0, double* y);

__global__ void gpuMatrixVectorCSR(int M, int N, const int* IRP, const int* JA,
 const double* AZ, const double* x, double* y);
__global__ void gpuMatrixVectorELL(int M, int N, int NNZ, int MAXNZ, const int* JA,
 const double* AZ, const double* x, double* y);


int main(int argc, char** argv) 
{
  // Create the CUDA SDK timer.
  StopWatchInterface* timer = 0;
  sdkCreateTimer(&timer);
  timer->reset();

  printf("run from file %s\n", argv[0]);
  char* matrix_file = "matrices/cage4.mtx"; // set default file name
  if (argc == 2) {
    matrix_file = argv[1];
  } else {
    printf("Usage: main <matrix_file> \n");
    return -1;
  }

  /* reading file into CSR */
  struct csr_matrix matrix_csr;
  int ret_code;
  ret_code = read_csr_matrix(matrix_file, &matrix_csr);
  if (ret_code != 0) {
    printf("Failed to read matrix file\n");
    return ret_code;
  }
  //printCSR(matrix_csr.M, matrix_csr.N, matrix_csr.NNZ, matrix_csr.IRP, matrix_csr.JA, matrix_csr.AZ);
  /* END reading file into CSR */

  /* reading file into ELLPACK */
  struct ellpack_matrix matrix_ellpack;
  ret_code = read_ellpack_matrix(matrix_file, &matrix_ellpack);
  if (ret_code != 0) {
    printf("Failed to read matrix file\n");
    return ret_code;
  }
  //printELLPACK(matrix_ellpack.M, matrix_ellpack.N, matrix_ellpack.NNZ, matrix_ellpack.MAXNZ, matrix_ellpack.JA, matrix_ellpack.AZ);
  /* END reading file into ELLPACK */

  double* x = (double*) malloc(sizeof(double)*matrix_csr.N);
  double* y = (double*) malloc(sizeof(double)*matrix_csr.M);
  double* y0 = (double*) malloc(sizeof(double)*matrix_csr.M);
  int row;
  for ( row = 0; row < matrix_csr.M; ++row) {
    x[row] = 100.0f * ((double) rand()) / RAND_MAX;      
  }
  double t1, t2;
  fprintf(stdout,"Matrix-Vector product of %s of size %d x %d\n", matrix_file, matrix_csr.M, matrix_csr.N);
  
  /* CSR Serial*/
  double tmlt_csr_serial = 1e100;
  timer->start();
  t1 = wtime();
  MatrixVectorCSR(matrix_csr.M, matrix_csr.N, matrix_csr.IRP, matrix_csr.JA,
   matrix_csr.AZ, x, y0);
  t2 = wtime();
  timer->stop();
  tmlt_csr_serial = dmin(tmlt_csr_serial,(t2-t1));
  double mflops_csr_serial = (2.0e-6)*matrix_csr.NNZ/tmlt_csr_serial;
  fprintf(stdout,"[CSR] with 1 thread: time %lf  MFLOPS %lf \n",
	  tmlt_csr_serial,mflops_csr_serial);

  double mflops_csr_serial2 = (2.0e-6)*matrix_csr.NNZ/(timer->getTime()/1000);
  fprintf(stdout,"[CSR 2] with X thread: time %lf  MFLOPS %lf\n",
	  timer->getTime(),mflops_csr_serial2);
  
  /* END CSR Serial */

  /* ELLPACK Serial */
  double tmlt_ell_serial = 1e100;
  timer->start();
  t1 = wtime();
  MatrixVectorELLPACK(matrix_ellpack.M, matrix_ellpack.N, matrix_ellpack.NNZ,
   matrix_ellpack.MAXNZ, matrix_ellpack.JA, matrix_ellpack.AZ, x, y);
  t2 = wtime();
  timer->stop();
  tmlt_ell_serial = dmin(tmlt_ell_serial,(t2-t1));
  double mflops_ell_serial = (2.0e-6)*matrix_ellpack.NNZ/tmlt_ell_serial;
  double max_diff_ell_serial = check_result(matrix_csr.M, y0, y);
  fprintf(stdout,"[ELL] with 1 thread: time %lf  MFLOPS %lf max_diff %lf\n",
	  tmlt_ell_serial,mflops_ell_serial, max_diff_ell_serial);

  double mflops_ell_serial2 = (2.0e-6)*matrix_csr.NNZ/(timer->getTime()/1000);
  fprintf(stdout,"[ELL 2] with X thread: time %lf  MFLOPS %lf max_diff %lf\n",
	  timer->getTime(),mflops_ell_serial2, max_diff_csr_cuda); 
  /* END ELLPACK Serial */

  /* ================================== */

  int *d_M, *d_N, *d_NNZ;
  int *d_ell_MAXNZ;
  checkCudaErrors(cudaMalloc(&d_M, sizeof(int)));
  checkCudaErrors(cudaMalloc(&d_N, sizeof(int)));
  checkCudaErrors(cudaMalloc(&d_NNZ, sizeof(int)));
  checkCudaErrors(cudaMalloc(&d_ell_MAXNZ, sizeof(int)));

  checkCudaErrors(cudaMemcpy(d_M, &matrix_csr.M, sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_N, &matrix_csr.N, sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_NNZ, &matrix_csr.NNZ, sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_ell_MAXNZ, &matrix_ellpack.MAXNZ, sizeof(int), cudaMemcpyHostToDevice));

  double *d_csr_AZ, *d_ell_AZ;
  int *d_csr_IRP, *d_csr_JA, *d_ell_JA;
  checkCudaErrors(cudaMalloc((void**) &d_csr_IRP, (matrix_csr.M+1) * sizeof(int)));
  checkCudaErrors(cudaMalloc((void**) &d_csr_JA, matrix_csr.NNZ * sizeof(int)));
  checkCudaErrors(cudaMalloc((void**) &d_csr_AZ, matrix_csr.NNZ * sizeof(double)));
  checkCudaErrors(cudaMalloc((void**) &d_ell_JA, matrix_csr.M * matrix_ellpack.MAXNZ * sizeof(int)));
  checkCudaErrors(cudaMalloc((void**) &d_ell_AZ, matrix_csr.M * matrix_ellpack.MAXNZ * sizeof(double)));

  checkCudaErrors(cudaMemcpy(d_csr_IRP, matrix_csr.IRP, (matrix_csr.M+1) * sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_csr_JA, matrix_csr.JA, matrix_csr.NNZ * sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_csr_AZ, matrix_csr.AZ, matrix_csr.NNZ * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_ell_JA, matrix_ellpack.JA, matrix_csr.M * matrix_ellpack.MAXNZ * sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_ell_AZ, matrix_ellpack.AZ, matrix_csr.M * matrix_ellpack.MAXNZ * sizeof(double), cudaMemcpyHostToDevice));

  double *d_x, *d_y;
  checkCudaErrors(cudaMalloc((void**) &d_x, (matrix_csr.N) * sizeof(double)));
  checkCudaErrors(cudaMalloc((void**) &d_y, (matrix_csr.M) * sizeof(double)));
  checkCudaErrors(cudaMemcpy(d_x, x, matrix_csr.N * sizeof(double), cudaMemcpyHostToDevice));

  const dim3 GRID_DIM((matrix_csr.M - 1 + BLOCK_DIM.x)/ BLOCK_DIM.x  ,1);
  printf("grid dim = %d , block dim = %d \n",GRID_DIM.x,BLOCK_DIM.x);

  timer->start();
  gpuMatrixVectorCSR<<<GRID_DIM, BLOCK_DIM >>>(matrix_csr.M, matrix_csr.N, d_csr_IRP, d_csr_JA, d_csr_AZ, d_x, d_y);
  checkCudaErrors(cudaDeviceSynchronize());
  timer->stop();
  checkCudaErrors(cudaMemcpy(y, d_y, matrix_csr.N*sizeof(double),cudaMemcpyDeviceToHost));
  double mflops_csr_cuda = (2.0e-6)*matrix_csr.NNZ/(timer->getTime()/1000);
  double max_diff_csr_cuda = check_result(matrix_csr.M, y0, y);
  fprintf(stdout,"[CSR cuda] with X thread: time %lf  MFLOPS %lf max_diff %lf\n",
	  timer->getTime(),mflops_csr_cuda, max_diff_csr_cuda);

  timer->reset();
  timer->start();
  gpuMatrixVectorELL<<<GRID_DIM, BLOCK_DIM >>>(matrix_csr.M, matrix_csr.N, matrix_csr.NNZ, matrix_ellpack.MAXNZ, d_ell_JA, d_ell_AZ, d_x, d_y);
  checkCudaErrors(cudaDeviceSynchronize());
  timer->stop();
  checkCudaErrors(cudaMemcpy(y, d_y, matrix_csr.N*sizeof(double),cudaMemcpyDeviceToHost));
  double mflops_ell_cuda = (2.0e-6)*matrix_csr.NNZ/(timer->getTime()/1000);
  double max_diff_ell_cuda = check_result(matrix_csr.M, y0, y);
  fprintf(stdout,"[ELL cuda] with X thread: time %lf  MFLOPS %lf max_diff %lf\n",
	  timer->getTime(),mflops_ell_cuda, max_diff_ell_cuda);

  /* ================================== */

  free(matrix_csr.IRP);
  free(matrix_csr.JA);
  free(matrix_csr.AZ);
  free(matrix_ellpack.JA);
  free(matrix_ellpack.AZ);
  free(x);
  free(y);
  free(y0);
  return 0;
}

void MatrixVectorCSR(int M, int N, const int* IRP, const int* JA,
 const double* AZ, const double* x, double* y) 
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
 const double* AZ, const double* x, double* y) 
{
  int row, col;
  double t;
  for (row = 0; row < M; row++) {
    t = 0;
    for (col = 0; col < MAXNZ; col++) {
      int ja_idx = row * MAXNZ + col;
      if (col >= NNZ || JA[ja_idx] < 0) {
        break;
      }
      t += AZ[ja_idx] * x[JA[ja_idx]];
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

int check_result(int M, double* y0, double* y)
{
  double max_diff = 0;
  double cal_diff = 0;
  for(int i=0; i < M; i++){
    cal_diff = abs(y0[i] - y[i]);
    if(max_diff < cal_diff) max_diff = cal_diff;
  }
  return max_diff;
}

__global__ void gpuMatrixVectorCSR(int M, int N, const int* IRP, const int* JA,
 const double* AZ, const double* x, double* y)
{
  int tr = threadIdx.x;
  int row = blockIdx.x*blockDim.x + tr;
  if (row < M) {
    double t = 0;
    for (int col = IRP[row]; col < IRP[row+1]; col++) {
      t += AZ[col] * x[JA[col]];
    }
    y[row] = t;
  }
}

__global__ void gpuMatrixVectorELL(int M, int N, int NNZ, int MAXNZ, const int* JA,
 const double* AZ, const double* x, double* y)
{
  int tr = threadIdx.x;
  int row = blockIdx.x*blockDim.x + tr;
  if (row < M) {
    double t = 0;
    for (int col = 0; col < MAXNZ; col++) {
      int ja_idx = row * MAXNZ + col;
      if (col >= NNZ || JA[ja_idx] < 0) {
        break;
      }
      t += AZ[ja_idx] * x[JA[ja_idx]];
    }
    y[row] = t;
  }
}