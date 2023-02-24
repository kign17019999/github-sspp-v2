#include <stdlib.h>  // Standard input/output library
#include <stdio.h>  // Standard library
#include "read_csr.h" // For import matrix into CSR format
#include "read_ellpack.h"  // For import matrix into ELLPACK format store in 1D array.
#include "read_ellpack_2d.h"  // For import matrix into ELLPACK format store in 2D array.
#include <cuda_runtime.h>  // For CUDA runtime API
#include <helper_cuda.h>  // For checkCudaError macro
#include <helper_timer.h>  // For CUDA SDK timers

const int ntimes = 5;

void MatrixVectorCSR(int M, int N, const int* IRP, const int* JA,
 const double* AZ, const double* x, double* y);
void MatrixVectorELLPACK(int M, int N, int NNZ, int MAXNZ, const int* JA,
 const double* AZ, const double* x, double* y);
double check_result(int M, double* y_s_c, double* y);
void save_result_cuda(char *program_name, char* matrix_file,          int M, int N,
                 int cudaXBD,             int cudaYBD,                int cudaXGD, int cudaYGD,
                 double time_csr_serial,  double mflops_csr_serial,   double max_diff_csr_serial,
                 double time_ell_serial,  double mflops_ell_serial,   double max_diff_ell_serial,
                 double time_csr_gpu,     double mflops_csr_gpu,      double max_diff_csr_gpu,
                 double time_ell_1d_gpu,  double mflops_ell_1d_gpu,   double max_diff_ell_1d_gpu,
                 double time_ell_2d_gpu,  double mflops_ell_2d_gpu,   double max_diff_ell_2d_gpu,
                 double time_ell_2dt_gpu, double mflops_ell_2dt_gpu,  double max_diff_ell_2dt_gpu);

__global__ void gpuMatrixVectorCSR(const int XBD, const int YBD, int M, int N, const int* IRP,
 const int* JA, const double* AZ, const double* x, double* y);
__global__ void gpuMatrixVectorELL(const int XBD, const int YBD, int M, int N, int NNZ, int MAXNZ,
 const int* JA, const double* AZ, const double* x, double* y);
__global__ void gpuMatrixVectorELL_2d(const int XBD, const int YBD, int M, int N, int NNZ, int MAXNZ,
 const int** JA, const double** AZ, const double* x, double* y);
__global__ void gpuMatrixVectorELL_2dt(const int XBD, const int YBD, int M, int N, int NNZ, int MAXNZ,
 const int** JAt, const double** AZt, const double* x, double* y);

int main(int argc, char** argv) 
{
  char *program_name = argv[0];
  //printf("Run from file %s\n", program_name);
  char* matrix_file;
  int XBD=128;  // 2d block dimension
  int YBD=8;  // 2d block dimension 
  if (argc == 2) {
    matrix_file = argv[1];
  } else if(argc == 4){
    matrix_file = argv[1];
    XBD = atoi(argv[2]);
    YBD = atoi(argv[3]);
  } else {
    printf(" Usage: %s matrixFile.mtx [XBD] [YBD] \n", argv[0]);
    return -1;
  }
  printf("---------------------------------------------------------------------\n");
  printf("Run from file: %s, reading matrix: %s, XBD: %d, YBD: %d\n", program_name, matrix_file, XBD, YBD);
  
  // Create the CUDA SDK timer.
  StopWatchInterface* timer = 0;
  sdkCreateTimer(&timer);

  // ----------------------- Import Matrix Data ----------------------- //

  // Save matrix file into memory in CSR format.
  struct csr_matrix matrix_csr;
  int ret_code;
  ret_code = read_csr_matrix(matrix_file, &matrix_csr);
  if (ret_code != 0) {
    printf(" Failed to read matrix file\n");
    return ret_code;
  }
  printf("finish loading matrix into CSR format\n");

  // Save matrix file into memory in ELLPACK format.
  struct ellpack_matrix matrix_ellpack;
  ret_code = read_ellpack_matrix(matrix_file, &matrix_ellpack);
  if (ret_code != 0) {
    printf(" Failed to read matrix file\n");
    return ret_code;
  }
  printf("finish loading matrix into 1D ELLPACK format\n");

  // Save matrix file into memory in ELLPACK format store in 2D array.
  struct ellpack_matrix_2d matrix_ellpack_2d;
  ret_code = read_ellpack_matrix_2d(matrix_file, &matrix_ellpack_2d);
  if (ret_code != 0) {
    printf(" Failed to read matrix file\n");
    return ret_code;
  }
  printf("finish loading matrix into 2D ELLPACK format\n");

  //transpose matrix JA and AZ from 2D ELLPACK format >. to achieve row-wise
  int **JAt = (int **) malloc(matrix_ellpack_2d.MAXNZ * sizeof(int *));
  double **AZt = (double **) malloc(matrix_ellpack_2d.MAXNZ * sizeof(double *));
  for (int j = 0; j < matrix_ellpack_2d.MAXNZ; j++) {
    JAt[j] = (int *) malloc(matrix_ellpack_2d.M * sizeof(int));
    AZt[j] = (double *) malloc(matrix_ellpack_2d.M * sizeof(double));
    for (int i = 0; i < matrix_ellpack_2d.M; i++) {
      JAt[j][i] = matrix_ellpack_2d.JA[i][j];
      AZt[j][i] = matrix_ellpack_2d.AZ[i][j];
    }
  }
  printf("finish loading matrix into 2D tranpose ELLPACK format\n");

  // ----------------------- Host memory initialisation ----------------------- //
  
  double* x = (double*) malloc(sizeof(double)*matrix_csr.N);
  double* y_s_c = (double*) malloc(sizeof(double)*matrix_csr.M); //as a reference of result
  double* y_s_e = (double*) malloc(sizeof(double)*matrix_csr.M); // result of serial ellpack 1d
  double* y_c_c = (double*) malloc(sizeof(double)*matrix_csr.M); // result of omp csr
  double* y_c_e1d = (double*) malloc(sizeof(double)*matrix_csr.M); // result of omp ellpack 1d
  double* y_c_e2d = (double*) malloc(sizeof(double)*matrix_csr.M); // result of omp ellpack 2d
  double* y_c_e2dt = (double*) malloc(sizeof(double)*matrix_csr.M); // result of omp ellpack 2d transpose
   
  // random vector element's values
  for (int row = 0; row < matrix_csr.M; ++row) {
    x[row] = 100.0f * ((double) rand()) / RAND_MAX;      
  }
  fprintf(stdout," Matrix-Vector product of %s of size %d x %d\n", matrix_file, matrix_csr.M, matrix_csr.N);

  // ---------------------- Device memory initialisation ---------------------- //

  //  Allocate memory space on the device. 
  double *d_csr_AZ, *d_ell_AZ;  // matrix data
  int *d_csr_IRP, *d_csr_JA, *d_ell_JA; // matrix data 
  int **d_ell_JA_2d, **d_ell_JA_2dt;  // 2D ell
  double **d_ell_AZ_2d, **d_ell_AZ_2dt; // 2D ell
  size_t pitch_JA_2d, pitch_AZ_2d, pitch_JA_2dt, pitch_AZ_2dt; // pitch
  checkCudaErrors(cudaMalloc((void**) &d_csr_IRP, (matrix_csr.M+1) * sizeof(int)));
  checkCudaErrors(cudaMalloc((void**) &d_csr_JA, matrix_csr.NNZ * sizeof(int)));
  checkCudaErrors(cudaMalloc((void**) &d_csr_AZ, matrix_csr.NNZ * sizeof(double)));
  checkCudaErrors(cudaMalloc((void**) &d_ell_JA, matrix_csr.M * matrix_ellpack.MAXNZ * sizeof(int)));
  checkCudaErrors(cudaMalloc((void**) &d_ell_AZ, matrix_csr.M * matrix_ellpack.MAXNZ * sizeof(double)));
  checkCudaErrors(cudaMallocPitch((void**)&d_ell_JA_2d, &pitch_JA_2d, matrix_ellpack.MAXNZ * sizeof(int), matrix_csr.M));
  checkCudaErrors(cudaMallocPitch((void**)&d_ell_AZ_2d, &pitch_AZ_2d, matrix_ellpack.MAXNZ * sizeof(double), matrix_csr.M));
  checkCudaErrors(cudaMallocPitch((void**)&d_ell_JA_2dt, &pitch_JA_2dt, matrix_csr.M * sizeof(int), matrix_ellpack.MAXNZ));
  checkCudaErrors(cudaMallocPitch((void**)&d_ell_AZ_2dt, &pitch_AZ_2dt, matrix_csr.M * sizeof(double), matrix_ellpack.MAXNZ));

  double *d_x, *d_y; // vector & result data
  checkCudaErrors(cudaMalloc((void**) &d_x, (matrix_csr.N) * sizeof(double)));
  checkCudaErrors(cudaMalloc((void**) &d_y, (matrix_csr.M) * sizeof(double)));

  // Copy data from the host (CPU) to the device (GPU).
  checkCudaErrors(cudaMemcpy(d_csr_IRP, matrix_csr.IRP, (matrix_csr.M+1) * sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_csr_JA, matrix_csr.JA, matrix_csr.NNZ * sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_csr_AZ, matrix_csr.AZ, matrix_csr.NNZ * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_ell_JA, matrix_ellpack.JA, matrix_csr.M * matrix_ellpack.MAXNZ * sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_ell_AZ, matrix_ellpack.AZ, matrix_csr.M * matrix_ellpack.MAXNZ * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy2D(d_ell_JA_2d,  pitch_JA_2d,  matrix_ellpack_2d.JA, matrix_ellpack.MAXNZ * sizeof(int),    matrix_ellpack.MAXNZ * sizeof(int),    matrix_csr.M,         cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy2D(d_ell_AZ_2d,  pitch_AZ_2d,  matrix_ellpack_2d.AZ, matrix_ellpack.MAXNZ * sizeof(double), matrix_ellpack.MAXNZ * sizeof(double), matrix_csr.M,         cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy2D(d_ell_JA_2dt, pitch_JA_2dt, JAt,                  matrix_csr.M         * sizeof(int),    matrix_csr.M         * sizeof(int),    matrix_ellpack.MAXNZ, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy2D(d_ell_AZ_2dt, pitch_AZ_2dt, AZt,                  matrix_csr.M         * sizeof(double), matrix_csr.M         * sizeof(double), matrix_ellpack.MAXNZ, cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMemcpy(d_x, x, matrix_csr.N * sizeof(double), cudaMemcpyHostToDevice));

  // ------------------------ Calculations on the CPU ------------------------- //

  // ---- perform serial code in CSR format ---- //
  timer->reset();
  timer->start();
  for(int tryloop=0; tryloop<ntimes; tryloop++){
    MatrixVectorCSR(matrix_csr.M, matrix_csr.N, matrix_csr.IRP, matrix_csr.JA, matrix_csr.AZ, x, y_s_c);
  }
  timer->stop();

  double time_csr_serial = timer->getTime()/1000/ntimes; // timing
  double mflops_csr_serial = (2.0e-6)*matrix_csr.NNZ/time_csr_serial; // mflops

  fprintf(stdout," [CPU CSR] with 1 thread: time %lf  MFLOPS %lf \n",
	  time_csr_serial,mflops_csr_serial);

  // ---- perform serial code in ELLPACK format ---- //
  timer->reset();
  timer->start();
  for(int tryloop=0; tryloop<ntimes; tryloop++){
    MatrixVectorELLPACK(matrix_ellpack.M, matrix_ellpack.N, matrix_ellpack.NNZ,
     matrix_ellpack.MAXNZ, matrix_ellpack.JA, matrix_ellpack.AZ, x, y_s_e);
  }
  timer->stop();

  double time_ell_serial = timer->getTime()/1000/ntimes;  // timing
  double mflops_ell_serial = (2.0e-6)*matrix_ellpack.NNZ/time_ell_serial; // mflops
  double max_diff_ell_serial = check_result(matrix_csr.M, y_s_c, y_s_e);  // calculate a difference of result

  fprintf(stdout," [CPU ELL] with 1 thread: time %lf  MFLOPS %lf max_diff %lf\n",
	  time_ell_serial,mflops_ell_serial, max_diff_ell_serial);

  // ------------------------ Calculations on the GPU ------------------------- //

  // define a 2D block structure
  const dim3 BLOCK_DIM(XBD,YBD);

  // ---- perform parallel code in CSR format ---- //
  // Calculate the dimension of the grid of blocks
  const dim3 GRID_DIM_CSR((matrix_csr.M-1+BLOCK_DIM.y)/BLOCK_DIM.y, 1);

  timer->reset();
  timer->start();
  for(int tryloop=0; tryloop<ntimes; tryloop++){
    gpuMatrixVectorCSR<<<GRID_DIM_CSR, BLOCK_DIM, XBD*YBD*sizeof(double)>>>(BLOCK_DIM.x, BLOCK_DIM.y,
     matrix_csr.M, matrix_csr.N, d_csr_IRP, d_csr_JA, d_csr_AZ, d_x, d_y);
    checkCudaErrors(cudaDeviceSynchronize());
  }
  timer->stop();

  // Download the resulting vector d_y from the device and store it in y.
  checkCudaErrors(cudaMemcpy(y_c_c, d_y, matrix_csr.M*sizeof(double),cudaMemcpyDeviceToHost));
  
  double time_csr_gpu = timer->getTime()/1000/ntimes; // timing
  double mflops_csr_gpu = (2.0e-6)*matrix_csr.NNZ/time_csr_gpu; // mflops
  double max_diff_csr_gpu = check_result(matrix_csr.M, y_s_c, y_c_c);  // calculate a difference of result

  fprintf(stdout," [GPU CSR] Grid dim = %d %d , Block dim = %d %d time %lf  MFLOPS %lf max_diff %lf\n",
	  GRID_DIM_CSR.x, GRID_DIM_CSR.y, BLOCK_DIM.x, BLOCK_DIM.y, time_csr_gpu,mflops_csr_gpu, max_diff_csr_gpu);

  // ---- perform parallel code in ELLPACK format ---- // 1D //
  // Calculate the dimension of the grid of blocks
  const dim3 GRID_DIM_ELL((matrix_csr.M-1+BLOCK_DIM.y)/BLOCK_DIM.y, 1);

  timer->reset();
  timer->start();
  for(int tryloop=0; tryloop<ntimes; tryloop++){
    gpuMatrixVectorELL<<<GRID_DIM_ELL, BLOCK_DIM, XBD*YBD*sizeof(double)>>>(BLOCK_DIM.x, BLOCK_DIM.y,
     matrix_csr.M, matrix_csr.N, matrix_csr.NNZ, matrix_ellpack.MAXNZ, d_ell_JA, d_ell_AZ, d_x, d_y);
    checkCudaErrors(cudaDeviceSynchronize());
  }
  timer->stop();

  // Download the resulting vector d_y from the device and store it in y.
  checkCudaErrors(cudaMemcpy(y_c_e1d, d_y, matrix_csr.M*sizeof(double),cudaMemcpyDeviceToHost));

  double time_ell_1d_gpu = timer->getTime()/1000/ntimes; // timing
  double mflops_ell_1d_gpu = (2.0e-6)*matrix_csr.NNZ/time_ell_1d_gpu; // mflops
  double max_diff_ell_1d_gpu = check_result(matrix_csr.M, y_s_c, y_c_e1d);  // calculate a difference of result

  fprintf(stdout," [GPU ELL] Grid dim = %d %d , Block dim = %d %d time %lf  MFLOPS %lf max_diff %lf\n",
	  GRID_DIM_ELL.x, GRID_DIM_ELL.y, BLOCK_DIM.x, BLOCK_DIM.y, time_ell_1d_gpu,mflops_ell_1d_gpu, max_diff_ell_1d_gpu);

  // ---- perform parallel code in ELLPACK format ---- // 2D // * * *

  timer->reset();
  timer->start();
  for(int tryloop=0; tryloop<ntimes; tryloop++){
    gpuMatrixVectorELL_2d<<<GRID_DIM_ELL, BLOCK_DIM, XBD*YBD*sizeof(double)>>>(BLOCK_DIM.x, BLOCK_DIM.y,
     matrix_csr.M, matrix_csr.N, matrix_csr.NNZ, matrix_ellpack.MAXNZ, const_cast<const int**>(d_ell_JA_2d), const_cast<const double**>(d_ell_AZ_2d), d_x, d_y);
    checkCudaErrors(cudaDeviceSynchronize());
  }
  timer->stop();

  // Download the resulting vector d_y from the device and store it in y.
  checkCudaErrors(cudaMemcpy(y_c_e2d, d_y, matrix_csr.M*sizeof(double),cudaMemcpyDeviceToHost));

  double time_ell_2d_gpu = timer->getTime()/1000/ntimes; // timing
  double mflops_ell_2d_gpu = (2.0e-6)*matrix_csr.NNZ/time_ell_2d_gpu; // mflops
  double max_diff_ell_2d_gpu = check_result(matrix_csr.M, y_s_c, y_c_e2d);  // calculate a difference of result

  fprintf(stdout," [GPU ELL] Grid dim = %d %d , Block dim = %d %d time %lf  MFLOPS %lf max_diff %lf\n",
	  GRID_DIM_ELL.x, GRID_DIM_ELL.y, BLOCK_DIM.x, BLOCK_DIM.y, time_ell_2d_gpu,mflops_ell_2d_gpu, max_diff_ell_2d_gpu);

  // ---- perform parallel code in ELLPACK format ---- // 2D Transpose // * * *

  timer->reset();
  timer->start();
  for(int tryloop=0; tryloop<ntimes; tryloop++){
    gpuMatrixVectorELL_2dt<<<GRID_DIM_ELL, BLOCK_DIM, XBD*YBD*sizeof(double)>>>(BLOCK_DIM.x, BLOCK_DIM.y,
     matrix_csr.M, matrix_csr.N, matrix_csr.NNZ, matrix_ellpack.MAXNZ, const_cast<const int**>(d_ell_JA_2dt), const_cast<const double**>(d_ell_AZ_2dt), d_x, d_y);
    checkCudaErrors(cudaDeviceSynchronize());
  }
  timer->stop();

  // Download the resulting vector d_y from the device and store it in y.
  checkCudaErrors(cudaMemcpy(y_c_e2dt, d_y, matrix_csr.M*sizeof(double),cudaMemcpyDeviceToHost));

  double time_ell_2dt_gpu = timer->getTime()/1000/ntimes; // timing
  double mflops_ell_2dt_gpu = (2.0e-6)*matrix_csr.NNZ/time_ell_2dt_gpu; // mflops
  double max_diff_ell_2dt_gpu = check_result(matrix_csr.M, y_s_c, y_c_e2dt);  // calculate a difference of result

  fprintf(stdout," [GPU ELL] Grid dim = %d %d , Block dim = %d %d time %lf  MFLOPS %lf max_diff %lf\n",
	  GRID_DIM_ELL.x, GRID_DIM_ELL.y, BLOCK_DIM.x, BLOCK_DIM.y, time_ell_2dt_gpu,mflops_ell_2dt_gpu, max_diff_ell_2dt_gpu);


  // ------------------------------- save result into CSV file ------------------------------ //
  
  save_result_cuda(program_name,    matrix_file,        matrix_csr.M,   matrix_csr.N,
                 GRID_DIM_ELL.x,    GRID_DIM_ELL.y,     GRID_DIM_CSR.x, GRID_DIM_CSR.y,
                 time_csr_serial,   mflops_csr_serial,  0,
                 time_ell_serial,   mflops_ell_serial,  max_diff_ell_serial,
                 time_csr_gpu,      mflops_csr_gpu,     max_diff_csr_gpu,
                 time_ell_1d_gpu,   mflops_ell_1d_gpu,  max_diff_ell_1d_gpu,
                 time_ell_2d_gpu,   mflops_ell_2d_gpu,  max_diff_ell_2d_gpu,
                 time_ell_2dt_gpu,  mflops_ell_2dt_gpu, max_diff_ell_2dt_gpu);

  // ------------------------------- Cleaning up ------------------------------ //

  delete timer;

  free(matrix_csr.IRP);
  free(matrix_csr.JA);
  free(matrix_csr.AZ);
  free(matrix_ellpack.JA);
  free(matrix_ellpack.AZ);
  free(matrix_ellpack_2d.JA);
  free(matrix_ellpack_2d.AZ);
  free(JAt);
  free(AZt);
  free(x);
  free(y_s_c);
  free(y_s_e);
  free(y_c_c);
  free(y_c_e1d);
  free(y_c_e2d);
  free(y_c_e2dt);

  checkCudaErrors(cudaFree(d_csr_AZ));
  checkCudaErrors(cudaFree(d_ell_AZ));
  checkCudaErrors(cudaFree(d_ell_AZ_2d));
  checkCudaErrors(cudaFree(d_ell_AZ_2dt));
  checkCudaErrors(cudaFree(d_csr_IRP));
  checkCudaErrors(cudaFree(d_csr_JA));
  checkCudaErrors(cudaFree(d_ell_JA));
  checkCudaErrors(cudaFree(d_ell_JA_2d));
  checkCudaErrors(cudaFree(d_ell_JA_2dt));
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y));

  return 0;
}

// Simple CPU implementation of matrix_vector product multiplication in CSR format.
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

// Simple CPU implementation of matrix_vector product in ELLPACK format.
void MatrixVectorELLPACK(int M, int N, int NNZ, int MAXNZ, const int* JA,
 const double* AZ, const double* x, double* y) 
{
  int row, col;
  double t;
  int ja_idx;
  for (row = 0; row < M; row++) {
    t = 0;
    for (col = 0; col < MAXNZ; col++) {
      ja_idx = row * MAXNZ + col;
      t += AZ[ja_idx] * x[JA[ja_idx]];
    }
    y[row] = t;
  }
}

// function to calcucate maximun different of result's element and return the maximun one 
double check_result(int M, double* y_s_c, double* y)
{
  double max_diff = 0;
  double cal_diff = 0;
  for(int i=0; i < M; i++){
    cal_diff = abs(y_s_c[i] - y[i]);
    if(max_diff < cal_diff) max_diff = cal_diff;
  }
  return max_diff;
}

// GPU implementation of matrix_vector product in CSR format
__global__ void gpuMatrixVectorCSR(const int XBD, const int YBD, int M, int N, const int* IRP,
 const int* JA, const double* AZ, const double* x, double* y)
{
  int row = blockIdx.x*blockDim.y + threadIdx.y;
  int tid_c = threadIdx.x;
  int tid_r = threadIdx.y;
  int num_threads_per_row = blockDim.x;

  // 1D shared memory is being used because the dimension of the shared memory needs to be specified at runtime.
  extern __shared__ double sdata[]; 

  if (row < M) {
    double t = 0.0;
    for (int col = IRP[row] + tid_c; col < IRP[row+1]; col += blockDim.x) {
      t += AZ[col] * x[JA[col]];
    }
    // Starting address of indexing 1d shared mamory for 2d data
    int sindex = tid_r*XBD+tid_c;
    sdata[sindex] = t;
    __syncthreads();
    
    // Perform row-reduction operation to sum the elements in sdata and store the result in y[row].
    int prev_stride = num_threads_per_row/2;
    for (int stride = num_threads_per_row/2; stride > 0; stride >>= 1) {
      if (tid_c < stride) {
        if(tid_c == stride -1 && prev_stride%2==1){
          sdata[sindex] += sdata[sindex + stride] + sdata[sindex + stride +1];
        }else{
          sdata[sindex] += sdata[sindex + stride];
        }
      }
      __syncthreads();
      prev_stride=stride;
    }

    // Thread 0 writes the final result to global memory
    if (tid_c == 0) {
      y[row] = sdata[sindex];
    }
  }
}

// GPU implementation of matrix_vector product in ELLPACK format // 1D //
__global__ void gpuMatrixVectorELL(const int XBD, const int YBD, int M, int N, int NNZ, int MAXNZ,
 const int* JA, const double* AZ, const double* x, double* y)
{
  int row = blockIdx.x*blockDim.y + threadIdx.y;
  int tid_c = threadIdx.x;
  int tid_r = threadIdx.y;
  int num_threads_per_row = blockDim.x;

  // 1D shared memory is being used because the dimension of the shared memory needs to be specified at runtime.
  extern __shared__ double sdata[];

  if (row < M) {
    double t = 0.0;
    int ja_idx;
    for (int col = tid_c; col < MAXNZ; col += num_threads_per_row) {
      ja_idx = row * MAXNZ + col;
      t += AZ[ja_idx] * x[JA[ja_idx]];
    }
    // Starting address of indexing 1d shared mamory for 2d data
    int sindex = tid_r*XBD+tid_c;
    sdata[sindex] = t;
    __syncthreads();

    // Perform row-reduction operation to sum the elements in sdata and store the result in y[row].
    int prev_stride = num_threads_per_row/2;
    for (int stride = num_threads_per_row/2; stride > 0; stride >>= 1) {
      if (tid_c < stride) {
        if(tid_c == stride -1 && prev_stride%2==1){
          sdata[sindex] += sdata[sindex + stride] + sdata[sindex + stride +1];
        }else{
          sdata[sindex] += sdata[sindex + stride];
        }
      }
      __syncthreads();
      prev_stride=stride;
    }

    // Thread 0 writes the final result to global memory
    if (tid_c == 0) {
      y[row] = sdata[sindex];
    }
  }
}

// GPU implementation of matrix_vector product in ELLPACK format // 2D //
__global__ void gpuMatrixVectorELL_2d(const int XBD, const int YBD, int M, int N, int NNZ, int MAXNZ,
 const int** JA, const double** AZ, const double* x, double* y)
{
  int row = blockIdx.x*blockDim.y + threadIdx.y;
  int tid_c = threadIdx.x;
  int tid_r = threadIdx.y;
  int num_threads_per_row = blockDim.x;

  // 1D shared memory is being used because the dimension of the shared memory needs to be specified at runtime.
  extern __shared__ double sdata[];

  if (row < M) {
    double t = 0.0;
    for (int col = tid_c; col < MAXNZ; col += num_threads_per_row) {
      // Compute the address of the (row, col) element in the JA and AZ arrays
      int* row_JA = (int*)((char*)JA[row] + col * sizeof(int));
      double* row_AZ = (double*)((char*)AZ[row] + col * sizeof(double));

      t += (*row_AZ) * x[*row_JA];
    }
    // Starting address of indexing 1d shared memory for 2d data
    int sindex = tid_r*XBD+tid_c;
    sdata[sindex] = t;
    __syncthreads();

    // Perform row-reduction operation to sum the elements in sdata and store the result in y[row].
    int prev_stride = num_threads_per_row/2;
    for (int stride = num_threads_per_row/2; stride > 0; stride >>= 1) {
      if (tid_c < stride) {
        if(tid_c == stride -1 && prev_stride%2==1){
          sdata[sindex] += sdata[sindex + stride] + sdata[sindex + stride +1];
        }else{
          sdata[sindex] += sdata[sindex + stride];
        }
      }
      __syncthreads();
      prev_stride=stride;
    }

    // Thread 0 writes the final result to global memory
    if (tid_c == 0) {
      y[row] = sdata[sindex];
    }
  }
}


// GPU implementation of matrix_vector product in ELLPACK format // 2D transposed //
__global__ void gpuMatrixVectorELL_2dt(const int XBD, const int YBD, int M, int N, int NNZ, int MAXNZ,
 const int** JAt, const double** AZt, const double* x, double* y)
{
  int row = blockIdx.x*blockDim.y + threadIdx.y;
  int tid_c = threadIdx.x;
  int tid_r = threadIdx.y;
  int num_threads_per_row = blockDim.x;

  // 1D shared memory is being used because the dimension of the shared memory needs to be specified at runtime.
  extern __shared__ double sdata[];

  if (row < M) {
    double t = 0.0;
    for (int col = tid_c; col < MAXNZ; col += num_threads_per_row) {
      // Compute the address of the (col, row) element in the JAt and AZt arrays
      int* row_JAt = (int*)((char*)JAt[col] + row * sizeof(int));
      double* row_AZt = (double*)((char*)AZt[col] + row * sizeof(double));

      t += (*row_AZt) * x[*row_JAt];
    }
    // Starting address of indexing 1d shared memory for 2d data
    int sindex = tid_r*XBD+tid_c;
    sdata[sindex] = t;
    __syncthreads();

    // Perform row-reduction operation to sum the elements in sdata and store the result in y[row].
    int prev_stride = num_threads_per_row/2;
    for (int stride = num_threads_per_row/2; stride > 0; stride >>= 1) {
      if (tid_c < stride) {
        if(tid_c == stride -1 && prev_stride%2==1){
          sdata[sindex] += sdata[sindex + stride] + sdata[sindex + stride +1];
        }else{
          sdata[sindex] += sdata[sindex + stride];
        }
      }
      __syncthreads();
      prev_stride=stride;
    }

    // Thread 0 writes the final result to global memory
    if (tid_c == 0) {
      y[row] = sdata[sindex];
    }
  }
}

// function to save result into CSV file
void save_result_cuda(char *program_name, char* matrix_file,          int M, int N,
                 int cudaXBD,             int cudaYBD,                int cudaXGD, int cudaYGD,
                 double time_csr_serial,  double mflops_csr_serial,   double max_diff_csr_serial,
                 double time_ell_serial,  double mflops_ell_serial,   double max_diff_ell_serial,
                 double time_csr_gpu,     double mflops_csr_gpu,      double max_diff_csr_gpu,
                 double time_ell_1d_gpu,  double mflops_ell_1d_gpu,   double max_diff_ell_1d_gpu,
                 double time_ell_2d_gpu,  double mflops_ell_2d_gpu,   double max_diff_ell_2d_gpu,
                 double time_ell_2dt_gpu, double mflops_ell_2dt_gpu,  double max_diff_ell_2dt_gpu)
{
  // open file for appending or create new file with header
  FILE *fp;
  char filename[] = "result_gpu_test1.csv";  //file name
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
    fprintf(fp, "program_name,matrix_file,M,N,");
    fprintf(fp, "cudaXBD,cudaYBD,cudaXGD,cudaYGD,");
    fprintf(fp, "time_csr_serial,mflops_csr_serial,max_diff_csr_serial,");
    fprintf(fp, "time_ell_serial,mflops_ell_serial,max_diff_ell_serial,");
    fprintf(fp, "time_csr_gpu,mflops_csr_gpu,max_diff_csr_gpu,");
    fprintf(fp, "time_ell_1d_gpu,mflops_ell_1d_gpu,max_diff_ell_1d_gpu,");
    fprintf(fp, "time_ell_2d_gpu,mflops_ell_2d_gpu,max_diff_ell_2d_gpu,");
    fprintf(fp, "time_ell_2dt_gpu,mflops_ell_2dt_gpu,max_diff_ell_2dt_gpu\n");
  }

  // write new row to file
  fprintf(fp, "%s,%s,%d,%d,%d,%d,%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
          program_name,      matrix_file,        M, N,
          cudaXBD, cudaYBD,  cudaXGD,            cudaYGD,
          time_csr_serial,   mflops_csr_serial,  max_diff_csr_serial,
          time_ell_serial,   mflops_ell_serial,  max_diff_ell_serial,
          time_csr_gpu,      mflops_csr_gpu,     max_diff_csr_gpu,
          time_ell_1d_gpu,   mflops_ell_1d_gpu,  max_diff_ell_1d_gpu,
          time_ell_2d_gpu,   mflops_ell_2d_gpu,  max_diff_ell_2d_gpu,
          time_ell_2dt_gpu,  mflops_ell_2dt_gpu, max_diff_ell_2dt_gpu);

  // close file
  fclose(fp);
}