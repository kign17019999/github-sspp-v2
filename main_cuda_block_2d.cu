#include <stdlib.h>  // Standard input/output library
#include <stdio.h>  // Standard library
#include "read_csr.h" // For import matrix into CSR format
#include "read_ellpack.h"  // For import matrix into ELLPACK format
#include <cuda_runtime.h>  // For CUDA runtime API
#include <helper_cuda.h>  // For checkCudaError macro
#include <helper_timer.h>  // For CUDA SDK timers

int XBD=128;  // 2d block dimension
int YBD=8;  // 2d block dimension

void MatrixVectorCSR(int M, int N, const int* IRP, const int* JA,
 const double* AZ, const double* x, double* y);
void MatrixVectorELLPACK(int M, int N, int NNZ, int MAXNZ, const int* JA,
 const double* AZ, const double* x, double* y);
double check_result(int M, double* y0, double* y);
void save_result_cuda(char *program_name, char* matrix_file, int M, int N,
                 int cudaXBD, int cudaYBD, int cudaXGD, int cudaYGD,
                 double time_csr_serial, double mflops_csr_serial, double max_diff_csr_serial,
                 double time_ell_serial, double mflops_ell_serial, double max_diff_ell_serial,
                 double time_csr_gpu, double mflops_csr_gpu, double max_diff_csr_gpu,
                 double time_ell_gpu, double mflops_ell_gpu, double max_diff_ell_gpu);

__global__ void gpuMatrixVectorCSR(int XBD, int YBD, int M, int N, const int* IRP, const int* JA,
 const double* AZ, const double* x, double* y);
__global__ void gpuMatrixVectorELL(int XBD, int YBD, int M, int N, int NNZ, int MAXNZ, const int* JA,
 const double* AZ, const double* x, double* y);

int main(int argc, char** argv) 
{
  char *program_name = argv[0];
  printf("Run from file %s\n", program_name);
  char* matrix_file;
  if (argc == 2) {
    matrix_file = argv[1];
  } else if(argc == 4){
    XBD = atoi(argv[2]);
    YBD = atoi(argv[3]);
  } else {
    printf(" Usage: %s [XBD] [YBD] \n", argv[0]);
    return -1;
  }

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

  // Save matrix file into memory in ELLPACK format.
  struct ellpack_matrix matrix_ellpack;
  ret_code = read_ellpack_matrix(matrix_file, &matrix_ellpack);
  if (ret_code != 0) {
    printf(" Failed to read matrix file\n");
    return ret_code;
  }

  // ----------------------- Host memory initialisation ----------------------- //
  
  double* x = (double*) malloc(sizeof(double)*matrix_csr.N);
  double* y = (double*) malloc(sizeof(double)*matrix_csr.M);
  double* y0 = (double*) malloc(sizeof(double)*matrix_csr.M); //as a reference of result
  for (int row = 0; row < matrix_csr.M; ++row) {
    x[row] = 100.0f * ((double) rand()) / RAND_MAX;      
  }
  fprintf(stdout," Matrix-Vector product of %s of size %d x %d\n", matrix_file, matrix_csr.M, matrix_csr.N);

  // ---------------------- Device memory initialisation ---------------------- //

  //  Allocate memory space on the device. 
  double *d_csr_AZ, *d_ell_AZ;  // matrix data
  int *d_csr_IRP, *d_csr_JA, *d_ell_JA; // matrix data
  checkCudaErrors(cudaMalloc((void**) &d_csr_IRP, (matrix_csr.M+1) * sizeof(int)));
  checkCudaErrors(cudaMalloc((void**) &d_csr_JA, matrix_csr.NNZ * sizeof(int)));
  checkCudaErrors(cudaMalloc((void**) &d_csr_AZ, matrix_csr.NNZ * sizeof(double)));
  checkCudaErrors(cudaMalloc((void**) &d_ell_JA, matrix_csr.M * matrix_ellpack.MAXNZ * sizeof(int)));
  checkCudaErrors(cudaMalloc((void**) &d_ell_AZ, matrix_csr.M * matrix_ellpack.MAXNZ * sizeof(double)));

  double *d_x, *d_y; // vector & result data
  checkCudaErrors(cudaMalloc((void**) &d_x, (matrix_csr.N) * sizeof(double)));
  checkCudaErrors(cudaMalloc((void**) &d_y, (matrix_csr.M) * sizeof(double)));

  // Copy data from the host (CPU) to the device (GPU).
  checkCudaErrors(cudaMemcpy(d_csr_IRP, matrix_csr.IRP, (matrix_csr.M+1) * sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_csr_JA, matrix_csr.JA, matrix_csr.NNZ * sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_csr_AZ, matrix_csr.AZ, matrix_csr.NNZ * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_ell_JA, matrix_ellpack.JA, matrix_csr.M * matrix_ellpack.MAXNZ * sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_ell_AZ, matrix_ellpack.AZ, matrix_csr.M * matrix_ellpack.MAXNZ * sizeof(double), cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMemcpy(d_x, x, matrix_csr.N * sizeof(double), cudaMemcpyHostToDevice));

  // ------------------------ Calculations on the CPU ------------------------- //

  // ---- perform serial code in CSR format ---- //
  timer->reset();
  timer->start();
  MatrixVectorCSR(matrix_csr.M, matrix_csr.N, matrix_csr.IRP, matrix_csr.JA,
   matrix_csr.AZ, x, y0);
  timer->stop();

  double time_csr_serial = timer->getTime()/1000;
  double mflops_csr_serial = (2.0e-6)*matrix_csr.NNZ/time_csr_serial;

  fprintf(stdout," [CPU CSR] with 1 thread: time %lf  MFLOPS %lf \n",
	  time_csr_serial,mflops_csr_serial);

  // ---- perform serial code in ELLPACK format ---- //
  timer->reset();
  timer->start();
  MatrixVectorELLPACK(matrix_ellpack.M, matrix_ellpack.N, matrix_ellpack.NNZ,
   matrix_ellpack.MAXNZ, matrix_ellpack.JA, matrix_ellpack.AZ, x, y);
  timer->stop();

  double time_ell_serial = timer->getTime()/1000;
  double mflops_ell_serial = (2.0e-6)*matrix_ellpack.NNZ/time_ell_serial;
  double max_diff_ell_serial = check_result(matrix_csr.M, y0, y);  // calculate a difference of result

  fprintf(stdout," [CPU ELL] with 1 thread: time %lf  MFLOPS %lf max_diff %lf\n",
	  time_ell_serial,mflops_ell_serial, max_diff_ell_serial);

  // ------------------------ Calculations on the GPU ------------------------- //

  // define a 1D block structure
  const dim3 BLOCK_DIM(XBD,YBD);

  // ---- perform parallel code in CSR format ---- //
  // Calculate the dimension of the grid of blocks
  const dim3 GRID_DIM_CSR((matrix_csr.M-1+BLOCK_DIM.y)/BLOCK_DIM.y, 1);
  printf(" [GPU CSR] Grid dim = %d %d , Block dim = %d %d \n",GRID_DIM_CSR.x, GRID_DIM_CSR.y, BLOCK_DIM.x, BLOCK_DIM.y);

  timer->reset();
  timer->start();
  gpuMatrixVectorCSR<<<GRID_DIM_CSR, BLOCK_DIM>>>(XBD, YBD, matrix_csr.M, matrix_csr.N, d_csr_IRP, d_csr_JA, d_csr_AZ, d_x, d_y);
  checkCudaErrors(cudaDeviceSynchronize());
  timer->stop();

  // Download the resulting vector d_y from the device and store it in y.
  checkCudaErrors(cudaMemcpy(y, d_y, matrix_csr.M*sizeof(double),cudaMemcpyDeviceToHost));
  
  double time_csr_gpu = timer->getTime()/1000;
  double mflops_csr_gpu = (2.0e-6)*matrix_csr.NNZ/time_csr_gpu;
  double max_diff_csr_gpu = check_result(matrix_csr.M, y0, y);  // calculate a difference of result

  fprintf(stdout," [GPU CSR] with CUDA: time %lf  MFLOPS %lf max_diff %lf\n",
	  time_csr_gpu,mflops_csr_gpu, max_diff_csr_gpu);

  // ---- perform parallel code in ELLPACK format ---- //
  // Calculate the dimension of the grid of blocks
  const dim3 GRID_DIM_ELL((matrix_csr.M-1+BLOCK_DIM.y)/BLOCK_DIM.y, 1);
  printf(" [GPU ELL] Grid dim = %d %d , Block dim = %d %d \n",GRID_DIM_ELL.x, GRID_DIM_ELL.y, BLOCK_DIM.x, BLOCK_DIM.y);

  timer->reset();
  timer->start();
  gpuMatrixVectorELL<<<GRID_DIM_ELL, BLOCK_DIM>>>(XBD, YBD, matrix_csr.M, matrix_csr.N, matrix_csr.NNZ, matrix_ellpack.MAXNZ, d_ell_JA, d_ell_AZ, d_x, d_y);
  checkCudaErrors(cudaDeviceSynchronize());
  timer->stop();

  // Download the resulting vector d_y from the device and store it in y.
  checkCudaErrors(cudaMemcpy(y, d_y, matrix_csr.M*sizeof(double),cudaMemcpyDeviceToHost));

  double time_ell_gpu = timer->getTime()/1000;
  double mflops_ell_gpu = (2.0e-6)*matrix_csr.NNZ/time_ell_gpu;
  double max_diff_ell_gpu = check_result(matrix_csr.M, y0, y);  // calculate a difference of result

  fprintf(stdout," [GPU ELL] with CUDA: time %lf  MFLOPS %lf max_diff %lf\n",
	  time_ell_gpu,mflops_ell_gpu, max_diff_ell_gpu);


  // ------------------------------- save result into CSV file ------------------------------ //
  
  save_result_cuda(program_name, matrix_file, matrix_csr.M, matrix_csr.N,
                 XBD, YBD, GRID_DIM_CSR.x, GRID_DIM_CSR.y,
                 time_csr_serial, mflops_csr_serial, 0,
                 time_ell_serial, mflops_ell_serial, max_diff_ell_serial,
                 time_csr_gpu, mflops_csr_gpu, max_diff_csr_gpu,
                 time_ell_gpu, mflops_ell_gpu, max_diff_ell_gpu);

  // ------------------------------- Cleaning up ------------------------------ //

  delete timer;

  free(matrix_csr.IRP);
  free(matrix_csr.JA);
  free(matrix_csr.AZ);
  free(matrix_ellpack.JA);
  free(matrix_ellpack.AZ);
  free(x);
  free(y);
  free(y0);

  checkCudaErrors(cudaFree(d_csr_AZ));
  checkCudaErrors(cudaFree(d_ell_AZ));
  checkCudaErrors(cudaFree(d_csr_IRP));
  checkCudaErrors(cudaFree(d_csr_JA));
  checkCudaErrors(cudaFree(d_ell_JA));
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y));

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

__global__ void gpuMatrixVectorCSR(int XBD, int YBD, int M, int N, const int* IRP, const int* JA, const double* AZ, const double* x, double* y)
{
  int row = blockIdx.x*blockDim.y + threadIdx.y;
  int tid_c = threadIdx.x;
  int tid_r = threadIdx.y;
  int num_threads_per_row = blockDim.x;

  __shared__ double sdata[YBD][XBD]; // Allocate shared memory for vector x

  if (row < M) {
    double t = 0.0;
    for (int col = IRP[row] + tid_c; col < IRP[row+1]; col += blockDim.x) {
      t += AZ[col] * x[JA[col]];
    }
    sdata[tid_r][tid_c] = t;
    __syncthreads();
    
    // Perform row-reduction operation to sum the elements in sdata and store the result in y[row].
    int prev_stride = num_threads_per_row/2;
    for (int stride = num_threads_per_row/2; stride > 0; stride >>= 1) {
      if (tid_c < stride) {
        if(tid_c == stride -1 && prev_stride%2==1){
          sdata[tid_r][tid_c] += sdata[tid_r][tid_c + stride] + sdata[tid_r][tid_c + stride +1];
        }else{
          sdata[tid_r][tid_c] += sdata[tid_r][tid_c + stride];
        }
      }
      __syncthreads();
      prev_stride=stride;
    }

    // Thread 0 writes the final result to global memory
    if (tid_c == 0) {
      y[row] = sdata[tid_r][tid_c];
    }
  }
}

__global__ void gpuMatrixVectorELL(int XBD, int YBD, int M, int N, int NNZ, int MAXNZ, const int* JA, const double* AZ, const double* x, double* y)
{
  int row = blockIdx.x*blockDim.y + threadIdx.y;
  int tid_c = threadIdx.x;
  int tid_r = threadIdx.y;
  int num_threads_per_row = blockDim.x;

  __shared__ double sdata[YBD][XBD]; // Allocate shared memory for vector x

  if (row < M) {
    double t = 0.0;
    for (int col = tid_c; col < MAXNZ; col += num_threads_per_row) {
      int ja_idx = row * MAXNZ + col;
      if (col >= NNZ || JA[ja_idx] < 0) {
        break;
      }
      t += AZ[ja_idx] * x[JA[ja_idx]];
    }
    sdata[tid_r][tid_c] = t;
    __syncthreads();

    // Perform row-reduction operation to sum the elements in sdata and store the result in y[row].
    int prev_stride = num_threads_per_row/2;
    for (int stride = num_threads_per_row/2; stride > 0; stride >>= 1) {
      if (tid_c < stride) {
        if(tid_c == stride -1 && prev_stride%2==1){
          sdata[tid_r][tid_c] += sdata[tid_r][tid_c + stride] + sdata[tid_r][tid_c + stride +1];
        }else{
          sdata[tid_r][tid_c] += sdata[tid_r][tid_c + stride];
        }
      }
      __syncthreads();
      prev_stride=stride;
    }

    // Thread 0 writes the final result to global memory
    if (tid_c == 0) {
      y[row] = sdata[tid_r][tid_c];
    }
  }
}

void save_result_cuda(char *program_name, char* matrix_file, int M, int N,
                 int cudaXBD, int cudaYBD, int cudaXGD, int cudaYGD,
                 double time_csr_serial, double mflops_csr_serial, double max_diff_csr_serial,
                 double time_ell_serial, double mflops_ell_serial, double max_diff_ell_serial,
                 double time_csr_gpu, double mflops_csr_gpu, double max_diff_csr_gpu,
                 double time_ell_gpu, double mflops_ell_gpu, double max_diff_ell_gpu)
{
  // open file for appending or create new file with header
  FILE *fp;
  char filename[] = "result_gpu.csv";  //file name
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
    fprintf(fp, "program_name,matrix_file,M,N,cudaXBD,cudaYBD,cudaXGD,cudaYGD,time_csr_serial,mflops_csr_serial,max_diff_csr_serial,time_ell_serial,mflops_ell_serial,max_diff_ell_serial,time_csr_gpu,mflops_csr_gpu,max_diff_csr_gpu,time_ell_gpu,mflops_ell_gpu,max_diff_ell_gpu\n");
  }

  // write new row to file
  fprintf(fp, "%s,%s,%d,%d,%d,%d,%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
          program_name, matrix_file, M, N,
          cudaXBD, cudaYBD, cudaXGD, cudaYGD, time_csr_serial, mflops_csr_serial, max_diff_csr_serial,
          time_ell_serial, mflops_ell_serial, max_diff_ell_serial,
          time_csr_gpu, mflops_csr_gpu, max_diff_csr_gpu,
          time_ell_gpu, mflops_ell_gpu, max_diff_ell_gpu);

  // close file
  fclose(fp);
}