#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"
#include "read_csr.h"

int read_csr_matrix(const char *file_name, struct csr_matrix *matrix) {
  MM_typecode matcode;
  int ret_code;

  FILE *f = fopen(file_name, "r");
  if (!f) { // check availability of file
    return -1;
  }

  if (mm_read_banner(f, &matcode) != 0) { // check availability to read banner
    fclose(f);
    return -1;
  }

  int M, N, NNZ;
  if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &NNZ)) != 0) {
    fclose(f);
    return ret_code;
  }

  int i;
  int new_NNZ = NNZ;
  matrix->M = M;
  matrix->N = N;
  
  matrix->IRP = (int *) malloc((M + 1) * sizeof(int));
  int *row_counts = (int *) calloc(M, sizeof(int));     //array store number of element each row

  int row, col;
  double AZ;
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
    if (mm_is_symmetric(matcode) && row != col) { // if symm and not a diagonal elements >> increase number if row_count and NNZ
        row_counts[col]++;
        new_NNZ++;
    }
  }
  matrix->IRP[0] = 0;  // initiate IRP
  for (i = 0; i < M; i++) matrix->IRP[i + 1] = matrix->IRP[i] + row_counts[i];  // add IRP values
  for (i = 0; i < M; i++) row_counts[i] = 0;    // reset row_count for used couting later

  
  matrix->JA = (int *) malloc(new_NNZ * sizeof(int));
  matrix->AZ = (double *) malloc(new_NNZ * sizeof(double));
  matrix->IRP[0] = 0;
  matrix->NNZ = new_NNZ;
  rewind(f);  // reopen (index) file

  mm_read_banner(f, &matcode);  // just for skip
  mm_read_mtx_crd_size(f, &M, &N, &NNZ); // just for skip

  for (i = 0; i < NNZ; i++) {
    if (fscanf(f, "%d %d", &row, &col) != 2) {
      fclose(f);
      return -1;
    }
    row--;
    col--;

    if(mm_is_pattern(matcode)) {  // if data is binary --> make value =1
      AZ = 1.0;
    } else {
      if (fscanf(f, "%lf", &AZ) != 1) {  //if a normal data --> get data from file
        fclose(f);
        return -1;
      }
    }

     int index = matrix->IRP[row] + row_counts[row]; // get avaiable index at current row
     matrix->JA[index] = col;
     matrix->AZ[index] = AZ;
     row_counts[row]++;
     if (mm_is_symmetric(matcode) && row != col){ // if symm and not a diagonal elements --> copy data into opposite site
         index = matrix->IRP[col] + row_counts[col]; // get avaiable index at current row
         matrix->JA[index] = row;
         matrix->AZ[index] = AZ;
         row_counts[col]++;
     }
  }

  fclose(f);
  free(row_counts);
  return 0;
}



