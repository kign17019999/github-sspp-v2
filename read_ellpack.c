#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"
#include "read_ellpack.h"

int read_ellpack_matrix(const char *file_name, struct ellpack_matrix *matrix) {
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

  int i, j;
  matrix->M = M;
  matrix->N = N;
  int new_NNZ = NNZ;

  int row, col;
  double AZ;
  int *row_counts = (int *) calloc(M, sizeof(int)); //array store number of element each row
  int *row_counts_0 = (int *) calloc(M, sizeof(int)); //array for count available slot each row

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

  int MAXNZ = 0;
  for (i = 0; i < M; i++) { // find mex element per row
    if (row_counts[i] > MAXNZ) {
      MAXNZ = row_counts[i];
    }
  }
  
  matrix->NNZ = new_NNZ;
  matrix->MAXNZ = MAXNZ;
  matrix->JA = (int *) calloc(M * matrix->MAXNZ, sizeof(int)); // initiate 0 to all JA's element
  matrix->AZ = (double *) malloc(M * matrix->MAXNZ * sizeof(double)); 

  rewind(f); // reopen (index) file
  mm_read_banner(f, &matcode); // just for skip
  mm_read_mtx_crd_size(f, &M, &N, &NNZ); // just for skip
  
  for (i = 0; i < NNZ; i++) {
    if (fscanf(f, "%d %d", &row, &col) != 2) {
      fclose(f);
      return -1;
    }
    row--;
    col--;

    if(mm_is_pattern(matcode)) { // if data is binary --> make value =1
      AZ = 1.0;
    } else {
      if (fscanf(f, "%lf", &AZ) != 1) { //if a normal data --> get data from file
        fclose(f);
        return -1;
      }    
    }

    int row_index = row * MAXNZ; // get row index
    int free_col = row_counts_0[row]; // get avaiable col at current row
    matrix->JA[row_index+free_col] = col;
    matrix->AZ[row_index+free_col] = AZ;
    row_counts_0[row]++;
  
    if(mm_is_symmetric(matcode) && row != col){ // if symm and not a diagonal elements --> copy data into opposite site
      row_index = col * MAXNZ; // get row index
      free_col = row_counts_0[col]; // get avaiable col at current row
      matrix->JA[row_index+free_col] = row;
      matrix->AZ[row_index+free_col] = AZ;
      row_counts_0[col]++;
    }
  }

  fclose(f);
  free(row_counts);
  free(row_counts_0);
  return 0;
}

