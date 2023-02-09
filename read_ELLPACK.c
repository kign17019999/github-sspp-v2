#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"

void read_matrix_ellpack(const char* file, int* row_indices, int* col_indices, double* values, int* num_rows, int* num_cols, int* num_nonzeros, int* max_num_entries_per_row) {
  int ret_code;
  MM_typecode matcode;
  FILE *f;

  if ((f = fopen(file, "r")) == NULL)
    return;

  if (mm_read_banner(f, &matcode) != 0) {
    printf("Could not process Matrix Market banner.\n");
    return;
  }

  /*  This is how one can screen matrix types if their application */
  /*  only supports a subset of the Matrix Market data types.      */
  if (mm_is_complex(matcode) && mm_is_matrix(matcode) &&
      mm_is_sparse(matcode) ) {
    printf("Sorry, this application does not support ");
    printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
    return;
  }

  /* find out size of sparse matrix .... */
  if ((ret_code = mm_read_mtx_crd_size(f, num_rows, num_cols, num_nonzeros)) !=0)
    return;

  int current_row = 0;
  int entry_count = 0;

  /* initialize row indices array */
  row_indices = (int *) malloc(*num_rows * sizeof(int));
  for (int i = 0; i < *num_rows; i++)
    row_indices[i] = -1;

  /* find the maximum number of entries per row */
  for (int i = 0; i < *num_nonzeros; i++) {
    int row, col;
    double value;
    fscanf(f, "%d %d %lg\n", &row, &col, &value);
    row--;  /* adjust from 1-based to 0-based indexing */
    col--;

    if (row != current_row) {
      row_indices[row] = entry_count;
      current_row = row;
    }

    col_indices[entry_count] = col;
    values[entry_count] = value;
    entry_count++;
  }
  row_indices[*num_rows] = *num_nonzeros;

  /* find the maximum number of entries per row */
  *max_num_entries_per_row = 0;
  for (int i = 0; i < *num_rows; i++) {
    int row_length = row_indices[i+1] - row_indices[i];
    if (row_length > *max_num_entries_per_row)
      *max_num_entries_per_row = row_length;
  }
}

int main(int argc, char* argv[]) {
  int num_rows, num_cols, num_nonzeros, max_num_entries_per_row;
  int* row_indices;
  int* col_indices;
  double* values;

  if (argc != 2) {
    printf("Usage: %s [matrix-market-file]\n", argv[0]);
    return 0;
  }

  /* read the sparse matrix */
  read_matrix_ellpack(argv[1], row_indices, col_indices, values, &num_rows, &num_cols, &num_nonzeros, &max_num_entries_per_row);

  /* print results */
  printf("Number of rows: %d\n", num_rows);
  printf("Number of columns: %d\n", num_cols);
  printf("Number of non-zero elements: %d\n", num_nonzeros);
  printf("Max number of entries per row: %d\n", max_num_entries_per_row);
  printf("Row indices:\n");
  for (int i = 0; i < num_rows; i++)
    printf("%d ", row_indices[i]);
  printf("\n");
  printf("Column indices:\n");
  for (int i = 0; i < num_nonzeros; i++)
    printf("%d ", col_indices[i]);
  printf("\n");
  printf("Non-zero values:\n");
  for (int i = 0; i < num_nonzeros; i++)
    printf("%.2lf ", values[i]);
  printf("\n");

  /* free memory */
  free(row_indices);
  free(col_indices);
  free(values);

  return 0;
}
