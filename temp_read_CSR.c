#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"

void read_matrix_csr(const char* file, int* I, int* J, double* val, int* n, int* m, int* nz) {
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
  if ((ret_code = mm_read_mtx_crd_size(f, n, m, nz)) !=0)
    return;

//   /* reseve memory for matrices */
//   I = (int *) malloc(*n + 1 * sizeof(int));
//   J = (int *) malloc(*nz * sizeof(int));
//   val = (double *) malloc(*nz * sizeof(double));

//   /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
//   /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
//   /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)        */

//   for (int i = 0; i < *n + 1; i++)
//     fscanf(f, "%d", &I[i]);

//   for (int i = 0; i < *nz; i++) {
//     fscanf(f, "%d", &J[i]);
//     fscanf(f, "%lf", &val[i]);
//   }

//   if (f !=stdin) fclose(f);

  return;
}

int main(int argc, char *argv[]) {
  int n, m, nz;
  int *I, *J;
  double *val;

  if (argc != 2) {
    printf("Usage: %s [martix-market-filename]\n", argv[0]);
    return 0;
  } else {
    read_matrix_csr(argv[1], I, J, val, &n, &m, &nz);
  }

  printf("Matrix stored in CSR format:\n");
  printf("Number of rows: %d\n", n);
  printf("Number of columns: %d\n", m);
  printf("Number of non-zero elements: %d\n", nz);
  printf("Row indices (starts from 0):\n");
//   for (int i = 0; i < n + 1; i++)
//     printf("%d ", I[i]);
//   printf("\n");
//   printf("Column indices (starts from 0):\n");
//   for (int i = 0; i < nz; i++)
//     printf("%d ", J[i]);
//   printf("\n");
//   printf("Non-zero values:\n");
//   for (int i = 0; i < nz; i++)
//     printf("%.2lf ", val[i]);
//   printf("\n");

  /* free memory */
  free(I);
  free(J);
  free(val);

  return 0;
}
