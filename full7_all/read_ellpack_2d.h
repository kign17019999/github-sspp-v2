#ifndef READ_ELLPACK_2D_H
#define READ_ELLPACK_2D_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>

struct ellpack_matrix_2d {
  int M;
  int N;
  int NNZ;
  int MAXNZ;
  int **JA;
  double **AZ;
};

int read_ellpack_matrix_2d(const char *file_name, struct ellpack_matrix_2d *matrix);

#ifdef __cplusplus
}
#endif

#endif
