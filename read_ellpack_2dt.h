#ifndef READ_ELLPACK_2DT_H
#define READ_ELLPACK_2DT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>

struct ellpack_matrix_2dt {
  int M;
  int N;
  int NNZ;
  int MAXNZ;
  int **JA;
  double **AZ;
};

int read_ellpack_matrix_2dt(const char *file_name, struct ellpack_matrix_2dt *matrix);

#ifdef __cplusplus
}
#endif

#endif
