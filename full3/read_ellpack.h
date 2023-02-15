#ifndef READ_ELLPACK_H
#define READ_ELLPACK_H

#include <stdio.h>
#include <stdlib.h>

struct ellpack_matrix {
  int M;
  int N;
  int NNZ;
  int MAXNZ;
  int *JA;
  double *AZ;
};

int read_ellpack_matrix(const char *file_name, struct ellpack_matrix *matrix);

#endif
