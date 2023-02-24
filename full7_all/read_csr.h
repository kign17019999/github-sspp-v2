#ifndef READ_CSR_H
#define READ_CSR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>

struct csr_matrix {
  int M;
  int N;
  int NNZ;
  int *IRP;
  int *JA;
  double *AZ;
};

int read_csr_matrix(const char *file_name, struct csr_matrix *matrix);

#ifdef __cplusplus
}
#endif

#endif
