#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mmio.h"

void save_matrix_metadata(char* filename, char* data_type, char* data_format, int M, int N, int NNZ)
{
    FILE* fp = fopen(filename, "w");
    if(fp == NULL)
    {
        printf("Error: failed to create output file '%s'\n", filename);
        exit(1);
    }

    fprintf(fp, "Data Type,Data Format,M,N,NNZ\n");
    fprintf(fp, "%s,%s,%d,%d,%d\n", data_type, data_format, M, N, NNZ);

    fclose(fp);
}

int main(int argc, char *argv[])
{
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int M, N, NNZ;
    char* data_type;
    char* data_format;
    char* output_filename = "matrix_info.csv";

    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s [matrix-market-filename]\n", argv[0]);
        exit(1);
    }

    if ((f = fopen(argv[1], "r")) == NULL)
    {
        printf("Error: failed to open input file '%s'\n", argv[1]);
        exit(1);
    }

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Error: could not process Matrix Market banner.\n");
        exit(1);
    }

    if (!mm_is_valid(matcode))
    {
        printf("Error: invalid Matrix Market file.\n");
        exit(1);
    }

    if (mm_is_complex(matcode))
    {
        printf("Error: this code does not support complex matrices.\n");
        exit(1);
    }

    // Get matrix dimensions and NNZ
    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &NNZ)) != 0)
    {
        printf("Error: failed to read matrix dimensions and NNZ.\n");
        exit(1);
    }

    // Determine data type
    if (mm_is_pattern(matcode))
    {
        data_type = "pattern";
    }
    else if (mm_is_real(matcode))
    {
        data_type = "real";
    }
    else
    {
        printf("Error: unsupported data type.\n");
        exit(1);
    }

    // Determine data format
    if (mm_is_symmetric(matcode))
    {
        data_format = "symmetric";
    }
    else
    {
        data_format = "general";
    }

    // Save matrix metadata to CSV file
    save_matrix_metadata(output_filename, data_type, data_format, M, N, NNZ);

    return 0;
}
