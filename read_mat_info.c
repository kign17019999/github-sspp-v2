#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mmio.h"

void save_matrix_metadata(char* filename, char* data_type, char* data_format, int M, int N, int NNZ)
{
    // open file for appending or create new file with header
    FILE *fp;
    char filename[] = "mat_info.csv";  //file name
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
        fprintf(fp, "filename,Data Type,Data Format,M,N,NNZ\n");
    }

    // write new row to file
    fprintf(fp, "%s,%s,%s,%d,%d,%d\n", filename, data_type, data_format, M, N, NNZ);
     
    // print into console
    fprintf(stdout, "filename: %s, Data Type: %s, Data Format: %s, M: %d, N: %d, NNZ: %d\n", filename, data_type, data_format, M, N, NNZ);

    // close file
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
    save_matrix_metadata(argv[1], data_type, data_format, M, N, NNZ);

    return 0;
}
