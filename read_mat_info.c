#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include "mmio.h"

void save_matrix_metadata(char* filename, char* data_type, char* data_format, int M, int N, int NNZ)
{
    // open file for appending or create new file with header
    FILE *fp;
    char save_file_name[] = "mat_info.csv";  //file name
    fp = fopen(save_file_name, "a+");
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

int process_matrix_file(char* filename)
{
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int M, N, NNZ;
    char* data_type;
    char* data_format;

    if ((f = fopen(filename, "r")) == NULL)
    {
        printf("Error: failed to open input file '%s'\n", filename);
        return 1;
    }

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Error: could not process Matrix Market banner in file '%s'\n", filename);
        return 1;
    }

    if (!mm_is_valid(matcode))
    {
        printf("Error: invalid Matrix Market file '%s'\n", filename);
        return 1;
    }

    if (mm_is_complex(matcode))
    {
        printf("Error: this code does not support complex matrices in file '%s'\n", filename);
        return 1;
    }

    // Get matrix dimensions and NNZ
    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &NNZ)) != 0)
    {
        printf("Error: failed to read matrix dimensions and NNZ in file '%s'\n", filename);
        return 1;
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
        printf("Error: unsupported data type in file '%s'\n", filename);
        return 1;
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
    save_matrix_metadata(filename, data_type, data_format, M, N, NNZ);

    return 0;
}

int main(int argc, char *argv[])
{
    char* dir_name;
    DIR* dir;
    struct dirent* ent;

    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s [directory-name]\n", argv[0]);
        exit(1);
    }

    dir_name = argv[1];
    dir = opendir(dir_name);
    if (dir == NULL)
    {
        printf("Error: failed to open directory '%s'\n", dir_name);
        exit(1);
    }

    while ((ent = readdir(dir)) != NULL)
    {
        if (ent->d_name[0] == '.') continue;  //skip hidden files

        char* filename = (char*) malloc(strlen(dir_name) + strlen(ent->d_name) + 2);
        sprintf(filename, "%s/%s", dir_name, ent->d_name);

        if (process_matrix_file(filename) != 0)
        {
            printf("Error processing file '%s'\n", filename);
        }

        free(filename);
    }

    closedir(dir);

    return 0;
}

