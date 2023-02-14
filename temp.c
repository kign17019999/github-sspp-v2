#include <omp.h>
#include <stdio.h>

int main () {
  int nthreads, tid;
  int x;
  int z=0;
  int arr1[3]={1,2,3};
  int arr2[3]={4,5,6};
  float dot=0.0;
  omp_set_num_threads(5);
#pragma omp parallel private(tid)
{
  tid = omp_get_thread_num();
  //printf("Hello World from thread = %d\n", tid);
  x++;
  //printf("%d | x = %d\n",tid, x);
  if (tid == 0)  {
    nthreads = omp_get_num_threads();
    //printf("Number of threads = %d\n", nthreads);
  }
  int y=0;
  #pragma omp for nowait
  for(int i=0; i<10; i++){
    y++;
  }
  //printf("%d | y = %d\n",tid, y);
  //int z=0;
  #pragma omp if(tid >= 2) num_threads(4) firstprivate(z)
  for(int i=0; i<10; i++){
    z++;
  }
  //printf("%d | x=%d, y=%d, z=%d\n",tid, x, y, z);
}
#pragma omp parallel shared(arr1,arr2,dot) private(tid)
{
  tid = omp_get_thread_num();
  #pragma omp for
  for (int i=5; i<5; i++){
    dot += arr1[i]*arr2[i];
  }
  //printf("TID=%d | dot=%d \n", tid, dot);

}
////////////////////////////////////////////////////

  printf("..dynamic..\n");
  for(int chk=1; chk<10; chk++){
    int N=6;
    printf("num chunk=%d | array size = %d\n",chk, N);
    int xx[N];
    int yy[N];
    int cc[N];
    for (int i=0; i < N; i++) xx[i] = yy[i] = i * 1.0;
    int chunk = chk;
    #pragma omp parallel shared(xx,yy,cc,chunk)
    {
      tid = omp_get_thread_num();
      #pragma omp for schedule(dynamic,chunk) nowait
      for (int i=0; i < N; i++){
        cc[i] = xx[i] + yy[i];
        printf("i=%d | tid=%d |cc[%d]=%d\n",i, tid, i, cc[i]);
      }
      //printf("TID=%d |chunk= %d| cc=%d %d %d %d %d \n", tid, chunk, cc[0], cc[1], cc[2], cc[3], cc[4]);

    }
}

////////////////////
  printf("..static..\n");
  for(int chk=1; chk<10; chk++){
    int N=6;
    printf("num chunk=%d | array size = %d\n",chk, N);
    int xx[N];
    int yy[N];
    int cc[N];
    for (int i=0; i < N; i++) xx[i] = yy[i] = i * 1.0;
    int chunk = chk;
    #pragma omp parallel shared(xx,yy,cc,chunk)
    {
      tid = omp_get_thread_num();
      #pragma omp for schedule(static,chunk) nowait
      for (int i=0; i < N; i++){
        cc[i] = xx[i] + yy[i];
        printf("i=%d | tid=%d |cc[%d]=%d\n",i, tid, i, cc[i]);
      }
    }
  }
////////////////////
  printf("..static..\n");
  for(int chk=1; chk<10; chk++){
    int N=6;
    printf("num chunk=%d | array size = %d\n",chk, N);
    int xx[N];
    int yy[N];
    int cc[N];
    for (int i=0; i < N; i++) xx[i] = yy[i] = i * 1.0;
    int chunk = chk;
    #pragma omp parallel shared(xx,yy,cc,chunk)
    {
      tid = omp_get_thread_num();
      #pragma omp for schedule(static,chunk) nowait
      for (int i=0; i < N; i++){
        cc[i] = xx[i] + yy[i];
        printf("i=%d | tid=%d |cc[%d]=%d\n",i, tid, i, cc[i]);
      }
    }
  }




  /* All threads join master thread and terminate */
  return 0;
}
