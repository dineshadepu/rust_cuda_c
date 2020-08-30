/* File:     vec_add.cu
 * Purpose:  Implement vector addition on a gpu using cuda
 *
 * Compile:  nvcc [-g] [-G] -o vec_add vec_add.cu
 * Run:      ./vec_add
 */

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>

__global__ void Vec_add(double x[], double y[], double z[], int n) {
  int thread_id = threadIdx.x;
  if (thread_id < n){
    z[thread_id] = x[thread_id] + y[thread_id];
  }
}


extern "C" void axpb_cpp_cuda(double h_x[], double h_y[], double h_z[], int n) {
  double *d_x, *d_y, *d_z;
  size_t size;

  /* Define vector length */
  size = n*sizeof(double);

  // Print original vectors.
  printf("h_x = ");
  for (int i = 0; i < n; i++){
    printf("%f ", h_x[i]);
  }
  printf("\n\n");
  printf("h_y = ");
  for (int i = 0; i < n; i++){
    printf("%.1f ", h_y[i]);
  }
  printf("\n\n");


  /* Allocate vectors in device memory */
  cudaMalloc(&d_x, size);
  cudaMalloc(&d_y, size);
  cudaMalloc(&d_z, size);

  /* Copy vectors from host memory to device memory */
  cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);

  /* Kernel Call */
  Vec_add<<<1,1000>>>(d_x, d_y, d_z, n);

  cudaThreadSynchronize();
  cudaMemcpy(h_z, d_z, size, cudaMemcpyDeviceToHost);
  printf("The sum is: \n");
  for (int i = 0; i < n; i++){
    printf("%.1f ", h_z[i]);
  }
  printf("\n");


  /* Free device memory */
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);

  /* Free host memory */
  /* free(h_x); */
  /* free(h_y); */
  /* free(h_z); */


}  /* main */
