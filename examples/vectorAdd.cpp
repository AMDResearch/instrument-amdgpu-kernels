
/******************************************************************************
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*******************************************************************************/
#include "hip/hip_runtime.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void vecAdd(double *a, double *b, double *c, int n) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < n)
    c[id] = a[id] + b[id];
}

int main(int argc, char *argv[]) {
  // Size of vectors
  int n = 1024;

  // Host input vectors
  double *h_a;
  double *h_b;
  // Host output vector
  double *h_c;

  // Device input vectors
  double *d_a;
  double *d_b;
  // Device output vector
  double *d_c;

  // Size, in bytes, of each vector
  size_t bytes = n * sizeof(double);

  // Allocate memory for each vector on host
  h_a = (double *)malloc(bytes);
  h_b = (double *)malloc(bytes);
  h_c = (double *)malloc(bytes);

  // Allocate memory for each vector on GPU
  (void)hipMalloc(&d_a, bytes);
  (void)hipMalloc(&d_b, bytes);
  (void)hipMalloc(&d_c, bytes);

  int i;
  // Initialize vectors on host
  for (i = 0; i < n; i++) {
    h_a[i] = sin(i) * sin(i);
    h_b[i] = cos(i) * cos(i);
  }

  // Copy host vectors to device
  (void)hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice);
  (void)hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice);

  int blockSize, gridSize;

  // Number of threads in each thread block
  blockSize = 1024;

  // Number of thread blocks in grid
  gridSize = (int)ceil((float)n / blockSize);

  // Execute the kernel
  vecAdd<<<1, blockSize>>>(d_a, d_b, d_c, n);

  // Copy array back to host
  (void)hipMemcpy(h_c, d_c, bytes, hipMemcpyDeviceToHost);

  // Sum up vector c and print result divided by n, this should equal 1 within
  // error
  double sum = 0;
  for (i = 0; i < n; i++)
    sum += h_c[i];
  printf("Result (should be 1.0): %f\n", sum / n);

  // Release device memory
  (void)hipFree(d_a);
  (void)hipFree(d_b);
  (void)hipFree(d_c);

  // Release host memory
  free(h_a);
  free(h_b);
  free(h_c);

  return 0;
}
