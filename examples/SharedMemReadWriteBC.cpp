
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
#include <iostream>
#include <string>

#define SHARED_SIZE 32

__global__ void kernel(int offset)
{
 int out;
    __shared__ uint32_t sharedMem[SHARED_SIZE];
    if (threadIdx.x == 0){
        for (int i = 0; i < SHARED_SIZE; i++) sharedMem[i] = 0;
    }
    __syncthreads();
    // repeatedly read and write to shared memory
    uint32_t index = threadIdx.x * offset;
    for (int i = 0; i < 10000; i++)
    {
        sharedMem[index] += index * i;
        index += 32;
        index %= SHARED_SIZE;
    }
}


int main(int argc, char *argv[]) {

  int offset = 32;
  int blockSize = 32;
  int gridSize = (int)ceil((float)offset / blockSize);
  kernel<<<gridSize, blockSize>>>(offset);
  return 0;
}
