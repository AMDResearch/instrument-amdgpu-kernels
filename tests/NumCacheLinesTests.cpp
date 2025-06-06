
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
#include <stdio.h>

// google test
#include <gtest/gtest.h>

#define N 10000
template<typename T>
__global__ void PlusOne(T* array, int stride){
  int index = stride * threadIdx.x;
  array[index] = array[index] + 1;
}

__global__ void IntPlusOne(int* array, int stride){
  int index = stride * threadIdx.x;
  array[index] = array[index] + 1;
}

__device__ uint32_t result;

TEST(MemCoalesingTest, Ints){
  int x[N] = {0};
  int* d_x;
  uint32_t h_result;
  (void)hipMalloc(&d_x, sizeof(int) * N);
  (void)hipMemcpy(d_x, x, sizeof(int)* N, hipMemcpyHostToDevice);

  int blocks = 1; int stride = 1;
  int threads = 32;   
  int expectedCacheLines = 1;
  IntPlusOne<<<blocks, threads >>>(d_x, stride);
  (void)hipDeviceSynchronize();
  (void)hipMemcpyFromSymbol(&h_result, result, sizeof(uint32_t), 0, hipMemcpyDeviceToHost);
  EXPECT_EQ(h_result, expectedCacheLines);


  blocks = 1; threads = 32; stride = 2;
  expectedCacheLines = 2;
  IntPlusOne<<<blocks, threads >>>(d_x, stride);
  (void)hipDeviceSynchronize();
  (void)hipMemcpyFromSymbol(&h_result, result, sizeof(uint32_t), 0, hipMemcpyDeviceToHost);
  EXPECT_EQ(h_result, expectedCacheLines);

  blocks = 1; threads = 32; stride = 32;
  expectedCacheLines = 32;
  IntPlusOne<<<blocks, threads >>>(d_x, stride);
  (void)hipDeviceSynchronize();
  (void)hipMemcpyFromSymbol(&h_result, result, sizeof(uint32_t), 0, hipMemcpyDeviceToHost);
  EXPECT_EQ(h_result, expectedCacheLines);

  blocks = 1; threads = 16; stride = 2;
  expectedCacheLines = 1;  
  IntPlusOne<<<blocks, threads >>>(d_x, stride);
  (void)hipDeviceSynchronize();
  (void)hipMemcpyFromSymbol(&h_result, result, sizeof(uint32_t), 0, hipMemcpyDeviceToHost);
  EXPECT_EQ(h_result, expectedCacheLines);  

  // blocks = 32; threads = 1; stride = 32;
  // expectedCacheLines = 1;  
  // IntPlusOne<<<blocks, threads >>>(d_x, stride);
  // (void)hipDeviceSynchronize();
  // (void)hipMemcpyFromSymbol(&h_result, result, sizeof(uint32_t), 0, hipMemcpyDeviceToHost);
  // EXPECT_EQ(h_result, expectedCacheLines);    

  (void)hipMemcpy(x, d_x, sizeof(int) * N, hipMemcpyDeviceToHost);
  (void)hipFree(x);
}


TEST(MemCoalesingTest, Chars){
  char x[N] = {0};
  char* d_x;
  uint32_t h_result;
  (void)hipMalloc(&d_x, sizeof(char) * N);
  (void)hipMemcpy(d_x, x, sizeof(char)* N, hipMemcpyHostToDevice);

  int blocks = 1; int threads = 32; int stride = 1;
  int expectedCacheLines = 1;
  PlusOne<char><<<blocks, threads >>>(d_x, stride);
  (void)hipDeviceSynchronize();
  (void)hipMemcpyFromSymbol(&h_result, result, sizeof(uint32_t), 0, hipMemcpyDeviceToHost);
  EXPECT_EQ(h_result, expectedCacheLines); 

  blocks = 1; threads = 32; stride = 2;
  expectedCacheLines = 1;
  PlusOne<char><<<blocks, threads >>>(d_x, stride);
  (void)hipDeviceSynchronize();
  (void)hipMemcpyFromSymbol(&h_result, result, sizeof(uint32_t), 0, hipMemcpyDeviceToHost);
  EXPECT_EQ(h_result, expectedCacheLines); 

  blocks = 1; threads = 32; stride = 4;
  expectedCacheLines = 1;
  PlusOne<char><<<blocks, threads >>>(d_x, stride);
  (void)hipDeviceSynchronize();
  (void)hipMemcpyFromSymbol(&h_result, result, sizeof(uint32_t), 0, hipMemcpyDeviceToHost);
  EXPECT_EQ(h_result, expectedCacheLines); 

  blocks = 1; threads = 32; stride = 8;
  expectedCacheLines = 2;
  PlusOne<char><<<blocks, threads >>>(d_x, stride);
  (void)hipDeviceSynchronize();
  (void)hipMemcpyFromSymbol(&h_result, result, sizeof(uint32_t), 0, hipMemcpyDeviceToHost);
  EXPECT_EQ(h_result, expectedCacheLines); 

  blocks = 1; threads = 16; stride = 8;
  expectedCacheLines = 1;
  PlusOne<char><<<blocks, threads >>>(d_x, stride);
  (void)hipDeviceSynchronize();
  (void)hipMemcpyFromSymbol(&h_result, result, sizeof(uint32_t), 0, hipMemcpyDeviceToHost);
  EXPECT_EQ(h_result, expectedCacheLines);   

  blocks = 1; threads = 32; stride = 128;
  expectedCacheLines = 32;
  PlusOne<char><<<blocks, threads >>>(d_x, stride);
  (void)hipDeviceSynchronize();
  (void)hipMemcpyFromSymbol(&h_result, result, sizeof(uint32_t), 0, hipMemcpyDeviceToHost);
  EXPECT_EQ(h_result, expectedCacheLines); 

  blocks = 1; threads = 8; stride = 16;
  expectedCacheLines = 1;
  PlusOne<char><<<blocks, threads >>>(d_x, stride);
  (void)hipDeviceSynchronize();
  (void)hipMemcpyFromSymbol(&h_result, result, sizeof(uint32_t), 0, hipMemcpyDeviceToHost);
  EXPECT_EQ(h_result, expectedCacheLines); 
  
  blocks = 1; threads = 32; stride = 9;
  expectedCacheLines = 3;
  PlusOne<char><<<blocks, threads >>>(d_x, stride);
  (void)hipDeviceSynchronize();
  (void)hipMemcpyFromSymbol(&h_result, result, sizeof(uint32_t), 0, hipMemcpyDeviceToHost);
  EXPECT_EQ(h_result, expectedCacheLines);

  //Needs 1 bytes of contiguous memory
  blocks = 5; threads = 1; stride = 1;
  expectedCacheLines = 1;
  PlusOne<char><<<blocks, threads >>>(d_x, stride);
  (void)hipDeviceSynchronize();
  (void)hipMemcpyFromSymbol(&h_result, result, sizeof(uint32_t), 0, hipMemcpyDeviceToHost);
  EXPECT_EQ(h_result, expectedCacheLines);

  blocks = 5; threads = 2; stride = 16;
  expectedCacheLines = 1;
  PlusOne<char><<<blocks, threads >>>(d_x, stride);
  (void)hipDeviceSynchronize();
  (void)hipMemcpyFromSymbol(&h_result, result, sizeof(uint32_t), 0, hipMemcpyDeviceToHost);
  EXPECT_EQ(h_result, expectedCacheLines);  

  blocks = 5; threads = 8; stride = 4;
  expectedCacheLines = 1;
  PlusOne<char><<<blocks, threads >>>(d_x, stride);
  (void)hipDeviceSynchronize();
  (void)hipMemcpyFromSymbol(&h_result, result, sizeof(uint32_t), 0, hipMemcpyDeviceToHost);
  EXPECT_EQ(h_result, expectedCacheLines);  

  blocks = 5; threads = 8; stride = 4;
  expectedCacheLines = 1;
  PlusOne<char><<<blocks, threads >>>(d_x, stride);
  (void)hipDeviceSynchronize();
  (void)hipMemcpyFromSymbol(&h_result, result, sizeof(uint32_t), 0, hipMemcpyDeviceToHost);
  EXPECT_EQ(h_result, expectedCacheLines);  

  blocks = 1; threads = 9; stride = 17;
  expectedCacheLines = 2;
  PlusOne<char><<<blocks, threads >>>(d_x, stride);
  (void)hipDeviceSynchronize();
  (void)hipMemcpyFromSymbol(&h_result, result, sizeof(uint32_t), 0, hipMemcpyDeviceToHost);
  EXPECT_EQ(h_result, expectedCacheLines);  

  blocks = 1; threads = 8; stride = 50;
  expectedCacheLines = 3;
  PlusOne<char><<<blocks, threads >>>(d_x, stride);
  (void)hipDeviceSynchronize();
  (void)hipMemcpyFromSymbol(&h_result, result, sizeof(uint32_t), 0, hipMemcpyDeviceToHost);
  EXPECT_EQ(h_result, expectedCacheLines);
  
  (void)hipMemcpy(x, d_x, sizeof(char) * N, hipMemcpyDeviceToHost);
  (void)hipFree(x);
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
