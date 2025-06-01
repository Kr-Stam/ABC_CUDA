#pragma once
#include <cuda_runtime.h>
#include "../problems.h"

namespace problems::gpu
{
	__device__ float ackley(float* args, int n, float a, float b, float c);

	__device__ float ackley2(float* args, int n);

	///\note this function is only a 2d function
	__device__ float bukin6(float* args, int n);

	///\note this function in only a 2d function
	__host__ __device__ float cross_in_tray(float* args, int n);

	///\note this function in only a 2d function
	__device__ float drop_wave(float* args, int n);

	///\note this function in only a 2d function
	__device__ float eggholder(float* args, int n);

	///\note this function is only a 1d function
	__device__ float gramacy_and_lee(float* args, int n);

	__device__ float griewank(float* args, int n);

	///\note This only a 2d function
	__device__ float holder_table(float* args, int n);

	///\note A has dimensions nxm
	__device__ float langerman(float* args, int n, float* c, int m, float* A);

	__device__ float langerman2(float* args, int n);

	__device__ float levy(float* args, int n);

	///\note This is only a 2d function
	__device__ float levy13(float* args, int n);

	///\brief This is a fucntion with mutiple global
	///       maxima which is commonly used to test
	///       optimization functions
	__device__ float rastrigin(int a, float* x, int n);

	///brief This function is a wrapper around rastrigin
	///      in order to make it compatible to the problem
	///      definition
	__device__ float rastrigin2(float* args, int n);
	
	///\note This is only a 2d function
	__device__ float schaffer2(float* args, int n);

	///\note This is only a 2d function
	__device__ float schaffer4(float* args, int n);

	__device__ float schwefel(float* args, int n);

	///\note This is only a 2d function
	__device__ float shubert(float* args, int n);
}

