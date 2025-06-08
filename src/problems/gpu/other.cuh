#pragma once
#include <cuda_runtime.h>
#include "../problems.h"

namespace problems::gpu
{
	///\note This is only a 2d function
	__host__ __device__ float beale(float* args, int n);
	
	///\note This is only a 2d function
	__host__ __device__ float branin(
		float* args,
		int    n,
		float  a,
		float  b,
		float  c,
		float  r,
		float  s,
		float  t
	);
	__host__ __device__ float branin2(float* args, int n);

	///\note This is only a 4d function
	///\warning This function has not been tested
	__host__ __device__ float colville(float* args, int n);

	///\note This is only a 1d function
	__host__ __device__ float forrester(float* args, int n);

	///\note This is only a 2d function
	__host__ __device__ float goldstein_price(float* args, int n);

	///\note All of the hartmann functions have not been tested
	__host__ __device__ float hartmann3d(float* args, int n);
	__host__ __device__ float hartmann4d(float* args, int n);
	__host__ __device__ float hartmann6d(float* args, int n);

	__host__ __device__ float permdb (float* args, int n, float b);
	__host__ __device__ float permdb2 (float* args, int n);

	///\note This function requires at least a 4d input
	__host__ __device__ float powell(float* args, int n);

	///\note This is only a 4d function
	__host__ __device__ float shekel(
		      float* args,
		      int    n,
		      int    m,
		const float* beta,
		const float* C
	);
	__host__ __device__ float shekel2(float* args, int n);

	__host__ __device__ float styblinsky_tang(float* args, int n);
}
