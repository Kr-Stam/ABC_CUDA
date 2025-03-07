#pragma once
#include <cuda_runtime.h>

namespace problems
{
	///\note This is only a 2d function
	__host__ __device__ double booth(double* args, int n);

	///\note This is only a 2d function
	__host__ __device__ double matyas(double* args, int n);

	///\note This is only a 2d function
	__host__ __device__ double mccormick(double* args, int n);

	__host__ __device__ double power_sum(double* args, int n, double* b);
	__host__ __device__ double power_sum2(double* args, int n);

	__host__ __device__ double zakharov(double* args, int n);
}
