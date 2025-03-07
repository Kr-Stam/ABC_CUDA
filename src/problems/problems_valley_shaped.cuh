#pragma once
#include <cuda_runtime.h>

namespace problems
{
	///\note This is only a 2d function
	__host__ __device__ double three_hump_camel(double* args, int n);

	///\note This is only a 2d function
	__host__ __device__ double six_hump_camel(double* args, int n);

	__host__ __device__ double dixon_price(double* args, int n);

	__host__ __device__ double rosenbrock(double* args, int n);
}
