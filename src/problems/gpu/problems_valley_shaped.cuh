#pragma once
#include <cuda_runtime.h>
#include "../problems.h"

namespace problems::gpu
{
	///\note This is only a 2d function
	__device__ double three_hump_camel(double* args, int n);

	///\note This is only a 2d function
	__device__ double six_hump_camel(double* args, int n);

	__device__ double dixon_price(double* args, int n);

	__device__ double rosenbrock(double* args, int n);
}
