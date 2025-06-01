#pragma once
#include <cuda_runtime.h>
#include "../problems.h"

namespace problems::gpu
{
	///\note This is only a 2d function
	__device__ float three_hump_camel(float* args, int n);

	///\note This is only a 2d function
	__device__ float six_hump_camel(float* args, int n);

	__device__ float dixon_price(float* args, int n);

	__device__ float rosenbrock(float* args, int n);
}
