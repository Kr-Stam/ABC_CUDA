#pragma once
#include <cuda_runtime.h>
#include "../problems.h"

namespace problems::gpu
{
	///\note This is only a 2d function
	__host__ __device__ float booth(float* args, int n);

	///\note This is only a 2d function
	__host__ __device__ float matyas(float* args, int n);

	///\note This is only a 2d function
	__host__ __device__ float mccormick(float* args, int n);

	__host__ __device__ float power_sum(float* args, int n, float* b);
	__host__ __device__ float power_sum2(float* args, int n);

	__host__ __device__ float zakharov(float* args, int n);
}
