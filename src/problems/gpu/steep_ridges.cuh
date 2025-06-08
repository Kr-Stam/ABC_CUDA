#pragma once
#include <cuda_runtime.h>
#include "../problems.h"

namespace problems::gpu
{
	__host__ __device__ float dejong5(float* args, int n);

	///\note This is only a 2d function
	__host__ __device__ float easom(float* args, int n);

	__host__ __device__ float michalewicz(float* args, int n, int m);
	__host__ __device__ float michalewicz2(float* args, int n);
}
