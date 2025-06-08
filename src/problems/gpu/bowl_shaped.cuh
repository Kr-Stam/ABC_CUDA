#pragma once
#include <cuda_runtime.h>
#include "../problems.h"

namespace problems::gpu
{
	///\note This is only a 2d function
	__host__ __device__ float bohachevsky1(float* args, int n);

	///\note This is only a 2d function
	__host__ __device__ float bohachevsky2(float* args, int n);

	///\note This is only a 2d function
	__host__ __device__ float bohachevsky3(float* args, int n);

	__host__ __device__ float sphere(float* args, int n);

	__host__ __device__ float perm(float* args, int n, int b);
	__host__ __device__ float perm2(float* args, int n);

	__host__ __device__ float rotated_hyper_elipsoid(float* args, int n);

	__host__ __device__ float sum_of_different_powers(float* args, int n);

	__host__ __device__ float sum_squares(float* args, int n);
	
	__host__ __device__ float trid(float* args, int n);
};
