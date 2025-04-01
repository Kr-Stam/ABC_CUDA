#pragma once
#include <cuda_runtime.h>
#include "../problems.h"

namespace problems::gpu
{
	///\note This is only a 2d function
	__device__ double bohachevsky1(double* args, int n);

	///\note This is only a 2d function
	__device__ double bohachevsky2(double* args, int n);

	///\note This is only a 2d function
	__device__ double bohachevsky3(double* args, int n);

	__device__ double sphere(double* args, int n);

	__device__ double perm(double* args, int n, int b);
	__device__ double perm2(double* args, int n);

	__device__ double rotated_hyper_elipsoid(double* args, int n);

	__device__ double sum_of_different_powers(double* args, int n);

	__device__ double sum_squares(double* args, int n);
	
	__device__ double trid(double* args, int n);
};
