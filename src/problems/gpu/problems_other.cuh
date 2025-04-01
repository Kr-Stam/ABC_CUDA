#pragma once
#include <cuda_runtime.h>
#include "../problems.h"

namespace problems::gpu
{
	///\note This is only a 2d function
	__device__ double beale(double* args, int n);
	
	///\note This is only a 2d function
	__device__ double branin(double* args, int n, double a, double b, double c, double r, double s, double t);
	__device__ double branin2(double* args, int n);

	///\note This is only a 4d function
	///\warning This function has not been tested
	__device__ double colville(double* args, int n);

	///\note This is only a 1d function
	__device__ double forrester(double* args, int n);

	///\note This is only a 2d function
	__device__ double goldstein_price(double* args, int n);

	///\note All of the hartmann functions have not been tested
	__device__ double hartmann3d(double* args, int n);
	__device__ double hartmann4d(double* args, int n);
	__device__ double hartmann6d(double* args, int n);

	__device__ double permdb (double* args, int n, double b);
	__device__ double permdb2 (double* args, int n);

	///\note This function requires at least a 4d input
	__device__ double powell(double* args, int n);

	///\note This is only a 4d function
	__device__ double shekel(double* args, int n, int m, const double* beta, const double* C);
	__device__ double shekel2(double* args, int n);

	__device__ double styblinsky_tang(double* args, int n);
}
