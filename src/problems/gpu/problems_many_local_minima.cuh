#pragma once
#include <cuda_runtime.h>
#include "../problems.h"

namespace problems::gpu
{
	__device__ double ackley(double* args, int n, double a, double b, double c);

	__device__ double ackley2(double* args, int n);

	///\note this function is only a 2d function
	__device__ double bukin6(double* args, int n);

	///\note this function in only a 2d function
	__host__ __device__ double cross_in_tray(double* args, int n);

	///\note this function in only a 2d function
	__device__ double drop_wave(double* args, int n);

	///\note this function in only a 2d function
	__device__ double eggholder(double* args, int n);

	///\note this function is only a 1d function
	__device__ double gramacy_and_lee(double* args, int n);

	__device__ double griewank(double* args, int n);

	///\note This only a 2d function
	__device__ double holder_table(double* args, int n);

	///\note A has dimensions nxm
	__device__ double langerman(double* args, int n, double* c, int m, double* A);

	__device__ double langerman2(double* args, int n);

	__device__ double levy(double* args, int n);

	///\note This is only a 2d function
	__device__ double levy13(double* args, int n);

	///\brief This is a fucntion with mutiple global
	///       maxima which is commonly used to test
	///       optimization functions
	__device__ double rastrigin(int a, double* x, int n);

	///brief This function is a wrapper around rastrigin
	///      in order to make it compatible to the problem
	///      definition
	__device__ double rastrigin2(double* args, int n);
	
	///\note This is only a 2d function
	__device__ double schaffer2(double* args, int n);

	///\note This is only a 2d function
	__device__ double schaffer4(double* args, int n);

	__device__ double schwefel(double* args, int n);

	///\note This is only a 2d function
	__device__ double shubert(double* args, int n);
}

