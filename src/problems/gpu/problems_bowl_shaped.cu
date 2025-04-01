#include "problems_bowl_shaped.cuh"
#include <math.h>
#include <stdio.h>

__device__ double problems::gpu::bohachevsky1(double* args, int n)
{
	if(n < 2) return 0;

	return args[0]*args[0] + 2*args[1]*args[1] -
		   0.3*std::cos(3*M_PI*args[0]) -
		   0.4*std::cos(4*M_PI*args[1]) + 0.7;
}

__device__ double problems::gpu::bohachevsky2(double* args, int n)
{
	if(n < 2) return 0;

	return args[0]*args[0] + 2*args[1]*args[1] -
		   0.3*std::cos(3*M_PI*args[0])*std::cos(4*M_PI*args[1]) + 0.3;
}

__device__ double problems::gpu::bohachevsky3(double* args, int n)
{
	if(n < 2) return 0;

	return args[0]*args[0] + 2*args[1]*args[1] -
		   0.3*std::cos(3*M_PI*args[0] + 4*M_PI*args[1]) + 0.3;
}

__device__ double problems::gpu::sphere(double* args, int n)
{
	double result = 0;
	for(int i = 0; i < n; i++)
	{
		result += args[i] * args[i];
	}
	return result;
}

__device__ double problems::gpu::perm(double* args, int n, int b)
{
	double result = 0;
	for(int i = 0; i < n; i++)
	{
		double tmp = 0;
		for(int j = 0; j < n; j++)
		{
			tmp += (j+b)*(std::pow(args[j], (double)i+1) - std::pow(1.0/(j+1), i+1));
		}
		result += tmp*tmp;
	}
	printf("x1: %.2f, x2: %.2f, result: %.2f, n: %d\n", args[0], args[1], result, n);
	//x1: -2.44, x2: -3.97, result: inf, n: 2
	return result;
}

__device__ double problems::gpu::perm2(double* args, int n)
{
	return perm(args, n, 10);
}

__device__ double problems::gpu::rotated_hyper_elipsoid(double* args, int n)
{
	double result = 0;
	for(int i = 0; i < n; i++)
	{
		for(int j = 0; j <= i; j++)
		{
			result += args[j] * args[j];
		}
	}

	//result = 2*args[0]*args[0] + args[1]*args[1];
	return result;
}

__device__ double problems::gpu::sum_of_different_powers(double* args, int n)
{
	double result = 0;
	for(int i = 0; i < n; i++)
	{
		result += std::pow(std::abs(args[i]), i+2);
	}
	return result;
}

__device__ double problems::gpu::sum_squares(double* args, int n)
{
	double result = 0;
	for(int i = 0; i < n; i++)
	{
		result += (i+1)*args[i]*args[i];
	}
	return result;
}

__device__ double problems::gpu::trid(double* args, int n)
{
	double result = 0;
	double tmp = args[0] - 1;
	result += tmp*tmp;

	for(int i = 1; i < n; i++)
	{
		double tmp = args[i] - 1;
		result += tmp*tmp - args[i]*args[i-1];
	}
	return result;
}
