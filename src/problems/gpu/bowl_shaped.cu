#include "bowl_shaped.cuh"
#include <math.h>

__host__ __device__ float problems::gpu::bohachevsky1(float* args, int n)
{
	if(n < 2) return 0;

	return args[0]*args[0] + 2*args[1]*args[1] -
		   0.3*std::cos(3*M_PI*args[0]) -
		   0.4*std::cos(4*M_PI*args[1]) + 0.7;
}

__host__ __device__ float problems::gpu::bohachevsky2(float* args, int n)
{
	if(n < 2) return 0;

	return args[0]*args[0] + 2*args[1]*args[1] -
		   0.3*std::cos(3*M_PI*args[0])*std::cos(4*M_PI*args[1]) + 0.3;
}

__host__ __device__ float problems::gpu::bohachevsky3(float* args, int n)
{
	if(n < 2) return 0;

	return args[0]*args[0] + 2*args[1]*args[1] -
		   0.3*std::cos(3*M_PI*args[0] + 4*M_PI*args[1]) + 0.3;
}

__host__ __device__ float problems::gpu::sphere(float* args, int n)
{
	float result = 0;

	for(int i = 0; i < n; i++)
		result += args[i] * args[i];

	return result;
}

__host__ __device__ float problems::gpu::perm(float* args, int n, int b)
{
	float result = 0;
	for(int i = 0; i < n; i++)
	{
		float tmp = 0;
		for(int j = 0; j < n; j++)
			tmp += (j+b)*(std::pow(args[j], (float)i+1) -
					      std::pow(1.0/(j+1), i+1));

		result += tmp*tmp;
	}

	return result;
}

__host__ __device__ float problems::gpu::perm2(float* args, int n)
{
	return perm(args, n, 10);
}

__host__ __device__ float problems::gpu::rotated_hyper_elipsoid(float* args, int n)
{
	float result = 0;

	for(int i = 0; i < n; i++)
		for(int j = 0; j <= i; j++)
			result += args[j] * args[j];

	return result;
}

__host__ __device__ float problems::gpu::sum_of_different_powers(float* args, int n)
{
	float result = 0;
	for(int i = 0; i < n; i++)
		result += std::pow(std::abs(args[i]), i+2);

	return result;
}

__host__ __device__ float problems::gpu::sum_squares(float* args, int n)
{
	float result = 0;
	for(int i = 0; i < n; i++)
		result += (i+1)*args[i]*args[i];

	return result;
}

__host__ __device__ float problems::gpu::trid(float* args, int n)
{
	float result = 0;
	float tmp = args[0] - 1;
	result += tmp*tmp;

	for(int i = 1; i < n; i++)
	{
		float tmp = args[i] - 1;
		result += tmp*tmp - args[i]*args[i-1];
	}
	return result;
}
