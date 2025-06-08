#include "steep_ridges.cuh"
#include <math.h>

__host__ __device__ float problems::gpu::dejong5(float* args, int n)
{
	float result = 0.002;
	for(int i = 0; i < 25; i++)
	{
		float a1 = (float) (i % 5 - 2) * 16;
		float a2 = (float) (i / 5 - 2) * 16;
		result += 1.0/(i + std::pow(args[0] - a1, 6) +
		          std::pow(args[1] - a2, 6));
	}
	return 1.0 / result;
}

__host__ __device__ float problems::gpu::easom(float* args, int n)
{
	if(n < 2) return 0;

	float tmp1 = args[0] - M_PI;
	float tmp2 = args[1] - M_PI;
	return -std::cos(args[0])*std::cos(args[1])*
	       std::pow(M_E, -tmp1*tmp1 - tmp2*tmp2);
}


__host__ __device__ float problems::gpu::michalewicz(float* args, int n, int m)
{
	float result = 0;

	for(int i = 0; i < n; i++)
	{
		result -= std::sin(args[i])*std::pow(std::sin((i+1)*args[i]*args[i]/M_PI), 2*m);
	}

	return result;
}

__host__ __device__ float problems::gpu::michalewicz2(float* args, int n)
{
	return michalewicz(args, n, 10);
}
