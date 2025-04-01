#include "problems_plate_shaped.cuh"
#include <math.h>

__device__ double problems::gpu::booth(double* args, int n)
{
	if(n < 2) return 0;

	double tmp1 = (args[0] + 2*args[1] - 7);
	double tmp2 = (2*args[0] + args[1] - 5);
	return tmp1*tmp1 + tmp2*tmp2;
}

__device__ double problems::gpu::matyas(double* args, int n)
{
	if (n < 2) return 0;

	return 0.26*(args[0]*args[0] + args[1]*args[1]) -
	       0.48*args[0]*args[1];
}

__device__ double problems::gpu::mccormick(double* args, int n)
{
	if (n < 2) return 0;

	double tmp = args[0]-args[1];
	return std::sin(args[0]+args[1])+tmp*tmp -
	       1.5*args[0] + 2.5*args[1] + 1;
}

__device__ double problems::gpu::power_sum(double* args, int n, double* b)
{
	double result = 0;
	for(int i = 0; i < n; i++)
	{
		double tmp = -b[i];
		for(int j = 0; j < n; j++)
		{
			tmp += std::pow(args[j], i+1);
		}
		result += tmp*tmp;
	}
	return result;
}

__device__ double problems::gpu::power_sum2(double* args, int n)
{
	double b[] = {8, 18, 44, 114};

	return problems::gpu::power_sum(args, n, b);
}

__device__ double problems::gpu::zakharov(double* args, int n)
{
	double sum1 = 0;
	double sum2 = 0;

	for(int i = 0; i < n; i++)
	{
		sum1 += args[i]*args[i];
		sum2 += 0.5*(i+1)*args[i];
	}

	sum2 = sum2 * sum2;
	return sum1 + sum2 + sum2*sum2;
}
