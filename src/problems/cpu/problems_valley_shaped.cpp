#include "problems_valley_shaped.h"
#include "../problems.h"
#include <math.h>

float problems::cpu::three_hump_camel(float* args, int n)
{
	float x2 = args[0]*args[0];

	return 2*x2 - 1.05*x2*x2 + x2*x2*x2/6.0 + args[0]*args[1] +
	       args[1]*args[1];
}

float problems::cpu::six_hump_camel(float* args, int n)
{
	float x2 = args[0]*args[0];
	float y2 = args[1]*args[1];

	return (4 - 2.1*x2 + x2*x2/3)*x2 + args[0]*args[1] +
	       (-4 + 4*y2)*y2;
}

float problems::cpu::dixon_price(float* args, int n)
{
	float result = 0;
	float tmp = (args[0] - 1);
	
	result += tmp*tmp;
	for(int i = 1; i < n; i++)
	{
		tmp = 2*args[i]*args[i] - args[i-1];
		result += (i+1)*tmp*tmp;
	}
	return result;
}

float problems::cpu::rosenbrock(float* args, int n)
{
	float result = 0;
	for(int i = 0; i < n - 1; i++)
	{
		float tmp;
		tmp     = args[i+1] - args[i]*args[i];
		result += 100*tmp*tmp;
		tmp     = args[i] - 1;
		result += tmp*tmp;
	}
	return result;
}
