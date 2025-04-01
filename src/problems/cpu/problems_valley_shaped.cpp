#include "problems_valley_shaped.h"
#include "../problems.h"
#include <math.h>

double problems::cpu::three_hump_camel(double* args, int n)
{
	double x2 = args[0]*args[0];

	return 2*x2 - 1.05*x2*x2 + x2*x2*x2/6.0 + args[0]*args[1] +
	       args[1]*args[1];
}

double problems::cpu::six_hump_camel(double* args, int n)
{
	double x2 = args[0]*args[0];
	double y2 = args[1]*args[1];

	return (4 - 2.1*x2 + x2*x2/3)*x2 + args[0]*args[1] +
	       (-4 + 4*y2)*y2;
}

double problems::cpu::dixon_price(double* args, int n)
{
	double result = 0;
	double tmp = (args[0] - 1);
	
	result += tmp*tmp;
	for(int i = 1; i < n; i++)
	{
		tmp = 2*args[i]*args[i] - args[i-1];
		result += (i+1)*tmp*tmp;
	}
	return result;
}

double problems::cpu::rosenbrock(double* args, int n)
{
	double result = 0;
	for(int i = 0; i < n - 1; i++)
	{
		double tmp;
		tmp     = args[i+1] - args[i]*args[i];
		result += 100*tmp*tmp;
		tmp     = args[i] - 1;
		result += tmp*tmp;
	}
	return result;
}
