#include "problems_steep_ridges.h"
#include <math.h>

double problems::cpu::dejong5(double* args, int n)
{
	double result = 0.002;
	for(int i = 0; i < 25; i++)
	{
		//this calculation is used instead of the array
		double a1 = (double) (i % 5 - 2) * 16;
		double a2 = (double) (i / 5 - 2) * 16;
		result += 1.0/(i + std::pow(args[0] - a1, 6) +
		          std::pow(args[1] - a2, 6));
	}
	return 1.0 / result;
}

double problems::cpu::easom(double* args, int n)
{
	if(n < 2) return 0;

	double tmp1 = args[0] - M_PI;
	double tmp2 = args[1] - M_PI;
	return -std::cos(args[0])*std::cos(args[1])*
	       std::pow(M_E, -tmp1*tmp1 - tmp2*tmp2);
}


double problems::cpu::michalewicz(double* args, int n, int m)
{
	double result = 0;

	for(int i = 0; i < n; i++)
	{
		result -= std::sin(args[i])*std::pow(std::sin((i+1)*args[i]*args[i]/M_PI), 2*m);
	}

	return result;
}

double problems::cpu::michalewicz2(double* args, int n)
{
	return michalewicz(args, n, 10);
}
