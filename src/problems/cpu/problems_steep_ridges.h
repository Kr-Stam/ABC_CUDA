#pragma once
#include "../problems.h"

namespace problems::cpu
{
	double dejong5(double* args, int n);

	///\note This is only a 2d function
	double easom(double* args, int n);

	double michalewicz(double* args, int n, int m);
	double michalewicz2(double* args, int n);
}
