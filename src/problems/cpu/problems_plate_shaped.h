#pragma once
#include "../problems.h"

namespace problems::cpu
{
	///\note This is only a 2d function
	double booth(double* args, int n);

	///\note This is only a 2d function
	double matyas(double* args, int n);

	///\note This is only a 2d function
	double mccormick(double* args, int n);

	double power_sum(double* args, int n, double* b);
	double power_sum2(double* args, int n);

	double zakharov(double* args, int n);
}
