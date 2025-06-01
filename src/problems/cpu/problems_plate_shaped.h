#pragma once
#include "../problems.h"

namespace problems::cpu
{
	///\note This is only a 2d function
	float booth(float* args, int n);

	///\note This is only a 2d function
	float matyas(float* args, int n);

	///\note This is only a 2d function
	float mccormick(float* args, int n);

	float power_sum(float* args, int n, float* b);
	float power_sum2(float* args, int n);

	float zakharov(float* args, int n);
}
