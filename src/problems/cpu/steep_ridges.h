#pragma once
#include "../problems.h"

namespace problems::cpu
{
	float dejong5(float* args, int n);

	///\note This is only a 2d function
	float easom(float* args, int n);

	float michalewicz(float* args, int n, int m);
	float michalewicz2(float* args, int n);
}
