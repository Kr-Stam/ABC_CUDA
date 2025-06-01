#pragma once
#include "../problems.h"

namespace problems::cpu
{
	///\note This is only a 2d function
	float beale(float* args, int n);
	
	///\note This is only a 2d function
	float branin(float* args, int n, float a, float b, float c, float r, float s, float t);
	float branin2(float* args, int n);

	///\note This is only a 4d function
	///\warning This function has not been tested
	float colville(float* args, int n);

	///\note This is only a 1d function
	float forrester(float* args, int n);

	///\note This is only a 2d function
	float goldstein_price(float* args, int n);

	///\note All of the hartmann functions have not been tested
	float hartmann3d(float* args, int n);
	float hartmann4d(float* args, int n);
	float hartmann6d(float* args, int n);

	float permdb (float* args, int n, float b);
	float permdb2 (float* args, int n);

	///\note This function requires at least a 4d input
	float powell(float* args, int n);

	///\note This is only a 4d function
	float shekel(float* args, int n, int m, const float* beta, const float* C);
	float shekel2(float* args, int n);

	float styblinsky_tang(float* args, int n);
}
