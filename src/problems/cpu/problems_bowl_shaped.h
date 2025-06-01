#pragma once
#include "../problems.h"

namespace problems::cpu
{
	///\note This is only a 2d function
	float bohachevsky1(float* args, int n);

	///\note This is only a 2d function
	float bohachevsky2(float* args, int n);

	///\note This is only a 2d function
	float bohachevsky3(float* args, int n);

	float sphere(float* args, int n);

	float perm(float* args, int n, int b);
	float perm2(float* args, int n);

	float rotated_hyper_elipsoid(float* args, int n);

	float sum_of_different_powers(float* args, int n);

	float sum_squares(float* args, int n);
	
	float trid(float* args, int n);
};
