#pragma once
#include "../problems.h"

namespace problems::cpu
{
	///\note This is only a 2d function
	double bohachevsky1(double* args, int n);

	///\note This is only a 2d function
	double bohachevsky2(double* args, int n);

	///\note This is only a 2d function
	double bohachevsky3(double* args, int n);

	double sphere(double* args, int n);

	double perm(double* args, int n, int b);
	double perm2(double* args, int n);

	double rotated_hyper_elipsoid(double* args, int n);

	double sum_of_different_powers(double* args, int n);

	double sum_squares(double* args, int n);
	
	double trid(double* args, int n);
};
