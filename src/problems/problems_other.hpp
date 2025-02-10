#pragma once

namespace problems
{
	///\note This is only a 2d function
	double beale(double* args, int n);

	
	///\note This is only a 2d function
	double branin(double* args, int n, double a, double b, double c, double r, double s, double t);
	double branin2(double* args, int n);

	///\note This is only a 4d function
	///\warning This function has not been tested
	double colville(double* args, int n);

	///\note This is only a 1d function
	double forrester(double* args, int n);

	///\note This is only a 2d function
	double goldstein_price(double* args, int n);

	///\note All of the hartmann functions have not been tested
	double hartmann3d(double* args, int n);
	double hartmann4d(double* args, int n);
	double hartmann6d(double* args, int n);

	double permdb (double* args, int n, double b);
	double permdb2 (double* args, int n);

	///\note This function requires at least a 4d input
	double powell(double* args, int n);

	///\note This is only a 4d function
	double shekel(double* args, int n, int m, const double* beta, const double* C);
	double shekel2(double* args, int n);

	double styblinsky_tang(double* args, int n);
}
