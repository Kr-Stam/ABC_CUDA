#pragma once
#include "../problems.h"

namespace problems::cpu
{
	double ackley(double* args, int n, double a, double b, double c);

	double ackley2(double* args, int n);

	///\note this function is only a 2d function
	double bukin6(double* args, int n);

	///\note this function in only a 2d function
	double cross_in_tray(double* args, int n);

	///\note this function in only a 2d function
	double drop_wave(double* args, int n);

	///\note this function in only a 2d function
	double eggholder(double* args, int n);

	///\note this function is only a 1d function
	double gramacy_and_lee(double* args, int n);

	double griewank(double* args, int n);

	///\note This only a 2d function
	double holder_table(double* args, int n);

	///\note A has dimensions nxm
	double langerman(double* args, int n, double* c, int m, double* A);

	double langerman2(double* args, int n);

	double levy(double* args, int n);

	///\note This is only a 2d function
	double levy13(double* args, int n);

	///\brief This is a fucntion with mutiple global
	///       maxima which is commonly used to test
	///       optimization functions
	double rastrigin(int a, double* x, int n);

	///brief This function is a wrapper around rastrigin
	///      in order to make it compatible to the problem
	///      definition
	double rastrigin2(double* args, int n);
	
	///\note This is only a 2d function
	double schaffer2(double* args, int n);

	///\note This is only a 2d function
	double schaffer4(double* args, int n);

	double schwefel(double* args, int n);

	///\note This is only a 2d function
	double shubert(double* args, int n);
}

