#pragma once
#include "../problems.h"

namespace problems::cpu
{
	float ackley(float* args, int n, float a, float b, float c);

	float ackley2(float* args, int n);

	///\note this function is only a 2d function
	float bukin6(float* args, int n);

	///\note this function in only a 2d function
	float cross_in_tray(float* args, int n);

	///\note this function in only a 2d function
	float drop_wave(float* args, int n);

	///\note this function in only a 2d function
	float eggholder(float* args, int n);

	///\note this function is only a 1d function
	float gramacy_and_lee(float* args, int n);

	float griewank(float* args, int n);

	///\note This only a 2d function
	float holder_table(float* args, int n);

	///\note A has dimensions nxm
	float langerman(float* args, int n, float* c, int m, float* A);

	float langerman2(float* args, int n);

	float levy(float* args, int n);

	///\note This is only a 2d function
	float levy13(float* args, int n);

	///\brief This is a fucntion with mutiple global
	///       maxima which is commonly used to test
	///       optimization functions
	float rastrigin(int a, float* x, int n);

	///brief This function is a wrapper around rastrigin
	///      in order to make it compatible to the problem
	///      definition
	float rastrigin2(float* args, int n);
	
	///\note This is only a 2d function
	float schaffer2(float* args, int n);

	///\note This is only a 2d function
	float schaffer4(float* args, int n);

	float schwefel(float* args, int n);

	///\note This is only a 2d function
	float shubert(float* args, int n);
}

