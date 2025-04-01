#pragma once

#include <stdbool.h>

namespace utils::array
{
	void print_array_double(
		double* arr,
		int     n
	);

	void print_array_int(
		int* arr,
		int  n
	);

	void generate_array(
		double* arr,
		int     n,
		int     lower_bound,
		int     upper_bound
	);

	bool arr_equals(
		double* A,
		double* B,
		int     n
	);

	void arr_copy(
		double* A,
		double* B,
		int     n
	);
}
