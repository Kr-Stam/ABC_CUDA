#pragma once

#include <stdbool.h>
#include <array>

namespace utils::arr
{
	void print_array_double(
		double* arr,
		int     n
	);

	void print_array_float(
		float* arr,
		int    n
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

	template <std::size_t N>
	constexpr std::array<char, N> str_to_arr(const char str[N]) {
		std::array<char, N> arr = { };

		int i = 0;
		while(str[i] != '\0' && i < N - 1)
		{
			arr[i] = str[i];
			i++;
		}

		arr[i] = '\0';

		return arr;
	}
}
