/******************************************************************************
 * @file array.cpp                                                            *
 * @brief Util functions used for debugging arrays                            *
 * @author Kristijan Stameski                                                 *
 ******************************************************************************/

#include "array.hpp"
#include "random.hpp"
#include <stdio.h>

/*
 * @brief Initialize an integer array with random variable
 *
 * @param[out] arr               array to write the values to
 * @param[in]  n                 number of elements
 * @param[in]  lower_bound lower bound of generated values 
 * @param[in]  upper_bound upper bound of generated values 
 * */
void utils::arr::generate_array(
	double* arr,
	int     n,
	int     lower_bound, 
	int     upper_bound
)
{
	for(int i = 0; i < n; i++)
	{
		arr[i] =
			utils::random::rand_bounded_double(
				lower_bound, 
				upper_bound
			);
	}
}

/*
 * @brief Copy array A into array B
 *
 * @param[in]  A source array
 * @param[out] B destination array
 * @param[in]  n number of elements to copy
 * */
void utils::arr::arr_copy(
	double* A,
	double* B,
	int     n
)
{
	for(int i = 0; i < n; i++)
		B[i] = A[i];
}

/*
 * @brief Test equality of two arrays
 *
 * @param[in]  A
 * @param[out] B
 * @param[in]  n number of elements to compare
 * */
bool utils::arr::arr_equals(
	double* A,
	double* B, 
	int     n
)
{
	for(int i = 0; i < n; i++)
		if(A[i] != B[i]) return false;

	return true;
}

/*
 * @brief Print an array of doubles
 *
 * @param[in]  arr  array to be printed
 * @param[in]  n    number of elements
 * */
void utils::arr::print_array_double(
	double* arr,
	int     n
)
{
	for(int i = 0; i < n; i++)
	{
		printf("%.2f ", arr[i]);
		if(i % 10 == 0 && i != 0)
			printf("\n");
	}
	printf("\n");
}

/*
 * @brief Print an array of doubles
 *
 * @param[in]  arr  array to be printed
 * @param[in]  n    number of elements
 * */
void utils::arr::print_array_float(
	float* arr,
	int    n
)
{
	for(int i = 0; i < n; i++)
	{
		printf("%.2f ", arr[i]);
		if(i % 10 == 0 && i != 0)
			printf("\n");
	}
	printf("\n");
}

/*
 * @brief Print an array of integers
 *
 * @param[in]  arr  array to be printed
 * @param[in]  n    number of elements
 * */
void utils::arr::print_array_int(
	int* arr,
	int  n
)
{
	for(int i = 0; i < n; i++)
	{
		printf("%d ", arr[i]);
		if(i % 10 == 0 && i != 0)
			printf("\n");
	}
	printf("\n");
}
