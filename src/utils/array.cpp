#include "array.hpp"
#include "random.hpp"

#include <stdio.h>

void utils::array::generate_array(
		double* arr,
		int size,
		int lower_bound, 
		int upper_bound
)
{
	for(int i = 0; i < size; i++)
	{
		arr[i] = utils::random::rand_bounded_double(
				lower_bound, 
				upper_bound
		);
	}
}

void utils::array::arr_copy(
		double* A,
		double* B,
		int n
)
{
	for(int i = 0; i < n; i++)
		B[i] = A[i];
}

bool utils::array::arr_equals(
		double* A,
		double* B, 
		int n
)
{
	for(int i = 0; i < n; i++)
		if(A[i] != B[i])
			return false;

	return true;
}

void utils::array::print_array_double(
		double* arr,
		int size
)
{
	for(int i = 0; i < size; i++)
	{
		printf("%.2f ", arr[i]);
		if(i % 10 == 0 && i != 0)
			printf("\n");
	}
	printf("\n");
}

void utils::array::print_array_int(
		int* arr,
		int size
)
{
	for(int i = 0; i < size; i++)
	{
		printf("%d ", arr[i]);
		if(i % 10 == 0 && i != 0)
			printf("\n");
	}
	printf("\n");
}
