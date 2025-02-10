#include <stdbool.h>

namespace utils::array
{
	void print_array_double(double* arr, int size);
	void print_array_int(int* arr, int size);
	void generate_array(double* arr, int size, int lower_bound, int upper_bound);

	bool arr_equals(double* A, double* B, int n);
	void arr_copy(double* A, double* B, int n);
}
