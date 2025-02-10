#include "sort.hpp"

void top_down_merge(
	double B[],
	int left,
	int middle,
	int right,
	double A[]
)
{
	int i = left;
	int j = middle;

	for(int k = left; k < right; k++)
	{
		if(i < middle && (j >= right || A[i] <= A[j]))
		{
			B[k] = A[i];
			i++;
		}
		else
		{
			B[k] = A[j];
			j++;
		}
	}
}
void top_down_split_merge(
	double A[],
	int left,
	int right,
	double B[]
)
{
	if(right - left <= 1)
		return;

	int middle = (right + left) / 2;
	top_down_split_merge(A, left, middle, B);
	top_down_split_merge(A, middle, right, B);

	top_down_merge(B, left, middle, right, A);
}
void utils::sort::top_down_merge_sort(
		double A[], 
		double B[],
		int n
)
{
	for(int i = 0; i < n; i++)
		B[i] = A[i];

	top_down_split_merge(A, 0, n, B);
}

void top_down_split_merge_mixed(
	double A[],
	int left,
	int right, 
	double B[]
)
{
	int range = right - left;
	if(range <= 100)
	{
		utils::sort::insertion_sort(A, range);
	}
	else
	{
		int middle = (right + left) / 2;
		top_down_split_merge(A, left, middle, B);
		top_down_split_merge(A, middle, right, B);

		top_down_merge(B, left, middle, right, A);
	}
}
void utils::sort::mixed_top_down_merge_sort(
		double A[],
		double B[],
		int n
)
{
	for(int i = 0; i < n; i++)
		B[i] = A[i];

	top_down_split_merge_mixed(A, 0, n, B);
}

void bottom_up_merge(
	double A[],
	int left, 
	int right,
	int end,
	double B[]
)
{
	int i = left;
	int j = right;

	for(int k = left; k < end; k++)
	{
		if(i < right && (j >= end || A[i] <= A[j]))
		{
			B[k] = A[i];
			i++;
		}
		else
		{
			B[k] = A[j];
			j++;
		}
	}
}
double min(double a, double b)
{
	if(a < b) 
		return a;
	else
		return b;
}
void utils::sort::bottom_up_merge_sort(
	double A[],
	double B[],
	int n
)
{
	for(int width = 1; width < n; width *= 2)
	{
		for(int i = 0; i < n; i += 2*width)
		{
			bottom_up_merge(A, i, min(i+width, n), min(i+2*width, n), B);
		}

		double* tmp = A;
		A = B;
		B = tmp;
	}
}

void utils::sort::mixed_bottom_up_merge_sort(
	double A[],
	double B[],
	int n
)
{
	for(int width = 1; width < n; width *= 2)
	{
		if(width > 50)
		{
			for(int i = 0; i < n; i += 2*width)
			{
				bottom_up_merge(A, i, min(i+width, n), min(i+2*width, n), B);
			}

			double* tmp = A;
			A = B;
			B = tmp;
		}
		else
		{
			for(int i = 0; i < n; i += 2*width)
			{
				insertion_sort(&A[i], min(i+width, n));
			}
		}
	}
}
