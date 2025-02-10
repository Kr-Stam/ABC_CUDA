#include <stdbool.h>
#include "sort.hpp"
#include "array.hpp"

using namespace utils::sort;

void utils::sort::bubble_sort(
		double arr[],
		int n
)
{
	bool swapped;
	for(int i = 0; i < n; i++)
	{
		swapped = false;
		for(int j = 0; j < n - 1; j++)
		{
			if(arr[j] > arr[j + 1])
			{
				swap(&arr[j], &arr[j+1]);
			}
			swapped = true;
		}
		if(!swapped) break;
	}
}

void utils::sort::selection_sort(
		double arr[],
		int n,
		int max_depth
)
{
	for(int i = 0; i < max_depth; i++)
	{
		for(int j = i + 1; j < n; j++)
		{
			if(arr[j] < arr[i])
				swap(&arr[i], &arr[j]);
		}
	}
}

void utils::sort::insertion_sort(
		double arr[],
		int n
)
{
	for(int i = 1; i < n; i++)
	{
		int j = i;
		while(j > 0 && arr[j] < arr[j -1])
		{
			swap(&arr[j], &arr[j-1]);
			j--;
		}
	}
}

