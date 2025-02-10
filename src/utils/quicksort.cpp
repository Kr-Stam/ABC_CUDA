#include "sort.hpp"

using namespace utils::sort;

//Lomuto
int partition_pivot_high(
	double arr[],
	int low,
	int high
) 
{
    double pivot = arr[high];  
    //double pivot = arr[low];   // Choose the last element as pivot
    int i = low;  // Index of smaller element

    for (int j = low; j < high - 1; j++) {
        if (arr[j] <= pivot) 
		{
            swap(&arr[i], &arr[j]);  // Swap elements
            i++;  // Increment index of smaller element
        }
    }
    swap(&arr[i], &arr[high]);  
    return i;  // Return the partition index
}
//Hoare
int partition_pivot_low(
	double arr[],
	int low, 
	int high
)
{
    double pivot = arr[low];  

    while (true)
	{
		while(arr[low] < pivot)
			low++;
		while(arr[high] > pivot)
			high--;

		if(low >= high)
			return high;

		swap(&arr[low], &arr[high]);  
    }
}

int partition_pivot_med(
	double arr[], 
	int low,
	int high
) 
{
	int mid = (low + high) / 2;
	if(arr[mid] < arr[low]) 
		swap(&arr[mid], &arr[low]);
	if(arr[high] < arr[low]) 
		swap(&arr[low], &arr[high]);
	if(arr[mid] < arr[high]) 
		swap(&arr[mid], &arr[high]);

	return partition_pivot_high(arr, low, high);
}

void utils::sort::quicksort_high(
	double arr[], 
	int low, 
	int high
) 
{
    if (low < high && low >= 0) 
	{
		int pi = partition_pivot_high(arr, low, high);
		quicksort_high(arr, low, pi - 1);
		quicksort_high(arr, pi + 1, high);
	}	
}

void utils::sort::quicksort_low(
	double arr[], 
	int low, 
	int high
)
{
    if (low >= 0 && high >= 0 && low < high) 
	{
        int pi = partition_pivot_low(arr, low, high);
        quicksort_low(arr, low, pi);
        quicksort_low(arr, pi + 1, high);
    }
}

void utils::sort::quicksort_med(
	double arr[],
	int low,
	int high
) 
{
    if (low >= 0 && high >= 0 && low < high) 
	{
        int pi = partition_pivot_med(arr, low, high);
        quicksort_low(arr, low, pi);
        quicksort_low(arr, pi + 1, high);
    }
}

void utils::sort::mixed_quicksort_high(
	double arr[],
	int low, 
	int high
) 
{
    while (low < high)
	{
        //int pi = partition_pivot_high(arr, low, high);
		if(high - low < 200)
		{
			insertion_sort(arr, high - low);
			break;
		}
		else
		{
			int pi = partition_pivot_high(arr, low, high);
			if(pi - low < high - pi)
			{
				mixed_quicksort_high(arr, low, pi - 1);
				low = pi + 1;
			}
			else
			{
				mixed_quicksort_high(arr, pi + 1, high);
				high = pi - 1;
			}
		}
    }
}

void utils::sort::mixed_quicksort_low(
	double arr[],
	int low,
	int high
)
{
    while (low < high && low >= 0) {
        int pi = partition_pivot_low(arr, low, high);
		if(high - low < 200)
		{
			insertion_sort(arr, pi - low);
			break;
		}
		else
		{
			if(pi - low < high - pi)
			{
				mixed_quicksort_low(arr, low, pi - 1);
				low = pi + 1;
			}
			else
			{
				mixed_quicksort_low(arr, pi + 1, high);
				high = pi - 1;
			}
		}
    }
}

void utils::sort::mixed_quicksort_med(
	double arr[],
	int low,
	int high
) 
{
    while (low < high && low >= 0) 
	{
        int pi = partition_pivot_med(arr, low, high);
		if(high - low < 200)
		{
			insertion_sort(arr, pi - low);
			break;
		}
		else
		{
			if(pi - low < high - pi)
			{
				mixed_quicksort_med(arr, low, pi - 1);
				low = pi + 1;
			}
			else
			{
				mixed_quicksort_med(arr, pi + 1, high);
				high = pi - 1;
			}
		}
    }
}

