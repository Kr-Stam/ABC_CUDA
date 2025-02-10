namespace utils::sort
{
	void bubble_sort(double arr[], int n);
	void selection_sort(double arr[], int n, int max_depth);
	void insertion_sort(double arr[], int n);

	void quicksort_high(double* arr, int low, int high);
	void quicksort_low(double* arr, int low, int high);
	void quicksort_med(double* arr, int low, int high);

	void mixed_quicksort_high(double* arr, int low, int high);
	void mixed_quicksort_low(double* arr, int low, int high);
	void mixed_quicksort_med(double* arr, int low, int high);

	void top_down_merge_sort(double* A, double* B, int n);
	void bottom_up_merge_sort(double* A, double* B, int n);
	void mixed_top_down_merge_sort(double* A, double* B, int n);
	void mixed_bottom_up_merge_sort(double* A, double* B, int n);

	void inline swap(double* a, double *b)
	{
		double tmp = *a;
		*a = *b;
		*b = tmp;
	}
}
