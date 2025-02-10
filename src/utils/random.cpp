#include "random.hpp"
#include "stdlib.h"
#include "time.h"

void utils::random::seed_random()
{
	srand(time(NULL));
}

double utils::random::rand_bounded_double(
		double lower_bound,
		double upper_bound
)
{
	double range = upper_bound - lower_bound;
	double div = RAND_MAX / range;
	if (div < -1) div *= -1;

	return lower_bound + rand() / div;
}

int utils::random::rand_bounded_int(int lower_bound, int upper_bound)
{
	int div = RAND_MAX / (upper_bound - lower_bound);
	return lower_bound + rand() / div;
}
