#pragma once

namespace utils::random
{
	double rand_bounded_double(double lower_bound, double upper_bound);
	int rand_bounded_int(int lower_bound, int upper_bound);
	void seed_random();
}
