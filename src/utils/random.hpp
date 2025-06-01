#pragma once

namespace utils::random
{
	double rand_bounded_double(double lower_bound, double upper_bound);
	float rand_bounded_float(float lower_bound, float upper_bound);
	int rand_bounded_int(int lower_bound, int upper_bound);
	void seed_random();
}
