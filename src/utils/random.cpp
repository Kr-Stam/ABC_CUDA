/******************************************************************************
 * @file random.cpp                                                           *
 * @brief Util functions used for generating random numbers                   *
 * @author Kristijan Stameski                                                 *
 ******************************************************************************/

#include "random.hpp"
#include "stdlib.h"
#include "time.h"

void utils::random::seed_random()
{
	srand(time(NULL));
}

/*
 * @brief Generates a random double in the range [lower_bound, upper_bound]
 * @param[in] lower_bound
 * @param[in] upper_bound
 * @note The range is inclusive but it is highly unlikely to hit the upper
 *       range because of the nature of floating point numbers
 */
double utils::random::rand_bounded_double(
		double lower_bound,
		double upper_bound
)
{
	double range = upper_bound - lower_bound;
	double div = RAND_MAX / range;

	return lower_bound + rand() / div;
}

/*
 * @brief Generates a random float in the range [lower_bound, upper_bound]
 * @param[in] lower_bound
 * @param[in] upper_bound
 * @note The range is inclusive but it is highly unlikely to hit the upper
 *       range because of the nature of floating point numbers
 */
float utils::random::rand_bounded_float(
		float lower_bound,
		float upper_bound
)
{
	float range = upper_bound - lower_bound;
	float div = RAND_MAX / range;

	return lower_bound + rand() / div;
}

/*
 * @brief Generates a random int32_t in the range [lower_bound, upper_bound]
 * @param[in] lower_bound
 * @param[in] upper_bound
 * @note The range is inclusive
 */
int utils::random::rand_bounded_int(int lower_bound, int upper_bound)
{
	int div = RAND_MAX / (upper_bound - lower_bound);
	return lower_bound + rand() / div;
}
