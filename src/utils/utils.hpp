#include "array.hpp"
#include "sort.hpp"
#include "random.hpp"

namespace utils
{
	//TODO: maybe move to another namespace
	inline double clip(double value, double lower_bound, double upper_bound)
	{
		if (value < lower_bound)
			return lower_bound;
		else if (value > upper_bound)
			return upper_bound;
		else
			return value;
	}

	//TODO: maybe move to another namespace
	inline float fast_clip(float n, float lower, float upper)
	{
		//ova e mnogu interesen kod
		n = (n > lower) * n + !(n > lower) * lower;
		return (n < upper) * n + !(n < upper) * upper;
	}
}
