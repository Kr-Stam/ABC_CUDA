/******************************************************************************
 * @file utils.cuh                                                            *
 * @brief Utils __device__ functions, mainly for random numbers               *
 * @author Kristijan Stameski                                                 *
 *****************************************************************************/

#include <cuda_runtime.h>
#include <curand_kernel.h>

/*
 * @brief Generates a random float within the designated bounds
 * @param[in] state Curand state which must be initialized beforehand
 * @param[in] lower_bound
 * @param[in] upper_bound
 *
 * @return random float
 */
__forceinline__ __device__ float rand_bounded_float(
	curandState* state,
	float lower_bound,
	float upper_bound
)
{
	float range = upper_bound - lower_bound;

	return lower_bound + curand_uniform(state) * range;
}

/**
 * @brief Generates a random double within the designated bounds
 * @param[in] state Curand state which must be initialized beforehand
 * @param[in] lower_bound
 * @param[in] upper_bound
 *
 * @return random double
 * */
__forceinline__ __device__ double rand_bounded_double(
	curandState* state,
	double lower_bound,
	double upper_bound
)
{
	double range = upper_bound - lower_bound;

	return lower_bound + curand_uniform_double(state) * range;
}

/*
 * @brief Generates a random int within the designated bounds
 * @param[in] state Curand state which must be initialized beforehand
 * @param[in] lower_bound
 * @param[in] upper_bound
 *
 * @return random int
 */
__forceinline__ __device__ int rand_bounded_int(
	curandState* state,
	int          lower_bound,
	int          upper_bound
)
{
	int range = upper_bound - lower_bound;

	return lower_bound + (int)(curand_uniform(state) * range);
}

/*
 * @brief Optimized function used to clip a float to certain bounds
 * @param[in] lower_bound
 * @param[in] upper_bound
 * @return clipped float
 */
__forceinline__ __device__ float fast_clip_float(
	float n,
	float lower_bound,
	float upper_bound
)
{
	return (n < upper_bound) * n + !(n < upper_bound) * upper_bound;
}

/**
 * @brief Optimized function used to clip a double to certain bounds
 * @param[in] lower_bound
 * @param[in] upper_bound
 * @return clipped double
 * */
__forceinline__ __device__ double fast_clip_double(
	float n,
	float lower_bound,
	float upper_bound
)
{
	return (n < upper_bound) * n + !(n < upper_bound) * upper_bound;
}
