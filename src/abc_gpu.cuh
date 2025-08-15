/******************************************************************************
 * @file abc_gpu.cuh                                                          *
 * @brief Optimized CUDA implementation of the artificial bee colony algorithm*
 * @details This header file contains a template function kernel that is      *
 *          designed as a monolith kernel with conditional template compiling *
 *          based on passed in parameters, which deterimine how the algorithm *
 *          is implemented                                                    *
 *                                                                            *
 * @author Kristijan Stameski                                                 *
 *****************************************************************************/

#pragma once

#include <curand_kernel.h>
#include "problems/problems.h"
#include "abc_main.cuh"
#include "rank_array.cuh"
#include "problems/problems_gpu.cuh"
#include <math.h>
#include <cstdint>
#include "utils.cuh"
#include "problems/problems.h"
#include "timer.cuh"

namespace gpu{

	using namespace abc_shared;


	#define MAX_DIMENSIONS 10

	__constant__ float c_lower_bounds[MAX_DIMENSIONS];
	__constant__ float c_upper_bounds[MAX_DIMENSIONS];

	//TODO: 
	//treba da napravam da ne prima nizi tuku specifichni indeksi/vrednosti
	//poradi toa shto nema potreba od celata niza
	/*
	 * @brief Generate a random solution/food source within a hypercube defined
	 *        by the given bounds
	 *
	 * @tparam[in] dim    the number of dimensions for the problem
	 * @tparam[in] inverse determines whether to inverse the result
	 
	 * @param[inout] cords an array of floats which includes all cords
	 * @param[inout] values      an array of floats which includes all values
	 * @param[in]    idx         the position of the bee in the given arrays
	 * @param[in]    function    function to be optimized
	 */
	template<uint32_t dim>
	__inline__ __device__ void generate_random_solution(
		float*       cords,
		float*       values,
		int          idx,
		opt_func     function,
		curandState* state
	)
	{
		//#pragma unroll
		for(int i = 0; i < dim; i++)
			cords[idx*dim+i] = rand_bounded_float(
				state,
				c_lower_bounds[i],
				c_upper_bounds[i]
			);

		values[idx] = function(&cords[idx], dim);
	}

	/*
	 * @brief Local optimization around the existing food source
	 *
	 * @tparam[in] dim    the number of dimensions for the problem
	 * @tparam[in] inverse determines whether to inverse the result
	 *
	 * @param[in] idx          index of the bee/thread to be optimized
	 * @param[in] cords  an coordinate array of all candidate bees/threads
	 * @details Randomly select another bee and merge the solutions with
	 *          a stochastic step
	 *
	 * @note The index is passed in because of simpler shared memory optimization
	 */
	template<uint32_t dim>
	__forceinline__ __device__ void local_optimization(
		int          idx,
		float*       cords,
		float*       values,
		int*         trials,
		opt_func     function,
		curandState* state,
		float c_lower_bounds[dim],
		float c_upper_bounds[dim]
	)
	{
		int choice = rand_bounded_int(state, 0, blockDim.x);
		float tmp_cords[dim];

		//! proveri dali e isto 
		for(int i = 0; i < dim; i++)
		{
			float step = cords[idx*dim + i] - cords[choice*dim + i];
			step *= curand_uniform(state);

			tmp_cords[i] = cords[idx*dim + i] + step;

			tmp_cords[i] = fast_clip_float(
				tmp_cords[i],
				c_lower_bounds[i],
				c_upper_bounds[i]
			);
		}

		float tmp_value = function(tmp_cords, dim);

		if(tmp_value < values[idx])
		{
			values[idx] = tmp_value;
			*trials = 0;
		}
		else
		{
			(*trials)++;
		}
	}

	/*
	 * @brief Optimized sum reduction exploiting warp-level parallelism
	 * @param[in] sdata source array to be sumed
	 * @param[in] tid   index in the sum array
	 */
	template<uint32_t BLOCK_SIZE>
	__forceinline__ __device__ void warp_reduce_sum(volatile float* sdata, int tid)
	{
		if(BLOCK_SIZE >= 64) sdata[tid] += sdata[tid + 32];
		if(BLOCK_SIZE >= 32) sdata[tid] += sdata[tid + 16];
		if(BLOCK_SIZE >= 16) sdata[tid] += sdata[tid +  8];
		if(BLOCK_SIZE >=  8) sdata[tid] += sdata[tid +  4];
		if(BLOCK_SIZE >=  4) sdata[tid] += sdata[tid +  2];
		if(BLOCK_SIZE >=  2) sdata[tid] += sdata[tid +  1];
	}

	/*
	 * @brief Optimized max calculation exploiting warp-level parallelism
	 * @param[in] sdata source array 
	 * @param[in] tid   index in the source array
	 */
	template<uint32_t BLOCK_SIZE>
	__forceinline__ __device__ void warp_reduce_max(volatile float* sdata, int tid)
	{
		if(BLOCK_SIZE >= 64) sdata[tid] = fmax(sdata[tid], sdata[tid + 32]);
		if(BLOCK_SIZE >= 32) sdata[tid] = fmax(sdata[tid], sdata[tid + 16]);
		if(BLOCK_SIZE >= 16) sdata[tid] = fmax(sdata[tid], sdata[tid +  8]);
		if(BLOCK_SIZE >=  8) sdata[tid] = fmax(sdata[tid], sdata[tid +  4]);
		if(BLOCK_SIZE >=  4) sdata[tid] = fmax(sdata[tid], sdata[tid +  2]);
		if(BLOCK_SIZE >=  2) sdata[tid] = fmax(sdata[tid], sdata[tid +  1]);
	}

	/*
	 * @brief Initializes a roulette wheel
	 *
	 * @tparam[in] BLOCK_SIZE    number of choices
	 * @tparam[in] roulette_type determines the normalization type used
	 *
	 * @param[in]  values   array of floats based on which the roulette wheel
	 *                      will be constructed
	 * @param[out] roulette  array of float values needed for weighted roulette
	 *                       wheel selection
	 * @param[in]  shmem_max array of at least BLOCK_SIZE used to calculate
	 *                       the maximum element
	 * @param[in]  shmem_sum array of at least BLOCK_SIZE used to calculate
	 *                       the sum of all elements
	 * @param[in]  tid       idx to be used to load the respective value element
	 *
	 * @note The resulting roulette array does not contain a cumulative sum of
	 *       elements which means that the elements must be sumed when iterating
	 *       for selection
	 */
	template<
		uint32_t BLOCK_SIZE,
		Roulette roulette_type
	>
	__device__ void create_roulette_wheel(
		float* sh_values,
		float* sh_roulette,
		float* sh_sum,
		float* sh_max
	)
	{
		//sum reduction and max calculation
		sh_sum[threadIdx.x] = sh_values[threadIdx.x];
		sh_max[threadIdx.x] = sh_values[threadIdx.x];

		if constexpr (BLOCK_SIZE == 1024)
		{
			if(threadIdx.x < 512)
			{
				sh_sum[threadIdx.x] += sh_sum[threadIdx.x + 512];
				sh_max[threadIdx.x] =  fmax(sh_max[threadIdx.x],
											sh_max[threadIdx.x + 512]);
			}
			__syncthreads();
		}
		
		if constexpr (BLOCK_SIZE >= 512)
		{
			if(threadIdx.x < 256)
			{
				sh_sum[threadIdx.x] += sh_sum[threadIdx.x + 256];
				sh_max[threadIdx.x] =  fmax(sh_max[threadIdx.x],
											sh_max[threadIdx.x + 256]);
			}
			__syncthreads();
		}
		if constexpr (BLOCK_SIZE >= 256)
		{
			if(threadIdx.x < 128)
			{
				sh_sum[threadIdx.x] += sh_sum[threadIdx.x + 128];
				sh_max[threadIdx.x] =  fmax(sh_max[threadIdx.x],
											sh_max[threadIdx.x + 128]);
			}
			__syncthreads();
		}
		if constexpr (BLOCK_SIZE >= 128)
		{
			if(threadIdx.x < 64)
			{
				sh_sum[threadIdx.x] += sh_sum[threadIdx.x + 64];
				sh_max[threadIdx.x] =  fmax(sh_max[threadIdx.x],
											sh_max[threadIdx.x + 64]);
			}
			__syncthreads();
		}

		if(threadIdx.x < 32) warp_reduce_sum<BLOCK_SIZE>(sh_sum, threadIdx.x);
		if(threadIdx.x < 32) warp_reduce_max<BLOCK_SIZE>(sh_max, threadIdx.x);

		__syncthreads();

		//kakva normalizacija pravam tuka (max - value) / (max*count - sum)
		
		
		//! treba da se usoglasi
		sh_roulette[threadIdx.x] = (sh_max[0] - sh_values[threadIdx.x]) /
								   (blockDim.x*sh_max[0] - sh_sum[0]);
		__syncthreads();
	}

	template<
		uint32_t dim,
		int      num_of_bees
	>
	__device__ void sort_bees(
		float* sh_values,
		float* sh_cords
	)
	{
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		//problemot so ova e shto dodatno treba i da go smenam redosledot na
		//koordinatite

		for(int size = 2; size < num_of_bees; size <<= 2)
		{
			for(int stride = size >> 1; stride > 0; stride >>= 1) 
			{
				int pos = (tid << 1) - (tid && (stride - 1));
				if(pos + stride < num_of_bees)
				{
					bool ascending = (tid & (size >> 1)) == 0;
					float a = sh_values[tid];
					float b = sh_values[tid + stride];

					if((a > b) == ascending)
					{
						sh_values[tid]        = b;
						sh_values[tid+stride] = a;

						#pragma unroll
						for(int i = 0; i < dim; i++)
						{
							float tmp = sh_cords[tid*dim+i];
							sh_cords[tid*dim+i] = sh_cords[(tid+stride)*dim+i];
							sh_cords[(tid+stride)*dim+i] = tmp;
						}
					}
				}
				__syncthreads();
			}

		}
	}

	template<uint32_t size>
	__forceinline__ __device__ int rank_selection_from_arr(
		float* sh_cords,
		float* sh_values,
		float  choice,
		float* rank_arr
	)
	{
		//for(int i = 0; i < size; i++)
		//	if(rank_arr[i] >= choice) return i;

		int low  = 0;
		int high = size - 1;

		//! od shto proveruvav na godbolt ova e najoptimiziran kod
		int mid;
		while(high > low)
		{
			mid = (high - low) >> 1;
			if(rank_arr[mid] < choice)
				low = mid + 1;
			else if(rank_arr[mid] > choice)
				high = mid - 1;
			else
				return mid;
		}
		return mid;
	}

	template<
		uint32_t dim,
		uint32_t size,
		Rank     rank_type,
		uint32_t c_num,
		uint32_t c_div,
		bool     inverse,
		bool     shmem
	>
	__forceinline__ __device__ int rank_selection_constant(
		float*       sh_cords,
		float*       sh_values,
		int*         trials,
		curandState* state
	)
	{
		float rand = curand_uniform(state);

		if constexpr (rank_type == CONSTANT_LINEAR)
			return rank_const::dev_lin<size>(rand);
		if constexpr (rank_type == CONSTANT_EXPONENTIAL)
			return rank_const::dev_exp<size, c_num, c_div>(rand);
		if constexpr (rank_type == CONSTANT_EXPONENTIAL_2)
			return rank_const::dev_exp2<size, c_num, c_div>(rand);

		return 0;
	}

	template<
		uint32_t size,
		uint32_t dim,
		uint32_t tournament_size
	>
	__forceinline__ __device__ int tournament_selection_standard(
		float*       cords,
		float*       values,
		curandState* state	
	)
	{
			int choices[tournament_size];

			#pragma unroll
			for(int j = 0; j < tournament_size; j++)
				choices[j] = (size-1) * curand_uniform(state);

			int   min_idx   = 0;
			float min_value = choices[0];

			#pragma unroll
			for(int j = 1; j < tournament_size; j++)
			{
				if(min_value > values[choices[j]])
				{
					min_idx   = choices[j];
					min_value = values[min_idx];
				}
			}

			return min_idx;
	}

	template<
		uint32_t size,
		uint32_t dim,
		uint32_t num_of_games,
		uint32_t num_of_contestants
	>
	__forceinline__ __device__ int tournament_selection_custom(
		float*       cords,
		float*       values,
		curandState* state
	)
	{
			int choices[num_of_games];

			int   min_idx;
			float min_value = INFINITY;

			for(int i = 0; i < num_of_games; i++)
			{
				#pragma unroll
				for(int j = 0; j < num_of_games; j++)
					choices[j] = (size-1) * curand_uniform(state);

				#pragma unroll
				for(int j = 0; j < num_of_games; j++)
				{
					if(min_value > values[choices[j]])
					{
						min_idx   = choices[j];
						min_value = values[min_idx];
					}
				}
			}

			return min_idx;
	}

	template<
		uint32_t dim,
		uint32_t BLOCK_SIZE,
		Roulette roulette_type
	>
	__forceinline__ __device__ int roulette_wheel_selection(
		float* sh_values,
		float* sh_cords,
		float* sh_roulette, 
		float* sh_sum,
		float* sh_max,
		curandState* state
	)
	{
		//create_roulette_wheel<
		//	BLOCK_SIZE,
		//	roulette_type
		//>(
		//	sh_values,
		//	sh_roulette,
		//	sh_sum,
		//	sh_max
		//);
		//sum reduction and max calculation
		sh_sum[threadIdx.x] = sh_values[threadIdx.x];
		sh_max[threadIdx.x] = sh_values[threadIdx.x];

		if constexpr (BLOCK_SIZE == 1024)
		{
			if(threadIdx.x < 512)
			{
				sh_sum[threadIdx.x] += sh_sum[threadIdx.x + 512];
				sh_max[threadIdx.x] =  fmax(sh_max[threadIdx.x],
											sh_max[threadIdx.x + 512]);
			}
			__syncthreads();
		}
		
		if constexpr (BLOCK_SIZE >= 512)
		{
			if(threadIdx.x < 256)
			{
				sh_sum[threadIdx.x] += sh_sum[threadIdx.x + 256];
				sh_max[threadIdx.x] =  fmax(sh_max[threadIdx.x],
											sh_max[threadIdx.x + 256]);
			}
			__syncthreads();
		}
		if constexpr (BLOCK_SIZE >= 256)
		{
			if(threadIdx.x < 128)
			{
				sh_sum[threadIdx.x] += sh_sum[threadIdx.x + 128];
				sh_max[threadIdx.x] =  fmax(sh_max[threadIdx.x],
											sh_max[threadIdx.x + 128]);
			}
			__syncthreads();
		}
		if constexpr (BLOCK_SIZE >= 128)
		{
			if(threadIdx.x < 64)
			{
				sh_sum[threadIdx.x] += sh_sum[threadIdx.x + 64];
				sh_max[threadIdx.x] =  fmax(sh_max[threadIdx.x],
											sh_max[threadIdx.x + 64]);
			}
			__syncthreads();
		}

		if(threadIdx.x < 32) warp_reduce_sum<BLOCK_SIZE>(sh_sum, threadIdx.x);
		if(threadIdx.x < 32) warp_reduce_max<BLOCK_SIZE>(sh_max, threadIdx.x);

		__syncthreads();

		sh_roulette[threadIdx.x] = (sh_max[0] - sh_values[threadIdx.x]) /
								   (blockDim.x*sh_max[0] - sh_sum[0]);
		__syncthreads();

		float sum = 0;
		float rand = curand_uniform(state);

		for(int idx = 0; idx < BLOCK_SIZE; idx++)
		{
			sum += sh_roulette[idx];
			if(rand <= sum) return idx;
		}

		return 0;
	}

	template<uint32_t dim>
	__device__ void copy_cords(
		float dest[dim],
		float src[dim]
	)
	{
		#pragma unroll
		for(int i = 0; i < dim; i++)
			dest[i] = src[i];
	}

	__constant__ float rank_arr[1024];

	/*
	 * @brief Optimized parallel GPU version of the ABC algorithm
	 *
	 * @tparam[in] dim             dimensions of the problem
	 * @tparam[in] BLOCK_SIZE      size of each block, used for min/max/sum
	 * @tparam[in] selection_type  determines which selection algorithm used
	 * @tparam[in] roulette_type   type of roulette selection
	 * @tparam[in] rank_type       type of rank selection
	 * @tparam[in] tournament_type type of tournament selection
	 * @tparam[in] shmem_enabled   determines if shared memory is used
	 *
	 * @param[inout] cords           initial solution/food source cordinates
	 * @param[inout] values          initial solution/food source fitness values
	 * @param[inout] hive_cords      used to pass values between blocks
	 * @param[in]    max_generations max number of iterations of the algorithm
	 * @param[in]    trials_limit    number of times a solution can be improved
	 *                               before being abandoned
	 * @param[in]    function        pointer to a function to be optimized,
	 *                               of type (array(float), int) -> float
	 */
	template<
		uint32_t   dim,
		uint32_t   BLOCK_SIZE,
		uint32_t   GRID_SIZE,
		uint32_t   COLONY_POOL,
		uint32_t   iterations, 
		Selection  selection_type,
		Roulette   roulette_type,
		Rank       rank_type,
		Tourn      tournament_type,
		uint32_t   tournament_size,
		uint32_t   tournament_num,
		bool       shmem_enabled
	>
	__global__ void abc(
		float*   cords,
		float*   values,
		float*   hive_cords,
		int      trials_limit, 
		opt_func optimization_function
	)
	{
		//-------------------SHARED-MEMORY-DECLARATION-----------------------//
		extern __shared__ float shmem[];

		float *sh_cords, *sh_values, *sh_sum, *sh_max, *sh_roulette;

		if constexpr (shmem_enabled)
		{
			sh_cords  = shmem;
			sh_values = sh_cords + blockDim.x*dim;

			sh_sum      = sh_values + blockDim.x;
			sh_max      = sh_sum    + blockDim.x;
			sh_roulette = sh_max    + blockDim.x;

			if (threadIdx.x < COLONY_POOL)
			{
				//Load values into shared memory
				copy_cords<dim>(
					&sh_cords[threadIdx.x*dim],
					//&hive_cords[(blockDim.x - blockIdx.x)*dim]
					&hive_cords[(GRID_SIZE - blockIdx.x)*dim]
				);
			}
			else
			{
				//Load values into shared memory
				copy_cords<dim>(
					&sh_cords[threadIdx.x*dim],
					&cords[(blockDim.x*blockIdx.x + threadIdx.x)*dim]
				);
			}

			sh_values[threadIdx.x] = optimization_function(
				&sh_cords[threadIdx.x*dim],
				dim
			);
		}
		//! ova sega za sega e sekogash so shared memory
		else
		{
			sh_cords  = cords;
			sh_values = values;
			
			sh_sum      = shmem;
			sh_max      = sh_sum + blockDim.x;
			sh_roulette = sh_max + blockDim.x;
		}
		__syncthreads();
		//-----------------------------------------------------------------------//

		//------------------------CURAND-INITIALIZATION--------------------------//
		curandState state;
		curand_init(clock64() + blockDim.x*blockIdx.x + threadIdx.x, 0, 0, &state);
		//-----------------------------------------------------------------------//

		//----------------------INITIALIZE-INITIAL-STATE-------------------------//
		float tmp_cords[dim];
		float tmp_value;
		#pragma unroll
		for(int i = 0; i < dim; i++)
			tmp_cords[i] = c_lower_bounds[i] + curand_uniform(&state) *
				(c_upper_bounds[i] - c_lower_bounds[i]);
		  
		tmp_value = optimization_function(tmp_cords, dim);
		if(tmp_value < sh_values[threadIdx.x])
		{
			sh_values[threadIdx.x] = tmp_value;
			//!!!
			copy_cords<dim>(
				&sh_cords[threadIdx.x*dim],
				tmp_cords
			);
		}
		int trials = 0;

		__syncthreads();
		//-----------------------------------------------------------------------//

		//--------------------------------MAIN-LOOOP-----------------------------//
		for(int i = 0; i < iterations; i++)
		{
			//------------------EMPLOYED-BEE-LOCAL-OPTIMIZATION------------------//
			int choice;
			choice = curand_uniform(&state) * BLOCK_SIZE;

			for(int i = 0; i < dim; i++)
			{
				tmp_cords[i] = sh_cords[threadIdx.x*dim + i] +
					(sh_cords[threadIdx.x*dim + i] - sh_cords[choice*dim + i]) *
					curand_uniform(&state);

				if(tmp_cords[i] > c_upper_bounds[i])
					tmp_cords[i] = c_upper_bounds[i];
				if(tmp_cords[i] < c_lower_bounds[i])
					tmp_cords[i] = c_lower_bounds[i];
			}

			tmp_value = optimization_function(tmp_cords, dim);
			if(tmp_value < sh_values[threadIdx.x])
			{
				sh_values[threadIdx.x] = tmp_value;
				//!!!
				copy_cords<dim>(
					&sh_cords[threadIdx.x*dim],
					tmp_cords
				);

				trials = 0;
			}
			else
			{
				trials++;
			}
			__syncthreads();
			//-------------------------------------------------------------------//

			//------------------ONLOOKER-BEE-GLOBAL-OPTIMIZATION-----------------//
			// the selection is inlined to reduce register usage in the kernel
			if constexpr (selection_type == ROULETTE_WHEEL)
			{
				if constexpr (roulette_type  == SUM)
				{
					sh_sum[threadIdx.x] = sh_values[threadIdx.x];

					if constexpr (BLOCK_SIZE == 1024)
					{
						if(threadIdx.x < 512)
							sh_sum[threadIdx.x] += sh_sum[threadIdx.x + 512];
						__syncthreads();
					}

					if constexpr (BLOCK_SIZE >= 512)
					{
						if(threadIdx.x < 256)
							sh_sum[threadIdx.x] += sh_sum[threadIdx.x + 256];
						__syncthreads();
					}
					if constexpr (BLOCK_SIZE >= 256)
					{
						if(threadIdx.x < 128)
							sh_sum[threadIdx.x] += sh_sum[threadIdx.x + 128];
						__syncthreads();
					}
					if constexpr (BLOCK_SIZE >= 128)
					{
						if(threadIdx.x < 64)
							sh_sum[threadIdx.x] += sh_sum[threadIdx.x + 64];
						__syncthreads();
					}

					if(threadIdx.x < 32)
					{
						warp_reduce_sum<BLOCK_SIZE>(sh_sum, threadIdx.x);
						warp_reduce_max<BLOCK_SIZE>(sh_max, threadIdx.x);
					}

					__syncthreads();

					sh_roulette[threadIdx.x] = sh_values[threadIdx.x] / sh_sum[0];
					__syncthreads();
				}
				else if constexpr (roulette_type == CUSTOM)
				{
					sh_sum[threadIdx.x] = sh_values[threadIdx.x];
					sh_max[threadIdx.x] = sh_values[threadIdx.x];

					if constexpr (BLOCK_SIZE == 1024)
					{
						if(threadIdx.x < 512)
						{
							sh_sum[threadIdx.x] += sh_sum[threadIdx.x + 512];
							sh_max[threadIdx.x] = fmax(sh_max[threadIdx.x],
							                           sh_max[threadIdx.x + 512]);
						}
						__syncthreads();
					}

					if constexpr (BLOCK_SIZE >= 512)
					{
						if(threadIdx.x < 256)
						{
							sh_sum[threadIdx.x] += sh_sum[threadIdx.x + 256];
							sh_max[threadIdx.x] =  fmax(sh_max[threadIdx.x],
							                            sh_max[threadIdx.x + 256]);
						}
						__syncthreads();
					}
					if constexpr (BLOCK_SIZE >= 256)
					{
						if(threadIdx.x < 128)
						{
							sh_sum[threadIdx.x] += sh_sum[threadIdx.x + 128];
							sh_max[threadIdx.x] =  fmax(sh_max[threadIdx.x],
							                            sh_max[threadIdx.x + 128]);
						}
						__syncthreads();
					}
					if constexpr (BLOCK_SIZE >= 128)
					{
						if(threadIdx.x < 64)
						{
							sh_sum[threadIdx.x] += sh_sum[threadIdx.x + 64];
							sh_max[threadIdx.x] =  fmax(sh_max[threadIdx.x],
							                            sh_max[threadIdx.x + 64]);
						}
						__syncthreads();
					}

					if(threadIdx.x < 32)
					{
						warp_reduce_sum<BLOCK_SIZE>(sh_sum, threadIdx.x);
						warp_reduce_max<BLOCK_SIZE>(sh_max, threadIdx.x);
					}

					__syncthreads();

					sh_roulette[threadIdx.x] =
						(sh_max[0] - sh_values[threadIdx.x]) /
						(blockDim.x*sh_max[0] - sh_sum[0]);
					__syncthreads();
				}
				else if constexpr (roulette_type == MIN_MAX)
				{
					float* sh_min = sh_sum;
					sh_min[threadIdx.x] = sh_values[threadIdx.x];
					sh_max[threadIdx.x] = sh_values[threadIdx.x];

					if constexpr (BLOCK_SIZE == 1024)
					{
						if(threadIdx.x < 512)
						{
							sh_min[threadIdx.x] = fmin(sh_min[threadIdx.x],
							                           sh_min[threadIdx.x + 512]);
							sh_max[threadIdx.x] = fmax(sh_max[threadIdx.x],
							                           sh_max[threadIdx.x + 512]);
						}
						__syncthreads();
					}

					if constexpr (BLOCK_SIZE >= 512)
					{
						if(threadIdx.x < 256)
						{
							sh_min[threadIdx.x] = fmin(sh_min[threadIdx.x],
							                           sh_min[threadIdx.x + 256]);
							sh_max[threadIdx.x] = fmax(sh_max[threadIdx.x],
							                           sh_max[threadIdx.x + 256]);
						}
						__syncthreads();
					}
					if constexpr (BLOCK_SIZE >= 256)
					{
						if(threadIdx.x < 128)
						{
							sh_min[threadIdx.x] =  fmin(sh_min[threadIdx.x],
							                            sh_min[threadIdx.x + 128]);
							sh_max[threadIdx.x] =  fmax(sh_max[threadIdx.x],
							                            sh_max[threadIdx.x + 128]);
						}
						__syncthreads();
					}
					if constexpr (BLOCK_SIZE >= 128)
					{
						if(threadIdx.x < 64)
						{
							sh_min[threadIdx.x] = fmin(sh_min[threadIdx.x],
							                           sh_min[threadIdx.x + 64]);
							sh_max[threadIdx.x] = fmax(sh_max[threadIdx.x],
							                           sh_max[threadIdx.x + 64]);
						}
						__syncthreads();
					}

					if(threadIdx.x < 32)
					{
						warp_reduce_sum<BLOCK_SIZE>(sh_min, threadIdx.x);
						warp_reduce_max<BLOCK_SIZE>(sh_max, threadIdx.x);
					}

					__syncthreads();

					sh_roulette[threadIdx.x] =
						(sh_values[threadIdx.x] - sh_min[0]) /
						(sh_max[0] - sh_min[0]);
					__syncthreads();
				}
				//! ova treba da bide dopraveno
				float sum = 0;
				float rand = curand_uniform(&state);
				choice = 0;

				for(int idx = 0; idx < BLOCK_SIZE; idx++)
				{
					sum += sh_roulette[idx];
					if(rand <= sum)
					{
						choice = idx;
						break;
					}
				}
			}
			else if constexpr (selection_type == RANK)
			{
				sort_bees<dim, BLOCK_SIZE>(sh_cords, sh_values);
				float rand = curand_uniform(&state);
				if constexpr (rank_type < CONSTANT_LINEAR)
				{
					choice = rank_selection_from_arr<dim>(
						sh_cords,
						sh_values,
						rand,
						rank_arr
					 );
				}
				else if constexpr (rank_type == CONSTANT_LINEAR)
					choice = rank_const::dev_lin<BLOCK_SIZE>(rand);
				else if constexpr (rank_type == CONSTANT_EXPONENTIAL)
					choice = rank_const::dev_exp<BLOCK_SIZE, 1, 20>(rand);
				else if constexpr (rank_type == CONSTANT_EXPONENTIAL_2)
					choice = rank_const::dev_exp2<BLOCK_SIZE, 1, 20>(rand);

				//! ova e poradi toa shto CONSTANT_EXPONENTIAL ne e pravilno
				//! napraven, 2 e okej
				if (choice > BLOCK_SIZE) choice = 0;
			}
			else if constexpr (selection_type == TOURNAMENT)
			{
				if constexpr (tournament_type == SINGLE)
				{
					choice = tournament_selection_standard<
						BLOCK_SIZE,
						dim,
						tournament_size
					>(
						sh_cords,
						sh_values,
						&state
					);
				}
				else if constexpr (tournament_type == MULTIPLE)
				{
					choice = tournament_selection_custom<
						BLOCK_SIZE,
						dim,
						tournament_num,
						tournament_size
					>(
						sh_cords,
						sh_values,
						&state
					 );
				}
			}

			//! radi sinhronizacija
			//!!!
			copy_cords<dim>(
				 tmp_cords,
				&sh_cords[choice*dim]
			);
			for(int i = 0; i < dim; i++)
			{
				if(tmp_cords[i] > c_upper_bounds[i])
					tmp_cords[i] = c_upper_bounds[i];
				if(tmp_cords[i] < c_lower_bounds[i])
					tmp_cords[i] = c_lower_bounds[i];
			}
			//! specifichno tuka pravi nekoj problem optimizacijata
			//tmp_value = sh_values[choice];
			//tmp_value = optimization_function(&sh_cords[choice*dim], dim);
			tmp_value = optimization_function(tmp_cords, dim);

			if(sh_values[threadIdx.x] > tmp_value)
			{
				copy_cords<dim>(
					&sh_cords[threadIdx.x*dim],
					 tmp_cords
				);
				sh_values[threadIdx.x] = tmp_value;

				trials = 0;
			}
			else
			{
				trials++;
			}
			//-------------------------------------------------------------------//

			//------------------------TRIAL-LIMIT-CHECK--------------------------//
			if(trials > trials_limit)
			{
				#pragma unroll
				for(int i = 0; i < dim; i++)
				{
					sh_cords[threadIdx.x*dim+i] = curand_uniform(&state) *
						(c_upper_bounds[i] - c_lower_bounds[i]);
					if(sh_cords[threadIdx.x*dim+i] > c_upper_bounds[i])
						sh_cords[threadIdx.x*dim+i] = c_upper_bounds[i];
					if(sh_cords[threadIdx.x*dim+i] < c_lower_bounds[i])
						sh_cords[threadIdx.x*dim+i] = c_lower_bounds[i];
				}

				//!!!
				sh_values[threadIdx.x] = optimization_function(
					&sh_cords[threadIdx.x*dim],
					dim
				);
				trials = 0;
			}
			//-------------------------------------------------------------------//
		}

		//-----------------------------RETURN-VARIABLES--------------------------//
		copy_cords<dim>(
			&cords[(blockDim.x*blockIdx.x + threadIdx.x)*dim],
			&sh_cords[threadIdx.x*dim]
		);
	
		//values[blockDim.x*blockIdx.x + threadIdx.x] = sh_values[threadIdx.x];
		values[blockDim.x*blockIdx.x + threadIdx.x] = 
			optimization_function(&sh_cords[threadIdx.x*dim], dim);

		if (threadIdx.x < COLONY_POOL)
		{
			//Load values into shared memory
			copy_cords<dim>(
				&hive_cords[(blockIdx.x*COLONY_POOL + threadIdx.x)*dim],
				&sh_cords[threadIdx.x*dim]
			);
		}
		//-----------------------------------------------------------------------//
	}

	/*
	 * @brief Function pointers used to assign the optimization function
	 * @details
	 * Due to the cpu and gpu memory spaces being separate,
	 * in order to pass a function pointer to the kernel,
	 * the CPU code must first	copy the symbol location of a GPU
	 * assigned function pointer
	 */
	__device__ opt_func d_rosenbrock     = problems::gpu::rosenbrock;
	__device__ opt_func d_cross_in_tray  = problems::gpu::cross_in_tray;
	__device__ opt_func d_schaffer2      = problems::gpu::schaffer2;
	__device__ opt_func d_schaffer4      = problems::gpu::schaffer4;
	__device__ opt_func d_bohachevsky1   = problems::gpu::bohachevsky1;
	__device__ opt_func d_bohachevsky2   = problems::gpu::bohachevsky2;
	__device__ opt_func d_bohachevsky3   = problems::gpu::bohachevsky3;
	__device__ opt_func d_schwefel       = problems::gpu::schwefel;
	
	#define COLONY_POOL 4

	/*
	 * @brief Function used to launch the GPU version of the ABC algorithm
	 * 
	 * @param[out] cords    array of potential solutions cords 
	 * @param[out] values         array of potential solutions fitness values 
	 * @param[in] num_of_bees     total number of bees
	 * @param[in] max_generations maximum number of iterations of the algorithm
	 * @param[in] trials_limit    the number of times a solution can't be improved
	 *                            before being discarded
	 * @param[in] function        pointer to the function to be optimized,
	 *                            of type (array(float), int) -> float
	 * @param[in] lower_bounds    lower bounds of the search space
	 * @param[in] upper_bounds    upper bounds of the search space
	 * @param[in] steps           number of steps over which to execute an 
	 *                            optimization cycle, used for testing convergance
	 *                            rates
	 *
	 * @details This function is used to call the GPU kernel and handles
	 *          inter-block communication and probes the state of the bee colony
	 *          at a set number of steps
	 */
	template<
		uint32_t  dimensions,
		uint32_t  grid_size,
		uint32_t  block_size,
		uint32_t  iterations,
		uint32_t  trials_limit,
		Selection selection_type,
		Roulette  roulette_type,
		Rank      rank_type,
		Tourn     tournament_type,
		uint32_t  tournament_size,
		uint32_t  tournament_num
	>
	void launch_abc(
		float*    cords,
		float*    values,
		TestFunc  test_function,
		float     lower_bounds[],
		float     upper_bounds[],
		uint64_t* duration
	)
	{
		float* d_cords;
		float* d_values;
		float* d_hive_cords;

		//TODO: treba da isprobam so 1024, ne znam dali kje raboti?
		const size_t num_of_bees = block_size*grid_size;

		size_t cords_size  = num_of_bees*sizeof(float)*dimensions;
		size_t values_size = num_of_bees*sizeof(float);

		size_t hive_cords_size =
			dimensions*COLONY_POOL*grid_size*sizeof(float);

		size_t bounds_size = dimensions  * sizeof(float);

		cudaMalloc((void**) &d_cords,      cords_size);
		cudaMalloc((void**) &d_values,     values_size);
		cudaMalloc((void**) &d_hive_cords, hive_cords_size);

		//! ova go ostaviv za da mozhe da proveram razlika
		//float* d_upper_bounds;
		//float* d_lower_bounds;
		//cudaMalloc((void**) &d_upper_bounds, bounds_size);
		//cudaMalloc((void**) &d_lower_bounds, bounds_size);

		cudaMemcpy(d_cords,  cords, cords_size,   cudaMemcpyHostToDevice);
		cudaMemcpy(d_values, values, values_size, cudaMemcpyHostToDevice);

		cudaMemcpyToSymbol(c_upper_bounds, upper_bounds, bounds_size);
		cudaMemcpyToSymbol(c_lower_bounds, lower_bounds, bounds_size);

		size_t rank_arr_size = block_size*sizeof(float);
		float *tmp_rank_arr = (float*) malloc(rank_arr_size);
		if constexpr (rank_type == LINEAR_ARRAY)
			rank_arr::init_arr_lin<block_size>(tmp_rank_arr, 1.9f);
		else if constexpr (rank_type == EXPONENTIAL_ARRAY)
			rank_arr::init_arr_exp<block_size>(tmp_rank_arr, 1.1f);
		else if constexpr (rank_type == LINEAR_SIMPLE_ARRAY)
			rank_arr::init_arr_simple<block_size>(tmp_rank_arr);
		else if constexpr (rank_type == EXPONENTIAL_SIMPLE_ARRAY)
			rank_arr::init_arr_simple_exp<block_size>(tmp_rank_arr, 0.5f);

		cudaMemcpyToSymbol(rank_arr, tmp_rank_arr, rank_arr_size);
		
		opt_func d_symbol;
		switch(test_function)
		{
		case ROSENBROCK:
			cudaMemcpyFromSymbol(
				&d_symbol,
				d_rosenbrock,
				sizeof(opt_func),
				0,
				cudaMemcpyDeviceToHost
			);
			break;
		case CROSS_IN_TRAY:
			cudaMemcpyFromSymbol(
				&d_symbol,
				d_cross_in_tray,
				sizeof(opt_func),
				0,
				cudaMemcpyDeviceToHost
			);
			break;
		case SCHAFFER_2:
			cudaMemcpyFromSymbol(
				&d_symbol,
				d_schaffer2,
				sizeof(opt_func),
				0,
				cudaMemcpyDeviceToHost
			);
			break;
		case SCHAFFER_4:
			cudaMemcpyFromSymbol(
				&d_symbol,
				d_schaffer4,
				sizeof(opt_func),
				0,
				cudaMemcpyDeviceToHost
			);
			break;
		case BOHACHEVSKY_1:
			cudaMemcpyFromSymbol(
				&d_symbol,
				d_bohachevsky1,
				sizeof(opt_func),
				0,
				cudaMemcpyDeviceToHost
			);
			break;
		case BOHACHEVSKY_2:
			cudaMemcpyFromSymbol(
				&d_symbol,
				d_bohachevsky2,
				sizeof(opt_func),
				0,
				cudaMemcpyDeviceToHost
			);
			break;
		case BOHACHEVSKY_3:
			cudaMemcpyFromSymbol(
				&d_symbol,
				d_bohachevsky3,
				sizeof(opt_func),
				0,
				cudaMemcpyDeviceToHost
			);
			break;
		case SCHWEFEL:
			cudaMemcpyFromSymbol(
				&d_symbol,
				d_schwefel,
				sizeof(opt_func),
				0,
				cudaMemcpyDeviceToHost
			);
			break;
		case INVALID:
			return;
		}

		size_t SHMEM_SIZE =
			dimensions*block_size*sizeof(float) + //sh_cords
			           block_size*sizeof(float) + //sh_values
			           block_size*sizeof(float) + //sh_sum
			           block_size*sizeof(float) + //sh_max
			           block_size*sizeof(float);  //sh_roulette

		cudaDeviceSynchronize();

		Timer timer = Timer();
		timer.start();
		abc<
			dimensions,
			block_size,
			grid_size,
			COLONY_POOL,
			iterations,
			selection_type,
			roulette_type,
			rank_type,
			tournament_type,
			tournament_size,
			tournament_num,
			true
		><<<grid_size, block_size, SHMEM_SIZE>>>(
			d_cords,
			d_values,
			d_hive_cords,
			trials_limit,
			d_symbol
		);
		(*duration) = timer.stop();

		cudaMemcpy(
			cords,
			d_cords,
			cords_size,
			cudaMemcpyDeviceToHost
		);
		cudaMemcpy(
			values,
			d_values,
			values_size,
			cudaMemcpyDeviceToHost
		);

		cudaFree(d_cords);
		cudaFree(d_values);
		cudaFree(d_hive_cords);

		free(tmp_rank_arr);
	}
}
