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

	template<
		uint32_t dim,
		uint32_t num_of_bees
	>
	__device__ void sort_bees(
		float* sh_values,
		float* sh_cords
	)
	{
		int tid = threadIdx.x;

		//problemot so ova e shto dodatno treba i da go smenam redosledot na
		//koordinatite

		for(uint32_t size = 2; size < num_of_bees; size <<= 1)
		{
			for(uint32_t stride = size >> 1; stride > 0; stride >>= 1) 
			{
				if (stride < 32)
				{
					// better memory access pattern
					uint32_t partner = tid ^ stride;
					if(partner < num_of_bees)
					{
						bool ascending = (tid & (size >> 1)) == 0;
						float a = sh_values[tid];
						float b = sh_values[partner];

						if((a > b) == ascending)
						{
							sh_values[tid]     = b;
							sh_values[partner] = a;

							#pragma unroll
							for(int i = 0; i < dim; i++)
							{
								float tmp = sh_cords[tid*dim+i];
								sh_cords[tid*dim+i] = sh_cords[(tid+stride)*dim+i];
								sh_cords[partner*dim+i] = tmp;
							}
						}
					}
					__syncthreads();
				}
				else
				{
					// Warp-level path
					uint32_t lane = tid & 31;
					bool ascending = (tid & (size >> 1)) == 0;

					float current = sh_values[tid];
					// Exchange values within warp using shuffle
					float other = __shfl_xor_sync(0xffffffffu, current, stride);

					if(current > other && ascending && (lane & stride) == 0)
					{
						uint32_t partner = tid ^ stride;
						#pragma unroll
						for(int i = 0; i < dim; i++)
						{
							float tmp = sh_cords[tid*dim + i];
							sh_cords[tid*dim + i] = sh_cords[partner*dim + i];
							sh_cords[partner*dim + i] = tmp;
						}
					}
					__syncwarp();
				}
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
		int        sync_barrier
	>
	__global__ void abc(
		float*       cords,
		float*       values,
		float*       hive_cords,
		int          trials_limit, 
		opt_func     optimization_function,
		curandState* states
	)
	{
		//-------------------------VARIABLE-DECLARATION----------------------//
		extern __shared__ float shmem[];
		float *sh_cords, *sh_values, *sh_sum, *sh_max, *sh_roulette;

		curandState state;

		int trials;

		float tmp_cords[dim];
		float tmp_value;
		//-------------------------------------------------------------------//

		{ // Initialize shared memory and curand variables
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
		__syncthreads();

		{ // Initial bee state
			#pragma unroll
			for(int i = 0; i < dim; i++)
				tmp_cords[i] = c_lower_bounds[i] + curand_uniform(&state) *
					(c_upper_bounds[i] - c_lower_bounds[i]);
			  
			tmp_value = optimization_function(tmp_cords, dim);
			if(tmp_value < sh_values[threadIdx.x])
			{
				sh_values[threadIdx.x] = tmp_value;
				copy_cords<dim>(
					&sh_cords[threadIdx.x*dim],
					tmp_cords
				);
			}
			trials = 0;
		}
		__syncthreads();

		// Main Loop
		for(int i = 0; i < iterations; i++)
		{
			{ // Local optimization
				int choice = curand_uniform(&state) * BLOCK_SIZE;
				for(int i = 0; i < dim; i++)
				{
					tmp_cords[i] =
						sh_cords[threadIdx.x*dim + i] +
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
			}
			__syncthreads();

			{ //Onlooker Bee Global Pptimization
				int choice;
				// selection type is inlined to reduce register usage
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

						sh_roulette[threadIdx.x] = sh_values[threadIdx.x] /
						                           sh_sum[0];
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
								sh_max[threadIdx.x] = fmax(
									sh_max[threadIdx.x],
									sh_max[threadIdx.x + 512]
								);
							}
							__syncthreads();
						}

						if constexpr (BLOCK_SIZE >= 512)
						{
							if(threadIdx.x < 256)
							{
								sh_sum[threadIdx.x] += sh_sum[threadIdx.x + 256];
								sh_max[threadIdx.x] =  fmax(
									sh_max[threadIdx.x],
									sh_max[threadIdx.x + 256]
								);
							}
							__syncthreads();
						}
						if constexpr (BLOCK_SIZE >= 256)
						{
							if(threadIdx.x < 128)
							{
								sh_sum[threadIdx.x] += sh_sum[threadIdx.x + 128];
								sh_max[threadIdx.x] =  fmax(
									sh_max[threadIdx.x],
									sh_max[threadIdx.x + 128]
								);
							}
							__syncthreads();
						}
						if constexpr (BLOCK_SIZE >= 128)
						{
							if(threadIdx.x < 64)
							{
								sh_sum[threadIdx.x] += sh_sum[threadIdx.x + 64];
								sh_max[threadIdx.x] =  fmax(
									sh_max[threadIdx.x],
									sh_max[threadIdx.x + 64]
								);
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
								sh_min[threadIdx.x] = fmin(
									sh_min[threadIdx.x],
									sh_min[threadIdx.x + 512]
								);
								sh_max[threadIdx.x] = fmax(
									sh_max[threadIdx.x],
									sh_max[threadIdx.x + 512]
								);
							}
							__syncthreads();
						}

						if constexpr (BLOCK_SIZE >= 512)
						{
							if(threadIdx.x < 256)
							{
								sh_min[threadIdx.x] = fmin(
									sh_min[threadIdx.x],
									sh_min[threadIdx.x + 256]
								);
								sh_max[threadIdx.x] = fmax(
									sh_max[threadIdx.x],
									sh_max[threadIdx.x + 256]
								);
							}
							__syncthreads();
						}
						if constexpr (BLOCK_SIZE >= 256)
						{
							if(threadIdx.x < 128)
							{
								sh_min[threadIdx.x] =  fmin(
									sh_min[threadIdx.x],
									sh_min[threadIdx.x + 128]
								);
								sh_max[threadIdx.x] =  fmax(
									sh_max[threadIdx.x],
									sh_max[threadIdx.x + 128]
								);
							}
							__syncthreads();
						}
						if constexpr (BLOCK_SIZE >= 128)
						{
							if(threadIdx.x < 64)
							{
								sh_min[threadIdx.x] = fmin(
									sh_min[threadIdx.x],
									sh_min[threadIdx.x + 64]
								);
								sh_max[threadIdx.x] = fmax(
									sh_max[threadIdx.x],
									sh_max[threadIdx.x + 64]
								);
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
				
					//! this is a bottleneck
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

				// coordinates are moved to a local variable to prevent a
				// race condition
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
			}

			// Trial Limit Check
			if(trials > trials_limit)
			{
				#pragma unroll
				for(int i = 0; i < dim; i++)
				{
					float tmp_cord = curand_uniform(&state) *
						(c_upper_bounds[i] - c_lower_bounds[i]);
					
					// this check is necessary because of floating point errors
					if(tmp_cord > c_upper_bounds[i])
						tmp_cord = c_upper_bounds[i];
					if(tmp_cord < c_lower_bounds[i])
						tmp_cord = c_lower_bounds[i];

					sh_cords[threadIdx.x*dim+i] = tmp_cord;
				}

				sh_values[threadIdx.x] = optimization_function(
					&sh_cords[threadIdx.x*dim],
					dim
				);
				trials = 0;
				
				// a synchronization barrier was deemed to be unnecessary
				// since employed bee optimization would not be adversely
				// affected with a partially correct state, this has proven
				// to be a valid assumption in the experimental results
			}
		}

		{ // Assign return variables
			copy_cords<dim>(
				&cords[(blockDim.x*blockIdx.x + threadIdx.x)*dim],
				&sh_cords[threadIdx.x*dim]
			);
		
			//values[blockDim.x*blockIdx.x + threadIdx.x] =
			//	sh_values[threadIdx.x];
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
		}
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

	__global__ void init_curand_states(curandState* states)
	{
		curand_init(
			clock64() + blockDim.x*blockIdx.x + threadIdx.x,
			0, 0, &states[blockDim.x*blockIdx.x + threadIdx.x]
		);
	}

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

		// Init curand states
		curandState* d_states;
		cudaMalloc((void**) &d_states, grid_size*block_size*sizeof(curandState));
		init_curand_states<<<grid_size, block_size>>>(d_states);

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
			2
		><<<grid_size, block_size, SHMEM_SIZE>>>(
			d_cords,
			d_values,
			d_hive_cords,
			trials_limit,
			d_symbol,
			d_states
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
