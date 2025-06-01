/******************************************************************************
 * @file abc_gpu.cu                                                           *
 * @brief Parallel GPU implementation of the ABC algorithm                    *
 * @author Kristijan Stameski                                                 *
 *****************************************************************************/

#include <math.h>
#include <time.h>
#include <cstdint>
#include "abc_gpu.cuh"
#include "utils.cuh"
#include "rank_array.cuh"
#include "problems/problems.h"
#include "problems/gpu/problems_valley_shaped.cuh"

using namespace gpu;
using namespace abc_shared;

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
	float        lower_bounds[dim],
	float        upper_bounds[dim],
	int          idx,
	opt_func     function,
	curandState* state
)
{
	//#pragma unroll
	for(int i = 0; i < dim; i++)
		cords[idx*dim+i] = rand_bounded_float(
			state,
			lower_bounds[i],
			upper_bounds[i]
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
 * @param[in] lower_bounds 
 * @param[in] upper_bounds
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
	float        lower_bounds[dim],
	float        upper_bounds[dim],
	int*         trials,
	opt_func     function,
	curandState* state
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
			lower_bounds[i],
			upper_bounds[i]
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

/**
 * @brief Optimized sum reduction exploiting warp-level parallelism
 * @param[in] sdata source array to be sumed
 * @param[in] tid   index in the sum array
 * */
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

/**
 * @brief Optimized max calculation exploiting warp-level parallelism
 * @param[in] sdata source array 
 * @param[in] tid   index in the source array
 * */
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

/**
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
 * */
template<
	uint32_t BLOCK_SIZE,
	Roulette roulette_type,
	bool     shmem
>
__device__ void create_roulette_wheel(
	float* values,
	float* roulette,
	float* shmem_sum,
	float* shmem_max,
	int    tid
)
{
	//sum reduction and max calculation
	shmem_sum[threadIdx.x] = values[tid];
	shmem_max[threadIdx.x] = values[tid];

	if(BLOCK_SIZE == 1024)
	{
		if(threadIdx.x < 512)
		{
			shmem_sum[threadIdx.x] += shmem_sum[threadIdx.x + 512];
			shmem_max[threadIdx.x] =  fmax(shmem_max[threadIdx.x],
			                               shmem_max[threadIdx.x + 512]);
		}
		__syncthreads();
	}
	
	if(BLOCK_SIZE >= 512)
	{
		if(threadIdx.x < 256)
		{
			shmem_sum[threadIdx.x] += shmem_sum[threadIdx.x + 256];
			shmem_max[threadIdx.x] =  fmax(shmem_max[threadIdx.x],
			                               shmem_max[threadIdx.x + 256]);
		}
		__syncthreads();
	}
	if(BLOCK_SIZE >= 256)
	{
		if(threadIdx.x < 128)
		{
			shmem_sum[threadIdx.x] += shmem_sum[threadIdx.x + 128];
			shmem_max[threadIdx.x] =  fmax(shmem_max[threadIdx.x],
			                               shmem_max[threadIdx.x + 128]);
		}
		__syncthreads();
	}
	if(BLOCK_SIZE >= 128)
	{
		if(threadIdx.x < 64)
		{
			shmem_sum[threadIdx.x] += shmem_sum[threadIdx.x + 64];
			shmem_max[threadIdx.x] =  fmax(shmem_max[threadIdx.x],
			                               shmem_max[threadIdx.x + 64]);
		}
		__syncthreads();
	}

	if(threadIdx.x < 32) warp_reduce_sum<BLOCK_SIZE>(shmem_sum, threadIdx.x);
	if(threadIdx.x < 32) warp_reduce_max<BLOCK_SIZE>(shmem_max, threadIdx.x);

	__syncthreads();

	//kakva normalizacija pravam tuka (max - value) / (max*count - sum)
	
	//! ova mora da go dopravam
	switch(roulette_type)
	{
	case FULL:
		roulette[threadIdx.x] = values[threadIdx.x] / shmem_sum[0];
		break;
	case PARTIAL:
		roulette[threadIdx.x] = (shmem_max[0] - values[threadIdx.x]) /
		                        (blockDim.x*shmem_max[0] - shmem_sum[0]);
		break;
	}

	__syncthreads();
}

/**
 * @brief Selects an element from the roulette wheel according to weighted
 *        roulette wheel selection
 *
 * @param[in] roulette initialized roulette wheel which represents the weights
 *                     of the underlying elements
 * @param[in] size     size of the roulette wheel
 * @param[in] state    curand state used for generating random numbers
 *
 * @return idx of the selected element
 *
 * @note This function expects the roulette wheel to not have a cumulative sum,
 *       the cumulative sum is performed during selection
 * */
template<uint32_t size>
__device__ int spin_roulette(
	float*       roulette,
	curandState* state
)
{
	//nema potreba za povikot poradi toa shto mi treba od 0 do 1
	//float choice = rand_bounded_float(state, 0, 1);
	float choice = curand_uniform(state);
	float sum = 0;
	for(int idx = 0; idx < size; idx++)
	{
		sum += roulette[idx];
		if(choice <= sum) return idx;
	}
	return 0;
}

template<
	uint32_t dim,
	int     num_of_bees
>
__device__ void sort_bees(
	float* cords,
	float* values,
	float* shmem_values,
	float* shmem_cords
)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	shmem_values[threadIdx.x] = values[tid];
	
	__syncthreads();
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
				float a = shmem_values[tid];
				float b = shmem_values[tid + stride];

				if((a > b) == ascending)
				{
					shmem_values[tid]        = b;
					shmem_values[tid+stride] = a;

					//#pragma unroll
					for(int i = 0; i < dim; i++)
					{
						float tmp = shmem_cords[tid];
						shmem_cords[tid] =shmem_cords[tid+stride];
						shmem_cords[tid+stride ] = tmp;
					}
				}
			}
			__syncthreads();
		}

	}
}

template<
	uint32_t dim,
	uint32_t size,
	bool     inverse,
	bool     shmem
>
__forceinline__ __device__ void rank_selection_from_arr(
	float*       sh_cords,
	float*       sh_values,
	int*         trials,
	int          tid,
	curandState* state,
	float*       rank_arr
)
{
	//! sortiraj gi site pcheli,
	
	int spin = spin_roulette<size>(rank_arr, state);

	if(sh_values[spin] > sh_values[tid])
	{
		sh_values[tid] = sh_values[spin];
		//#pragma unroll
		for(int i = 0; i < dim; i++)
			sh_cords[tid*dim + i] = sh_cords[spin*dim + i];

		(*trials) = 0;
	}
	else
	{
		(*trials)++;
	}
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
__forceinline__ __device__ void rank_selection_constant(
	float*       sh_cords,
	float*       sh_values,
	int*         trials,
	int          tid,
	curandState* state
)
{
	//! sortiraj gi site pcheli,
	
	int spin;
	float rand = curand_uniform(state);
	switch(rank_type)
	{
	case CONSTANT_LINEAR:
		spin = rank_constant::dev_lin<size>(rand); break;
	case CONSTANT_EXPONENTIAL:
		spin = rank_constant::dev_exp<size, c_num, c_div>(rand); break;
	case CONSTANT_EXPONENTIAL_2:
		spin = rank_constant::dev_exp2<size, c_num, c_div>(rand); break;
	default:
		spin = 0; break;
	}

	if(sh_values[spin] > sh_values[tid])
	{
		sh_values[tid] = sh_values[spin];
		//#pragma unroll
		for(int i = 0; i < dim; i++)
			sh_cords[tid*dim + i] = sh_cords[spin*dim + i];

		(*trials) = 0;
	}
	else
	{
		(*trials)++;
	}
}

template<
	uint32_t size,
	uint32_t dim,
	uint32_t tournament_size
>
__forceinline__ __device__ void tournament_selection_standard(
	float*      cords,
	float*      values,
	int*         trials,
	int          idx,
	curandState* state	
)
{
		int choices[tournament_size];
		int min_idx = -1;
		float min_value = values[idx];

		//#pragma unroll
		for(int j = 0; j < tournament_size; j++)
			choices[j] = (size-1) * curand_uniform(state);


		//#pragma unroll
		for(int j = 1; j < tournament_size; j++)
		{
			if(min_value < values[choices[j]])
			{
				min_value = values[choices[j]];
				min_idx   = j;
			}
		}

		if(values[idx] > min_value)
		{
			values[idx] = min_value;
			//#pragma unroll
			for(int i = 0; i < dim; i++)
				cords[idx*dim + i] = cords[min_idx*dim + i];

			(*trials) = 0;
		}
		else
		{
			(*trials)++;
		}	
}

template<
	uint32_t size,
	uint32_t dim,
	uint32_t num_of_games,
	uint32_t num_of_contestants
>
__forceinline__ __device__ void tournament_selection_custom(
	float*       cords,
	float*       values,
	int*         trials,
	int          idx,
	curandState* state
)
{
		int choices[num_of_games];
		int min_idx = -1;
		float min_value = values[idx];

		for(int i = 0; i < num_of_games; i++)
		{
			//#pragma unroll
			for(int j = 0; j < num_of_games; j++)
				choices[j] = (size-1) * curand_uniform(state);

			//#pragma unroll
			for(int j = 1; j < num_of_games; j++)
			{
				if(min_value < values[choices[j]])
				{
					min_value = values[choices[j]];
					min_idx   = j;
				}
			}
		}

		if(values[idx] > min_value)
		{
			values[idx] = min_value;
			//#pragma unroll
			for(int i = 0; i < dim; i++)
				cords[idx*dim + i] = cords[min_idx*dim + i];

			(*trials) = 0;
		}
		else
		{
			(*trials)++;
		}	
}

template<
	uint32_t dim,
	uint32_t BLOCK_SIZE,
	Roulette roulette_type,
	bool     inverse,
	bool     shmem
>
__forceinline__ __device__ void roulette_wheel_selection(
	float*       sh_values,
	float*       sh_cords,
	float*       sh_roulette, 
	float*       sh_sum,
	float*       sh_max,
	curandState* state,
	int*         trials,
	int          tid,
	int          gid
)
{
		create_roulette_wheel<BLOCK_SIZE, roulette_type, shmem>(
			sh_values,
			sh_roulette,
			sh_sum,
			sh_max,
			tid
		);

		int spin = spin_roulette<BLOCK_SIZE>(sh_roulette, state);

		if(sh_values[tid] > sh_values[spin])
		{
			//#pragma unroll
			for(int i = 0; i < dim; i++)
				sh_cords[tid*dim + i] = sh_cords[spin*dim + i];

			sh_values[threadIdx.x] = sh_values[spin];

			(*trials) = 0;
		}
		else
		{
			(*trials)++;
		}
}

/*
 * @brief Optimized parallel GPU version of the ABC algorithm
 *
 * @tparam[in] dim             the number of dimensions for the problem
 * @tparam[in] BLOCK_SIZE      the size of each block, used for min/max/sum
 * @tparam[in] inverse         determines whether to inverse the result
 * @tparam[in] selection_type  determines the selection algorithm used
 * @tparam[in] roulette_type   variant of roulette selection
 * @tparam[in] rank_type       variant of rank selection
 * @tparam[in] tournament_type variant of tournament selection
 * @tparam[in] shmem_enabled   determines if shared memory is used
 *
 * @param[inout] cords           Initial solution/food source cordinates
 * @param[inout] values          Initial solution/food source fitness values
 * @param[in]    num_of_bees     Total number of bees
 * @param[in]    max_generations Maximum number of iterations of the algorithm
 * @param[in]    trials_limit    Number of times a solution can't be improved
 *                               before being abandoned
 * @param[in]    function        Pointer to a function to be optimized,
 *                               of type (array(float), int) -> float
 * @param[in]    lower_bounds    Lower bounds of the search space
 * @param[in]    upper_bounds    Upper bounds of the search space
 *
 * TODO: Ova treba da go implementiram
 * @note Inter-bee communication occurs only at the per block level,
 *       in order to have cross block communicate interspersed communication
 *       passes are needed
 */
template<
	uint32_t   dim,
	uint32_t   BLOCK_SIZE,
	bool       inverse,
	Selection  selection_type,
	Roulette   roulette_type,
	Rank       rank_type,
	Tourn      tournament_type,
	bool       shmem_enabled
>
__global__ void abc(
	float*   cords,
	float*   values,
	int      num_of_bees, //? dali mi treba ova voopsto?
	int      max_generations, 
	int      trials_limit, 
	opt_func optimization_function,
	float*   lower_bounds,
	float*   upper_bounds
)
{
	int tid = threadIdx.x;
	int gid = tid + blockIdx.x * blockDim.x;
	//---------------------SHARED-MEMORY-DECLARATION-------------------------//
	extern __shared__ float shmem[];

	float *sh_cords, *sh_values, *sh_sum, *sh_max, *sh_roulette;

	if(shmem_enabled)
	{
		//za da se ostvari ova na 1024 threads dovolno e 36KB od shared memory
		//shto e vo 48KB opshtiot limit
		sh_cords  = shmem;
		sh_values = sh_cords + blockDim.x*dim;

		//TODO: treba da proveram dali go sobira vo prostorot na shmem 
		sh_sum      = sh_values + blockDim.x;
		sh_max      = sh_sum    + blockDim.x;
		sh_roulette = sh_max    + blockDim.x;
		//za roulette kje ja koristam istata niza poradi nedovolna memorija

		//Load values into shared memory
		//#pragma unroll
		for(int i = 0; i < dim; i++)
			sh_cords[tid*dim + i] = cords[gid*dim + i];

		float tmp_value = optimization_function(&sh_cords[tid*dim], dim);
		sh_values[tid] = tmp_value;

		__syncthreads();
	}
	else
	{
		sh_cords  = cords;
		sh_values = values;
		
		//! ova sega za sega e sekogash so shared memory
		sh_sum      = shmem;
		sh_max      = sh_sum + blockDim.x;
		sh_roulette = sh_max + blockDim.x;
	}
	//-----------------------------------------------------------------------//

	//------------------------CURAND-INITIALIZATION--------------------------//
	curandState state;
	curand_init(clock64() + gid, 0, 0, &state);
	//-----------------------------------------------------------------------//
	
	//----------------------RANK-ARR-INITIALIZATION--------------------------//
	float* rank_arr;
	if(selection_type == RANK && rank_type < CONSTANT_LINEAR)
	{
		//sega za ova treba ubavo da go razmislam deka kje se osnova
		//na toa da deklariram __constant__ niza so constexpr
		//! ova realno najpametno bi bilo pred toa da se deklarira vo cpu delot
	}
	//-----------------------------------------------------------------------//

	//----------------------INITIALIZE-INITIAL-STATE-------------------------//
	generate_random_solution<dim>(
			sh_cords,
			sh_values,
			lower_bounds,
			upper_bounds,
			threadIdx.x,
			optimization_function,
			&state
	);
	int trials = 0;

	__syncthreads();
	//-----------------------------------------------------------------------//

	//--------------------------------MAIN-LOOOP-----------------------------//
	for(int i = 0; i < max_generations; i++)
	{
		//------------------EMPLOYED-BEE-LOCAL-OPTIMIZATION------------------//
		local_optimization<dim>(
			threadIdx.x,
			sh_cords,
			sh_values,
			lower_bounds,
			upper_bounds,
			&trials,
			optimization_function,
			&state
		);
		__syncthreads();
		//-------------------------------------------------------------------//

		//------------------ONLOOKER-BEE-GLOBAL-OPTIMIZATION-----------------//
		switch(selection_type)
		{
			case ROULETTE_WHEEL: 
				//TODO: treba da go dopravam tuka toggle-ot za shmem,
				//      i isto taka za inverse
				roulette_wheel_selection<
					dim,
					BLOCK_SIZE,
					roulette_type,
					inverse,
					shmem_enabled
				>(
					sh_values,
					sh_cords,
					sh_roulette,
					sh_sum,
					sh_max,
					&state,
					&trials,
					tid,
					gid
				);
			case RANK:
				if(rank_type < CONSTANT_LINEAR)
					rank_selection_from_arr<
						dim,
						BLOCK_SIZE,
						inverse,
						shmem_enabled
					>(
						sh_cords,
						sh_values,
						&trials,
						tid,
						&state,
						rank_arr
					);
				else
					rank_selection_constant<
						dim,
						BLOCK_SIZE,
						rank_type,
						5, 10,
						false,
						true
					>(
						sh_cords,
						sh_values,
						&trials,
						tid,
						&state
					);
		}
		//-------------------------------------------------------------------//

		//------------------------TRIAL-LIMIT-CHECK--------------------------//
		if(trials > trials_limit)
		{
			generate_random_solution<dim>(
				sh_cords,
				sh_values,
				lower_bounds,
				upper_bounds,
				threadIdx.x,
				optimization_function,
				&state
			);
			trials = 0;
		}
		//-------------------------------------------------------------------//
	}

	//-----------------------------RETURN-VARIABLES--------------------------//
	values[gid] = sh_values[tid];

	//#pragma unroll
	for(int i = 0; i < dim; i++)
		cords[gid*dim + i] = sh_cords[tid*dim + i];
	//-----------------------------------------------------------------------//
}

/**
 * @brief Function pointer used to assign the needed optimization function
 * @details Due to the cpu and gpu memory spaces being separate, in order to
 *          pass a function pointer to the kernel, the CPU code must first
 *          copy the symbol location of a GPU assigned function pointer
 * */
__device__ opt_func d_optimization_function = problems::gpu::rosenbrock;

//TODO: Treba funkcijava da se prefrli vo template i da se implementira
//      funkcionalnosta na logiranje strukturirani csv podatoci na sekoj step
#define DIMENSIONS 2

/*
 * @brief Function used to launch the parallel GPU version of the ABC algorithm
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
//TODO: odkomentirano deka me mrzi da go prefrlam vo headerot sega za sega
//template<uint32_t DIMENSIONS>
void gpu::launch_abc(
	float*   cords,
	float*   values,
	int      num_of_bees, 
	int      max_generations, 
	int      trials_limit, 
	opt_func optimization_function,
	float    lower_bounds[],
	float    upper_bounds[],
	int      steps
)
{
	float* d_cords;
	float* d_values;
	float* d_upper_bounds;
	float* d_lower_bounds;

	size_t cords_size  = DIMENSIONS  * num_of_bees * sizeof(float);
	size_t values_size = num_of_bees * sizeof(float);
	size_t bounds_size = DIMENSIONS  * sizeof(float);

	cudaMalloc((void**) &d_cords, cords_size);
	cudaMalloc((void**) &d_values, values_size);

	//TODO: bounds treba da gi stavam vo constant memory
	cudaMalloc((void**) &d_upper_bounds, bounds_size);
	cudaMalloc((void**) &d_lower_bounds, bounds_size);

	cudaMemcpy(
		d_cords,
		cords,
		cords_size,
		cudaMemcpyHostToDevice
	);
	cudaMemcpy(
		d_values,
		values,
		values_size,
		cudaMemcpyHostToDevice
	);
	cudaMemcpy(
		d_upper_bounds,
		upper_bounds,
		bounds_size,
		cudaMemcpyHostToDevice
	);
	cudaMemcpy(
		d_lower_bounds,
		lower_bounds,
		bounds_size,
		cudaMemcpyHostToDevice
	);
	
	cudaMemcpyFromSymbol(
		&optimization_function,
		d_optimization_function,
		sizeof(opt_func),
		0,
		cudaMemcpyDeviceToHost
	);

	//TODO: treba da isprobam so 1024, ne znam dali kje raboti?
	size_t GRID_SIZE  = 100;
	size_t BLOCK_SIZE = 512;

	size_t SHMEM_SIZE =
		BLOCK_SIZE * sizeof(float) * DIMENSIONS + //sh_cords
	    BLOCK_SIZE * sizeof(float)              + //sh_values
	    BLOCK_SIZE * sizeof(float)              + //sh_sum
	    BLOCK_SIZE * sizeof(float)              + //sh_max
	    BLOCK_SIZE * sizeof(float);               //sh_roulette

	cudaDeviceSynchronize();

	abc<
		DIMENSIONS,
		512,
		true,
		ROULETTE_WHEEL,
		FULL,
		LINEAR_ARRAY,
		SINGLE,
		true
	><<<GRID_SIZE, BLOCK_SIZE, SHMEM_SIZE>>>(
		d_cords,
		d_values,
		num_of_bees,
		max_generations,
		trials_limit,
		optimization_function,
		d_lower_bounds,
		d_upper_bounds
	);

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
}
