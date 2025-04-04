/******************************************************************************
 * @file abc_gpu.cu                                                           *
 * @brief Parallel GPU implementation of the ABC algorithm                    *
 * @author Kristijan Stameski                                                 *
 ******************************************************************************/

#include <math.h>
#include <time.h>
#include "abc_gpu.cuh"
#include "problems/problems.h"
#include "problems/gpu/problems_valley_shaped.cuh"

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

/**
 * @brief Generates a random int within the designated bounds
 * @param[in] state Curand state which must be initialized beforehand
 * @param[in] lower_bound
 * @param[in] upper_bound
 *
 * @return random int
 * */
__forceinline__ __device__ double rand_bounded_int(
		curandState* state,
		int lower_bound,
		int upper_bound
)
{
	int range = upper_bound - lower_bound;

	return lower_bound + (int)(curand_uniform(state) * range);
}

/**
 * @brief Optimized function used to clip a float to certain bounds
 * @param[in] lower_bound
 * @param[in] upper_bound
 * @return clipped float
 * */
__forceinline__ __device__ float fast_clip_float(
	float n,
	float lower_bound,
	float upper_bound
)
{
	n = (n > lower_bound) * n + !(n > lower_bound) * lower_bound;
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
	n = (n > lower_bound) * n + !(n > lower_bound) * lower_bound;
	return (n < upper_bound) * n + !(n < upper_bound) * upper_bound;
}

//TODO: ova treba da bide napraveno vo template
//      treba da napravam da ne prima nizi tuku specifichni indeksi/vrednosti
//      poradi toa shto nema potreba od celata niza
/**
 * @brief Generate a random solution/food source within a hypercube defined
 *        by the given bounds
 *
 * @param[inout] coordinates an array of doubles which includes all coordinates 
 * @param[inout] values      an array of doubles which includes all values
 * @param[in]    dimensions  the dimensionality of the problem
 * @param[in]    idx         the position of the bee in the given arrays
 * @param[in]    function    function to be optimized
 * */
__inline__ __device__ void generate_random_solution(
		double*  coordinates,
		double*  values,
		int      dimensions,
		int      idx,
		opt_func function,
		double   lower_bounds[], //TODO: za optimizacija treba da e __constant__
		double   upper_bounds[],
		curandState* state
)
{
	//init random number
	for(int i = 0; i < dimensions; i++)
	{
		coordinates[idx*dimensions+i] = 
			rand_bounded_double(
				state,
				lower_bounds[i],
				upper_bounds[i]
			);
	}

	values[idx] = function(&coordinates[idx], dimensions);
}

#define MAX_DIMENSIONS 6

/**
 * @brief Local optimization around the existing food source
 * @param[in] idx          index of the bee/thread to be optimized
 * @param[in] coordinates  an coordinate array of all candidate bees/threads
 * @param[in] lower_bounds 
 * @param[in] upper_bounds
 * @details Randomly select another bee and merge the solutions with
 *          a stochastic step
 *
 * @note The index is passed in because of simpler shared memory optimization
 * */
__forceinline__ __device__ void local_optimization(
		int          idx,
		double*      coordinates,
		double*      values,
		int*         trials,
		int          dimensions,
		opt_func     function,
		double*      lower_bounds,
		double*      upper_bounds,
		curandState* state
)
{
	int choice = rand_bounded_int(
		state,
		0,
		blockDim.x
	);
	double tmp_coordinates[MAX_DIMENSIONS];

	//TODO: Treba tuka da ima selekcija za druga pchela 

	for(int i = 0; i < dimensions; i++)
	{
		double step = coordinates[idx*dimensions    + i] -
		              coordinates[choice*dimensions + i];
		step *= curand_uniform(state);

		tmp_coordinates[i] = coordinates[idx*dimensions + i] + step;

		tmp_coordinates[i] = fast_clip_double(
			tmp_coordinates[i],
			lower_bounds[i],
			upper_bounds[i]
		);
	}

	double tmp_value = function(
			tmp_coordinates,
			dimensions
	);

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
template<unsigned int BLOCK_SIZE>
__forceinline__ __device__ void warp_reduce_sum(volatile double* sdata, int tid)
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
template<unsigned int BLOCK_SIZE>
__forceinline__ __device__ void warp_reduce_max(volatile double* sdata, int tid)
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
 * @param[in]  values   array of doubles based on which the roulette wheel
 *                      will be constructed
 * @param[out] roulette  array of double values needed for weighted roulette
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
template<unsigned int BLOCK_SIZE>
__device__ void create_roulette_wheel(
	double* values,
	double* roulette,
	double* shmem_sum,
	double* shmem_max,
	int     tid
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

	roulette[threadIdx.x] = (shmem_max[0] - values[threadIdx.x]) /
	                        (blockDim.x*shmem_max[0] - shmem_sum[0]);

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
__device__ int spin_roulette(
		double*      roulette,
		int          size,
		curandState* state
)
{
	//nema potreba za povikot poradi toa shto mi treba od 0 do 1
	//double choice = rand_bounded_double(state, 0, 1);
	double choice = curand_uniform(state);
	double sum = 0;
	for(int idx = 0; idx < size; idx++)
	{
		sum += roulette[idx];
		if(choice <= sum)
			return idx;
	}
	return 0;
}

/**
 * @brief Optimized parallel GPU version of the ABC algorithm
 *
 * @param[inout] coordinates     Initial solution/food source coordinates
 * @param[inout] values          Initial solution/food source fitness values
 * @param[in]    num_of_bees     Total number of bees
 * @param[in]    num_of_scouts   Total number of scout bees
 * @param[in]    max_generations Maximum number of iterations of the algorithm
 * @param[in]    trials_limit    Number of times a solution can't be improved
 *                               before being abandoned
 * @param[in]    function        Pointer to a function to be optimized,
 *                               of type (array(double), int) -> double
 * @param[in]    lower_bounds    Lower bounds of the search space
 * @param[in]    upper_bounds    Upper bounds of the search space
 *
 * @note Inter-bee communication occurs only at the per block level,
 *       in order to have cross block communicate interspersed communication
 *       passes are needed
 * */
template<
	unsigned int dimensions,
	unsigned int BLOCK_SIZE
>
__global__ void abc(
	double*  coordinates,
	double*  values,
	int      num_of_bees,   //? dali mi treba ova voopsto?
	int      num_of_scouts, //? dali mi treba ova voopsto? 
	int      max_generations, 
	int      trials_limit, 
	opt_func optimization_function,
	double*  lower_bounds,
	double*  upper_bounds
)
{
	//----------------------SHARED-MEMORY-DECLARATION-------------------------//
	extern __shared__ double shmem[];

	//za da se ostvari ova na 1024 threads dovolno e 36KB od shared memory
	//shto e vo 48KB opshtiot limit
	double* sh_coordinates = shmem;
	double* sh_values      = shmem + blockDim.x*dimensions;
	//int*    sh_trials      = (int*)    sh_values + sizeof(double)*num_of_bees;

	//TODO: treba da proveram ova dali mozhe da go sobere vo prostorot na shmem 
	double* sh_sum      = sh_values + blockDim.x;
	double* sh_max      = sh_sum    + blockDim.x;
	double* sh_roulette = sh_max    + blockDim.x;
	//za roulette kje ja koristam istata niza poradi nedovolna memorija
	
	int global_idx = threadIdx.x + blockIdx.x * blockDim.x;

	//Load values into shared memory
	//ova treba da go napravam so templates za for loop ekspanzija
	for(int i = 0; i < dimensions; i++)
	{
		sh_coordinates[threadIdx.x*dimensions + i] = 
			coordinates[global_idx*dimensions + i];
	}
	
	sh_values[threadIdx.x] = optimization_function(
		&sh_coordinates[threadIdx.x*dimensions],
		dimensions
	);
	//sh_trials[threadIdx.x] = 0; //? ova mozhe i da ne e potrebno so memset
	//------------------------------------------------------------------------//
	

	//------------------------CURAND-INITIALIZATION---------------------------//
	curandState state;
	curand_init(clock64() + global_idx, 0, 0, &state);
	//------------------------------------------------------------------------//

	//----------------------INITIALIZE-INITIAL-STATE--------------------------//
	//TODO: Ova treba da se izvadi od tuka
	generate_random_solution(
			sh_coordinates,
			sh_values,
			dimensions,
			threadIdx.x,
			optimization_function,
			lower_bounds,
			upper_bounds,
			&state
	);
	int trials = 0;

	__syncthreads();
	//------------------------------------------------------------------------//

	//--------------------------------MAIN-LOOOP------------------------------//
	for(int i = 0; i < max_generations; i++)
	{
		//------------------EMPLOYED-BEE-LOCAL-OPTIMIZATION-------------------//
		//TODO: napravi go so template
		local_optimization(
				threadIdx.x,
				sh_coordinates,
				sh_values,
				&trials,
				dimensions,
				optimization_function,
				lower_bounds,
				upper_bounds,
				&state
		);
		__syncthreads();
		//--------------------------------------------------------------------//

		//------------------ONLOOKER-BEE-GLOBAL-OPTIMIZATION------------------//
		//TODO: refaktoriraj go ova kako edna opcija na selekcija
		//Roulette selection
		//initialize roulette
		create_roulette_wheel<BLOCK_SIZE>(
				sh_values,
				sh_roulette,
				sh_sum,
				sh_max,
				threadIdx.x
		);

		//select from roulette
		int spin = spin_roulette(
				sh_roulette,
				BLOCK_SIZE,
				&state
		);

		if(sh_values[threadIdx.x] > sh_values[spin])
		{
			for(int i = 0; i < dimensions; i++)
			{
				sh_coordinates[threadIdx.x*dimensions + i] = 
					sh_coordinates[spin*dimensions + i];
			}
			sh_values[threadIdx.x] = sh_values[spin];
			trials = 0;
		}
		else
		{
			trials++;
		}
		//--------------------------------------------------------------------//

		//-------------------------TRIAL-LIMIT-CHECK--------------------------//
		if(trials > trials_limit)
		{
			generate_random_solution(
					sh_coordinates,
					sh_values,
					dimensions,
					threadIdx.x,
					optimization_function,
					lower_bounds,
					upper_bounds,
					&state
			);
			trials = 0;
		}
		//--------------------------------------------------------------------//
	}

	//-----------------------------RETURN-VARIABLES---------------------------//
	values[global_idx] = sh_values[threadIdx.x];
	for(int i = 0; i < dimensions; i++)
	{
		coordinates[global_idx*dimensions + i] = 
			sh_coordinates[threadIdx.x*dimensions + i];
	}
	//------------------------------------------------------------------------//
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

/**
 * @brief Function used to launch the parallel GPU version of the ABC algorithm
 * 
 * @param[out] coordinates    array of potential solutions coordinates 
 * @param[out] values         array of potential solutions fitness values 
 * @param[in] num_of_bees     total number of bees
 * @param[in] max_generations maximum number of iterations of the algorithm
 * @param[in] trials_limit    the number of times a solution can't be improved
 *                            before being discarded
 * @param[in] function        pointer to the function to be optimized,
 *                            of type (array(double), int) -> double
 * @param[in] lower_bounds    lower bounds of the search space
 * @param[in] upper_bounds    upper bounds of the search space
 * @param[in] steps           number of steps over which to execute an 
 *                            optimization cycle, used for testing convergance
 *                            rates
 *
 * @details This function is used to call the GPU kernel and handles
 *          inter-block communication and probes the state of the bee colony
 *          at a set number of steps
//template<unsigned int DIMENSIONS> //TODO: dopravi go template-ot
 * */
void gpu::launch_abc(
	double*  coordinates,
	double*  values,
	int      num_of_bees, 
	int      max_generations, 
	int      trials_limit, 
	opt_func optimization_function,
	double   lower_bounds[],
	double   upper_bounds[],
	int      steps
)
{
	double*  d_coordinates;
	double*  d_values;
	double*  d_upper_bounds;
	double*  d_lower_bounds;

	size_t coordinates_size = num_of_bees*2*sizeof(double);
	size_t values_size      = num_of_bees*sizeof(double);
	size_t trials_size      = num_of_bees*sizeof(int);
	size_t bounds_size      = 2*sizeof(double);

	cudaMalloc((void**) &d_coordinates, coordinates_size);
	cudaMalloc((void**) &d_values, values_size);

	//TODO: bounds treba da gi stavam vo constant memory
	cudaMalloc((void**) &d_upper_bounds, bounds_size);
	cudaMalloc((void**) &d_lower_bounds, bounds_size);

	cudaMemcpy(
		d_coordinates,
		coordinates,
		coordinates_size,
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

	size_t BLOCK_SIZE = 512;
	size_t GRID_SIZE  = 100; 

	size_t SHMEM_SIZE = BLOCK_SIZE * sizeof(double) * 2 + //sh_coordinates
	                    BLOCK_SIZE * sizeof(double)     + //sh_values
	                    BLOCK_SIZE * sizeof(double)     + //sh_sum
	                    BLOCK_SIZE * sizeof(double)     + //sh_max
	                    BLOCK_SIZE * sizeof(double);      //sh_roulette

	cudaDeviceSynchronize();

	abc<2, 512><<<GRID_SIZE, BLOCK_SIZE, SHMEM_SIZE>>>(
		d_coordinates,
		d_values,
		num_of_bees,
		max_generations,
		trials_limit,
		1.0,
		optimization_function,
		d_lower_bounds,
		d_upper_bounds
	);

	cudaMemcpy(
		coordinates,
		d_coordinates,
		coordinates_size,
		cudaMemcpyDeviceToHost
	);
	cudaMemcpy(
		values,
		d_values,
		values_size,
		cudaMemcpyDeviceToHost
	);
}
