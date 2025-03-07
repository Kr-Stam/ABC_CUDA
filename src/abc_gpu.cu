//#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <time.h>
#include <stdio.h>
#include "abc_gpu.cuh"
#include "problems/problems.cuh"

__global__ void init_curand_states(curandState * state, unsigned long seed)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed, idx, 0, &state[idx]);
}

__device__ double bohachevsky1(double* args, int n)
{
	if(n < 2) return 0;

	return args[0]*args[0] + 2*args[1]*args[1] -
		   0.3*std::cos(3*M_PI*args[0]) -
		   0.4*std::cos(4*M_PI*args[1]) + 0.7;
}

__forceinline__ __device__ double rand_bounded_double(
		curandState state,
		double lower_bound,
		double upper_bound
)
{
	double range = upper_bound - lower_bound;

	return lower_bound + curand_uniform(&state) * range;
}

__forceinline__ __device__ double rand_bounded_int(
		curandState state,
		int lower_bound,
		int upper_bound
)
{
	int range = upper_bound - lower_bound;

	return lower_bound + (int)(curand_uniform(&state) * range);
}

__inline__ __device__ void generate_random_solution(
		double*  coordinates,
		double*  values,
		int      dimensions,
		int      idx,
		opt_func function,
		double   lower_bounds[], //TODO: treba ovie da gi loadiram vo __constant__
		double   upper_bounds[],
		curandState* states
)
{
	//init random number
	curandState local_state = states[idx];
	for(int i = 0; i < dimensions; i++)
	{
		coordinates[idx*dimensions] = 
			rand_bounded_double(
				local_state,
				lower_bounds[i],
				upper_bounds[i]
		);
	}

	values[idx] = 
		function(
			&coordinates[idx],
			dimensions
		);
}

__forceinline__ __device__ float fast_clip(float n, float lower, float upper)
{
	n = (n > lower) * n + !(n > lower) * lower;
	return (n < upper) * n + !(n < upper) * upper;
}

#define MAX_DIMENSIONS 6

__forceinline__ __device__ void local_optimization(
		int         idx,
		double*     coordinates,
		double*     values,
		int*        trials,
		int         dimensions,
		opt_func    function,
		double*     lower_bounds,
		double*     upper_bounds,
		curandState state
)
{
	int choice = rand_bounded_int(state, 0, blockDim.x);
	double tmp_coordinates[MAX_DIMENSIONS];

	//TODO: Treba tuka da ima selekcija za druga pchela 

	for(int i = 0; i < dimensions; i++)
	{
		double step = coordinates[idx] - coordinates[choice*dimensions + i];
		step *= curand_uniform(&state);

		tmp_coordinates[i] = coordinates[idx*dimensions + i] + step;

		tmp_coordinates[i] = fast_clip(
			tmp_coordinates[i],
			lower_bounds[i],
			upper_bounds[i]
		);
	}

	double tmp_value = function(
			&coordinates[idx*dimensions],
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

//? zoshto go napraviv ova koga znam deka ne e optimalniot
//? izbor za paraleliziran algoritam ne znam
template<unsigned int BLOCK_SIZE>
__device__ void create_roulette_wheel(
	double* values,
	double* roulette,
	double* shmem_sum,
	double* shmem_max,
	int num_of_candidates,
	int total_num
)
{
	//sum reduction
	//treba da se napravi sum reduction na celata niza i da se
	//skalira so maksimalnata vrednost

	//1 2 3 4 5
	//sum: dwa`]
	//max: 5
	//(max-element)/sum

	//Calculate max and sum
	int tid = threadIdx.x + blockIdx.x * blockDim.x * 2;

	unsigned int grid_size = BLOCK_SIZE * 2 * gridDim.x;

	while(tid < num_of_candidates)
	{
		shmem_sum[threadIdx.x] += values[tid] + values[tid + BLOCK_SIZE];
		shmem_max[threadIdx.x] += fmax(values[tid], values[tid + BLOCK_SIZE]);
		tid += grid_size;
	}
	__syncthreads();
	
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
	if(threadIdx.x < 32) warp_reduce_max<BLOCK_SIZE>(shmem_sum, threadIdx.x);

	__syncthreads();

	roulette[tid] = shmem_max[0] - values[tid] / shmem_sum[0];

	//cumulative distribution
	//? mora da se pomine barem ednash vo for loop pri detekcija
	//? pa poradi toa nema nekoja prichina sega da se napravi kumulativna
	//? suma, mozhebi bi bilo pobrzo poradi shared memorija,
	//? ama i opshto mnogu golem warp divergence kje ima tuka
	//? poradi toa shto na razlichni delovi kje sopre
}

__device__ int spin_roulette(
		double*     roulette,
		int         size,
		curandState state
)
{
	//nema potreba za povikot poradi toa shto mi treba od 0 do 1
	//double choice = rand_bounded_double(state, 0, 1);
	double choice = curand_uniform(&state);
	double sum = 0;
	for(int idx = 0; idx < size; idx++)
	{
		sum += roulette[idx];
		if(choice <= sum)
			return idx;
	}
	return 0;
}


template<
	unsigned int dimensions,
	unsigned int BLOCK_SIZE
>
__global__ void abc(
	double*      coordinates,
	double*      values,
	//int*         trials,
	int          num_of_bees, 
	int          num_of_scouts, 
	int          max_generations, 
	int          trials_limit, 
	double       ratio_of_scouts,
	opt_func     optimization_function,
	double       lower_bounds[],
	double       upper_bounds[],
	curandState* curand_states //TODO: treba da proveram kako mi se deklarira ova
)
{
	extern __shared__ char shmem[];

	//za da se ostvari ova na 1024 threads dovolno e 36KB od shared memory
	//shto e vo 48KB opshtiot limit
	double* sh_coordinates = (double*) shmem;
	double* sh_values      = (double*) sh_coordinates + sizeof(double)*num_of_bees*dimensions;
	//int*    sh_trials      = (int*)    sh_values + sizeof(double)*num_of_bees;

	//TODO: treba da proveram ova dali mozhe da go sobere vo prostorot na shmem 
	double* sh_sum      = (double*) sh_values + sizeof(double)*num_of_bees;
	double* sh_max      = (double*) sh_sum    + sizeof(double)*num_of_bees;
	double* sh_roulette = (double*) sh_max    + sizeof(double)*num_of_bees;
	//za roulette kje ja koristam istata niza poradi nedovolna memorija

	int global_idx = threadIdx.x + blockIdx.x * blockDim.x;

	//Load values into shared memory
	//ova treba da go napravam so templates za for loop ekspanzija
	for(int i = 0; i < dimensions; i++)
	{
		sh_coordinates[threadIdx.x*dimensions + i] = 
			coordinates[global_idx*dimensions + i];
	}
	
	sh_values[threadIdx.x] = optimization_function(sh_coordinates, dimensions);
	//sh_trials[threadIdx.x] = 0; //? ova mozhe i da ne e potrebno so memset

	__syncthreads();

	//? ovaa funkcija mozhebi mozhe da se podobri,
	//? realno dimensions treba da e vo template
	generate_random_solution(
			sh_coordinates,
			sh_values,
			dimensions,
			threadIdx.x,
			optimization_function,
			lower_bounds,
			upper_bounds,
			curand_states
	);
	int trials = 0;

	//Main Loop
	for(int i = 0; i < max_generations; i++)
	{
		//Employed Bee Local Optimization
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
				curand_states[threadIdx.x]
		);

		//Sort
		//std::sort((*bees).begin(), (*bees).end(), BeeCompare);
		__syncthreads();

		//Roulette selection
		//initialize roulette
		create_roulette_wheel<BLOCK_SIZE>(
				sh_values,
				sh_roulette,
				sh_sum,
				sh_max,
				num_of_bees, //? treba ova da go proveram
				num_of_bees
		);

		//select from roulette
		int spin = spin_roulette(
				sh_roulette,
				num_of_bees,
				curand_states[threadIdx.x]
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

		//Search for new solutions if over trials
		for(int bee_idx = 0; bee_idx < num_of_bees; bee_idx++)
		{
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
						curand_states
				);
				trials = 0;
			}
		}
	}
}

__device__ opt_func d_optimization_function = problems::rosenbrock;

template<unsigned int DIMENSIONS>
void gpu::launch_abc(
	double*      coordinates,
	double*      values,
	int*         trials,
	int          num_of_bees, 
	int          num_of_scouts, 
	int          max_generations, 
	int          trials_limit, 
	double       ratio_of_scouts,
	opt_func     optimization_function,
	double       lower_bounds[],
	double       upper_bounds[]
)
{
	double*  d_coordinates;
	double*  d_values;
	int*     d_trials;

	size_t coordinates_size = num_of_bees*DIMENSIONS*sizeof(double);
	size_t values_size = num_of_bees*sizeof(double);
	size_t trials_size = num_of_bees*sizeof(int);

	cudaMalloc((void**) &d_coordinates, coordinates_size);
	cudaMalloc((void**) &d_values, values_size);
	cudaMalloc((void**) &d_trials, trials_size);

	cudaMemcpy(d_coordinates, coordinates, coordinates_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_values, values, values_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_trials, trials, trials_size, cudaMemcpyHostToDevice);
	
	cudaMemcpyFromSymbol(&optimization_function, d_optimization_function, sizeof(opt_func), 0, cudaMemcpyDeviceToHost);

	size_t BLOCK_SIZE = 1024; 
	size_t GRID_SIZE  =  100; 

	size_t SHMEM_SIZE = BLOCK_SIZE * sizeof(double) * DIMENSIONS + //sh_coordinates
	                    BLOCK_SIZE * sizeof(double) +              //sh_values
	                    BLOCK_SIZE * sizeof(int)    +              //sh_trials
	                    BLOCK_SIZE * sizeof(double) +              //sh_sum
	                    BLOCK_SIZE * sizeof(double) +              //sh_max
	                    BLOCK_SIZE * sizeof(double);               //sh_roulette

	curandState* curand_states;
	cudaMalloc((void**) &curand_states, BLOCK_SIZE*GRID_SIZE*sizeof(curandState));

	init_curand_states<<<GRID_SIZE, BLOCK_SIZE>>>(curand_states, time(NULL));

	cudaDeviceSynchronize();

	abc<512, DIMENSIONS><<<GRID_SIZE, BLOCK_SIZE, SHMEM_SIZE>>>(
			d_coordinates,
			d_values,
			d_trials,
			num_of_bees,
			num_of_bees, //TODO: num_of_scouts najverojatno kje go brisham
			max_generations,
			trials_limit,
			1.0,
			optimization_function,
			lower_bounds,
			upper_bounds,
			curand_states
	);

	cudaMemcpy(coordinates, d_coordinates, coordinates_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(values, d_values, values_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(trials, d_trials, trials_size, cudaMemcpyDeviceToHost);
}
