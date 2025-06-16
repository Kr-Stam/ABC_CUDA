/******************************************************************************
 * @file main.cu                                                              *
 * @brief Testing and measuring execution time and accuracy of the sequential *
 *        and parallel implementation of the ABC (Artificial Bee Colony)      *
 *        algorithm                                                           *
 * @author Kristijan Stameski                                                 *
 ******************************************************************************/

#include <stdio.h>
#include <math.h>
#include <array>
#include "abc_cpu.cuh"
#include "abc_gpu.cuh"
#include "problems/problems.h"
#include "problems/gpu/many_local_minima.cuh"
#include "problems/gpu/valley_shaped.cuh"
#include "utils/array.hpp"

//kje mora da gi izvadam kako vrednosti posle
using namespace abc_shared;

template<
	uint32_t dimensions,
	uint32_t bees_count,
	uint32_t max_trials
>
void abc_cpu_test(
	opt_func optimization_function,
	float    lower_bound[],
	float    upper_bound[],
	int      iterations
)
{
	std::vector<cpu::Bee> bees(bees_count);

	cpu::init_bees<dimensions>(
		&bees,
		bees_count,
		optimization_function,
		lower_bound,
		upper_bound
	);

	printf("Start\n");


	//debugging
	printf("Printf debugging ABC\n");
	int num_of_steps = 10;
	int step = iterations / num_of_steps;
	for(int i = 0; i < num_of_steps; i++)
	{
		cpu::abc<
			dimensions,
			bees_count,
			bees_count/10,
			max_trials,
			true,
			//ROULETTE_WHEEL,
			RANK,
			CUSTOM,
			PARTIAL_SORT,
			//LINEAR_ARRAY,
			CONSTANT_LINEAR,
			SINGLE,
			bees_count/10,
			3
		>(
			&bees,
			step,
			optimization_function,
			lower_bound,
			upper_bound
		);

		cpu::Bee min_bee = cpu::min_bee(&bees, bees_count);
		printf("Iteration #%03d: ", i*step);

		for(int j = 0; j <= dimensions; j++)
			printf("x%d: %.2f ", j, min_bee.coordinates[j]);

		printf("value: %.2f\n", min_bee.value);
	}

	printf("Done\n");

	//ova debagiranje isto taka sakam da go smenam da e	pogenerichno
	std::vector<float> coordinates_x(bees_count);
	std::vector<float> coordinates_y(bees_count);
	//std::vector<double> values(num_of_bees);

	std::sort(bees.begin(), bees.end(), cpu::BeeCompare);
	double values[bees_count];
	for(int bee_idx = 0; bee_idx < bees_count; bee_idx++)
	{
		coordinates_x[bee_idx] = bees[bee_idx].coordinates[0];
		coordinates_y[bee_idx] = bees[bee_idx].coordinates[1];
		values[bee_idx] = bees[bee_idx].value;
	}
	utils::array::print_array_double(values, 10);
	for(int bee_idx = 0; bee_idx < 10; bee_idx++)
	{
		printf(
			"x: %.2f y: %.2f value: %.2f\n",
			bees[bee_idx].coordinates[0],
			bees[bee_idx].coordinates[1],
			bees[bee_idx].value
		);
	}
}

#define DIMENSIONS 2

void abc_gpu_test()
{
	float lower_bound[DIMENSIONS] = {-10.0f, -10.0f};
	float upper_bound[DIMENSIONS] = {+10.0f, +10.0f};
	//opt_func optimization_function = problems::gpu::cross_in_tray;
	opt_func optimization_function = problems::gpu::rosenbrock;

	//variable initialization
	const size_t grid_size  = 100;
	const size_t block_size = 512;

	int num_of_bees     = grid_size*block_size;
	int max_generations = 10000;
  	int max_trials      = 20;

	float* cords  = (float*) malloc(num_of_bees * sizeof(float) * DIMENSIONS);
	float* values = (float*) malloc(num_of_bees * sizeof(float));
	gpu::launch_abc
	<
		DIMENSIONS,
		grid_size,
		block_size,
		//ROULETTE_WHEEL,
		RANK,
		//TOURNAMENT,
		MIN_MAX_SCALED,
		LINEAR_ARRAY,
		//EXPONENTIAL_ARRAY,
		//CONSTANT_LINEAR,
		//CONSTANT_EXPONENTIAL,
		//CONSTANT_EXPONENTIAL_2,
		SINGLE,
		//MULTIPLE,
		10,
		2
	>
	(
		cords,
		values,
		max_generations,
		max_trials,
		optimization_function,
		lower_bound,
		upper_bound,
		10
	);

	//proverka
	for(int i = 0; i < 10; i++)
	{
		float min = values[i];
		int min_idx;
		for(int j = i; j < num_of_bees; j++)
		{
			if(min > values[j])
			{
				min_idx = j;
				min = values[j];
			}
		}
		float tmp = values[i];
		values[i] = values[min_idx];
		values[min_idx] = tmp;

		for(int k = 0; k < DIMENSIONS; k++)
		{
			tmp = cords[i*DIMENSIONS + k];
			cords[i*DIMENSIONS + k] = cords[min_idx*DIMENSIONS + k];
			cords[min_idx*DIMENSIONS + k] = tmp;
		}

	}
	for(int i = 0; i < 10; i++)
	{
		printf(
			"Bee%03d: x=%f y=%f value=%f\n",
			i,
			cords[i*DIMENSIONS + 0],
			cords[i*DIMENSIONS + 1],
			values[i]
		);
	}
}

int main()
{
	float lower_bounds[] = {-10, -10};
	float upper_bounds[] = { 10,  10};
	//abc_cpu_test<2, 1000, 10>(
	//	problems::cpu::rosenbrock,
	//	lower_bounds,
	//	upper_bounds,
	//	1000
	//);
	abc_gpu_test();

	return 0;
}
