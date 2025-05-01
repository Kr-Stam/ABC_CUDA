/******************************************************************************
 * @file main.cu                                                              *
 * @brief Testing and measuring execution time and accuracy of the sequential *
 *        and parallel implementation of the ABC (Artificial Bee Colony)      *
 *        algorithm                                                           *
 * @author Kristijan Stameski                                                 *
 ******************************************************************************/

#include <stdio.h>
#include <math.h>
#include "abc_cpu.h"
#include "abc_gpu.cuh"
#include "problems/problems.h"
#include "problems/cpu/problems_many_local_minima.h"
#include "problems/cpu/problems_valley_shaped.h"
#include "problems/gpu/problems_many_local_minima.cuh"
#include "utils/array.hpp"

//kje mora da gi izvadam kako vrednosti posle

template<
	unsigned int dimensions,
	unsigned int bees_count
>
void abc_cpu_test(
	opt_func optimization_function,
	double   lower_bound[],
	double   upper_bound[],
	int      iterations,
	int      max_trials,
	double   scout_ratio
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
		cpu::abc<dimensions, bees_count>(
			&bees,
			step,
			max_trials,
			scout_ratio,
			optimization_function,
			lower_bound,
			upper_bound
		);

		cpu::Bee min_bee = cpu::min_bee(&bees, bees_count);
		printf("Iteration #%03d: ", i*step);

		for(int j = 0; j <= dimensions; j++)
		{
			printf("x%d: %.2f ", j, min_bee.coordinates[j]);
		}
		printf("value: %.2f\n", min_bee.value);
	}

	printf("Done\n");

	//ova debagiranje isto taka sakam da go smenam da e
	//pogenerichno
	std::vector<double> coordinates_x(bees_count);
	std::vector<double> coordinates_y(bees_count);
	//std::vector<double> values(num_of_bees);
	//
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
	double upper_bound[] = {+10, +10};
	double lower_bound[] = {-10, -10};
	opt_func optimization_function = problems::gpu::cross_in_tray;

	//abc_cpu_test<2>(
	//	1000,
	//	optimization_function,
	//	lower_bound,
	//	upper_bound,
	//	1000,
	//	10,
	//	0.2
	//);

	//variable initialization
	int num_of_bees     = 1024;
	int max_generations = 10000;
  	int max_trials      =    20;

	double* coordinates = (double*) malloc(
		num_of_bees * sizeof(double) * DIMENSIONS
	);
	double* values = (double*) malloc(
		num_of_bees * sizeof(double)
	);
	gpu::launch_abc(
		coordinates,
		values,
		num_of_bees,
		max_generations,
		max_trials,
		optimization_function,
		lower_bound,
		upper_bound,
		10
	);

	for(int i = 0; i < num_of_bees; i++)
	{
		printf(
			"Bee%03d: x=%.2f y=%.2f value=%.2f\n",
			i,
			coordinates[i*DIMENSIONS + 0],
			coordinates[i*DIMENSIONS + 1],
			values[i]
		);
	}
}

int main()
{
	double lower_bounds[] = {-10, -10};
	double upper_bounds[] = { 10,  10};
	abc_cpu_test<2, 10000>(
		problems::cpu::rosenbrock,
		lower_bounds,
		upper_bounds,
		100,
		10,
		0.2
	);
	//abc_gpu_test();

	return 0;
}
