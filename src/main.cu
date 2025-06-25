/******************************************************************************
 * @file main.cu                                                              *
 * @brief Testing and measuring execution time and accuracy of the sequential *
 *        and parallel implementation of the ABC (Artificial Bee Colony)      *
 *        algorithm                                                           *
 * @author Kristijan Stameski                                                 *
 ******************************************************************************/

#include "abc_test.cuh"
#include <stdio.h>
#include <math.h>
#include <array>
#include "abc_cpu.cuh"
#include "abc_gpu.cuh"
#include "problems/problems.h"
#include "problems/gpu/many_local_minima.cuh"
#include "problems/gpu/valley_shaped.cuh"
#include "utils/array.hpp"
#include "timer.cuh"

#include <filesystem>
#include <cstdio>

//kje mora da gi izvadam kako vrednosti posle
using namespace abc_shared;

int main()
{
	float lower_bounds[] = {-10, -10};
	float upper_bounds[] = { 10,  10};

	//abc_cpu_test<
	//	2,
	//	1000,
	//	1000/10,
	//	20,
	//	//ROULETTE_WHEEL,
	//	RANK,
	//	CUSTOM,
	//	PARTIAL_SORT,
	//	//LINEAR_ARRAY,
	//	CONSTANT_LINEAR,
	//	SINGLE,
	//	1000/10,
	//	3,
	//	1000,
	//	100,
	//	true
	//>(
	//	problems::cpu::rosenbrock,
	//	lower_bounds,
	//	upper_bounds
	//);
	//abc_gpu_test<
	//	2,
	//	20,
	//	RANK,
	//	CUSTOM,
	//	CONSTANT_LINEAR,
	//	SINGLE,
	//	10,
	//	3,
	//	512,
	//	100,
	//	100,
	//	1000,
	//	true
	//>(
	//	problems::gpu::rosenbrock,
	//	lower_bounds,
	//	upper_bounds
	//); 
	RouletteWheelGpuTest roulette_test = RouletteWheelGpuTest<2, 1000>();
	roulette_test.for_all_combinations(
		problems::gpu::rosenbrock,
		lower_bounds,
		upper_bounds
	);
	//RankGpuTest rank_test = RankGpuTest<2>();
	//rank_test.for_all_combinations(problems::gpu::rosenbrock, lower_bounds, upper_bounds);
	//TournamentGpuTestSingle tourn_single_test = TournamentGpuTestSingle<2>();
	//tourn_single_test.for_all_combinations(problems::gpu::rosenbrock, lower_bounds, upper_bounds);
	//TournamentGpuTestMultiple tourn_multiple_test = TournamentGpuTestMultiple<2>();
	//tourn_multiple_test.for_all_combinations(problems::gpu::rosenbrock, lower_bounds, upper_bounds);
	//RouletteCpuTest roulette_test = RouletteCpuTest<2>();
	//roulette_test.for_all_combinations(problems::cpu::rosenbrock, lower_bounds, upper_bounds);
	//RankCpuTest rank_test = RankCpuTest<2>();
	//rank_test.for_all_combinations(problems::cpu::rosenbrock, lower_bounds, upper_bounds);
	//TournamentCpuTestSingle tourn_single_test = TournamentCpuTestSingle<2>();
	//tourn_single_test.for_all_combinations(problems::gpu::rosenbrock, lower_bounds, upper_bounds);
//	TournamentCpuTestMultiple tourn_multiple_test = TournamentCpuTestMultiple<2>();
//	tourn_multiple_test.for_all_combinations(problems::gpu::rosenbrock, lower_bounds, upper_bounds);

	return 0;
}
