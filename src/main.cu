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
	for(int i = TestFunc::SCHWEFEL; i < TestFunc::INVALID; i++)
	{
		//RouletteWheelGpuTest test = RouletteWheelGpuTest<2, 100>();
		//RankGpuTest test = RankGpuTest<2, 100>();
		//RankCpuTest test = RankCpuTest<2, 10000>();
		//RouletteCpuTest test = RouletteCpuTest<2, 10000>();
		//TournamentCpuTestSingle test = TournamentCpuTestSingle<2, 10000>();
		//TournamentCpuTestMultiple test = TournamentCpuTestMultiple<2, 10000>();
		//TournamentGpuTestSingle test = TournamentGpuTestSingle<2, 10000>();
		TournamentGpuTestMultiple test = TournamentGpuTestMultiple<2, 10000>();
		test.for_all_combinations(
			(TestFunc) i
		);
	}

	return 0;
}
