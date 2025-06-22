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
#include "timer.cuh"
#include "magic_enum.hpp"

#include <filesystem>
#include <cstdio>

//kje mora da gi izvadam kako vrednosti posle
using namespace abc_shared;

constexpr int max_title_size = 1000;

constexpr std::array<char, max_title_size> create_title_cpu(
	Selection   selection_type,
	Roulette    roulette_type,
	RouletteCpu roulette_sorting,
	Rank        rank_type,
	Tourn       tournament_type,
	uint32_t    tournament_size,
	uint32_t    tournament_num
)
{
	std::array<char, max_title_size> title = { };

	snprintf(
		title.data(),
		max_title_size,
		"output/cpu/%s_%s_%s_%s_%s(%d_%d).csv",
		SelectionTypeToString(selection_type),
		RouletteTypeToString(roulette_type),
		RouletteCpuToString(roulette_sorting),
		RankTypeToString(rank_type),
		TournamentTypeToString(tournament_type),
		tournament_size,
		tournament_num
	);

	return title;
}

constexpr std::array<char, max_title_size> create_title_gpu(
	Selection   selection_type   = ROULETTE_WHEEL,
	Roulette    roulette_type    = CUSTOM,
	Rank        rank_type        = LINEAR_ARRAY,
	Tourn       tournament_type  = SINGLE,
	uint32_t    tournament_size  = 10,
	uint32_t    tournament_num   = 2,
	uint32_t    grid_size        = 0,
	uint32_t    block_size       = 0
)
{
	std::array<char, max_title_size> title = { };

	snprintf(
		title.data(),
		max_title_size,
		"output/gpu/%s_%s_%s_%s(%d_%d)_%dx%d.csv",
		SelectionTypeToString(selection_type),
		RouletteTypeToString(roulette_type),
		RankTypeToString(rank_type),
		TournamentTypeToString(tournament_type),
		tournament_size,
		tournament_num,
		grid_size,
		block_size
	);

	return title;
}

template<
	uint32_t    dimensions,
	uint32_t    bees_count,
	uint32_t    scouts_count,
	uint32_t    trials_limit, 
	Selection   selection_type,
	Roulette    roulette_type,
	RouletteCpu roulette_sorting,
	Rank        rank_type,
	Tourn       tournament_type,
	uint32_t    tournament_size,
	uint32_t    tournament_num,
	//dodatno
	uint32_t    iterations,
	uint32_t    step_size,
	bool        debug
>
void abc_cpu_test(
	opt_func optimization_function,
	float    lower_bound[],
	float    upper_bound[]
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

	if (!std::filesystem::exists("output"))
		std::filesystem::create_directory("output");
	if (!std::filesystem::exists("output/cpu"))
		std::filesystem::create_directory("output/cpu");
	
	std::array<char, max_title_size> title = create_title_cpu(
		selection_type,
		roulette_type,
		roulette_sorting,
		rank_type,
		tournament_type,
		tournament_size,
		tournament_num
	);
	FILE *file = std::fopen(title.data(), "w");

	Timer timer = Timer();

	if(debug)
	{
		printf("Start...\n");
		printf("Printf debugging ABC\n");
	}

	std::fprintf(
		file,
		"duration;iterations;min_value;bees_countscouts_count\n"
	);


	int num_of_steps = iterations / step_size;
	for(int i = 0; i < num_of_steps; i++)
	{
		timer.start();
		cpu::abc<
			dimensions,
			bees_count,
			scouts_count,
			trials_limit,
			selection_type,
			roulette_type,
			roulette_sorting,
			rank_type,
			tournament_type,
			tournament_size,
			tournament_num
		>(
			&bees,
			step_size,
			optimization_function,
			lower_bound,
			upper_bound
		);
		uint64_t duration = timer.stop();

		cpu::Bee min_bee = cpu::min_bee(&bees, bees_count);

		std::fprintf(
			file,
			"%ld;%d;%f;%d;%d\n",
			duration,
			i*step_size,
			min_bee.value,
			bees_count,
			scouts_count
		);

		if (debug)
		{
			printf("Iteration #%03d Time %ld: ", i*step_size, duration);
			for(int j = 0; j <= dimensions; j++)
				printf("x%d: %.2f ", j, min_bee.coordinates[j]);

			printf("value: %.2f\n", min_bee.value);
		}
	}

	std::fclose(file);

	if (debug)
	{
		printf("Done...\n");

		printf("Printing top 10 final values\n");

		//ova debagiranje isto taka sakam da go smenam da e	pogenerichno
		std::vector<float> coordinates_x(bees_count);
		std::vector<float> coordinates_y(bees_count);
		//std::vector<double> values(num_of_bees);

		std::sort(bees.begin(), bees.end(), cpu::BeeCompare);
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
}

template<
	uint32_t    dimensions,
	uint32_t    trials_limit, 
	Selection   selection_type,
	Roulette    roulette_type,
	Rank        rank_type,
	Tourn       tournament_type,
	uint32_t    tournament_size,
	uint32_t    tournament_num,
	//gpu
	uint32_t    block_size,
	uint32_t    grid_size,
	//dodatno
	uint32_t    step_size,
	uint32_t    iterations,
	bool        debug
>
void abc_gpu_test(
	opt_func optimization_problem,
	float lower_bounds[dimensions],
	float upper_bounds[dimensions]
)
{
	//variable initialization
	int bees_count = grid_size*block_size;
  	int max_trials = 20;

	float* cords  = (float*) malloc(dimensions*bees_count*sizeof(float));
	float* values = (float*) malloc(           bees_count*sizeof(float));

	if (!std::filesystem::exists("output"))
		std::filesystem::create_directory("output");
	if (!std::filesystem::exists("output/gpu"))
		std::filesystem::create_directory("output/gpu");


	std::array<char, max_title_size> title = create_title_gpu(
		selection_type,
		roulette_type,
		rank_type,
		tournament_type,
		tournament_size,
		tournament_num,
		grid_size,
		block_size
	);
	FILE *file = std::fopen(title.data(), "w");

	std::fprintf(
		file,
		"duration;iterations;min_value;bees_count;\n"
	);

	int num_of_steps = iterations / step_size;
	for(int i = 0; i < num_of_steps; i++)
	{
		uint64_t duration;
		gpu::launch_abc
		<
			dimensions,
			grid_size,
			block_size,
			iterations,
			trials_limit,
			selection_type,
			roulette_type,
			rank_type,
			tournament_type,
			tournament_size,
			tournament_num
		>
		(
			cords,
			values,
			optimization_problem,
			lower_bounds,
			upper_bounds,
			&duration
		);

		int min_idx = 0;
		float min_value = values[0];
		for(int i = 1; i < bees_count; i++)
		{
			if(values[i] < min_value)
			{
				min_value = values[i];
				min_idx = i;
			}
		}

		std::fprintf(
			file,
			"%ld;%d;%f;%d\n",
			duration,
			i*step_size,
			min_value,
			bees_count
		);

		if (debug)
		{
			for(int i = 0; i < 10; i++)
			{
				float min = values[i];
				int min_idx;
				for(int j = i; j < bees_count; j++)
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

				for(int k = 0; k < dimensions; k++)
				{
					tmp = cords[i*dimensions + k];
					cords[i*dimensions + k] = cords[min_idx*dimensions + k];
					cords[min_idx*dimensions + k] = tmp;
				}

			}
			for(int i = 0; i < 10; i++)
			{
				printf(
					"Bee%03d: x=%f y=%f value=%f\n",
					i,
					cords[i*dimensions + 0],
					cords[i*dimensions + 1],
					values[i]
				);
			}
		}
	}
}


class RouletteWheelGpuTest
{
	static constexpr std::array<uint32_t, 1> dimensions_vals = {2};
	static constexpr std::array<uint32_t, 1> trials_vals = {10};
	//constexpr auto selection_vals = magic_enum::enum_values<Selection>();
	static constexpr std::array<Selection, 1> selection_vals = {ROULETTE_WHEEL};
	//constexpr auto roulette_vals = magic_enum::enum_values<Roulette>();
	static constexpr std::array<Roulette, 1> roulette_vals   = {CUSTOM};
	//constexpr auto rank_vals = magic_enum::enum_values<Rank>();
	static constexpr std::array<Rank, 1> rank_vals   = {LINEAR_ARRAY};
	//constexpr auto tourn_vals = magic_enum::enum_values<Tourn>();
	static constexpr std::array<Tourn, 1> tourn_vals  = {SINGLE};

	static constexpr std::array<uint32_t, 1> tourn_size_vals = {4};
	static constexpr std::array<uint32_t, 1> tourn_num_vals = {2};
	static constexpr std::array<uint32_t, 1> block_vals = {512};
	static constexpr std::array<uint32_t, 1> grid_vals = {100};
	static constexpr std::array<uint32_t, 1> step_vals = {100};
	static constexpr std::array<uint32_t, 1> iter_vals = {1000};
	static constexpr std::array<bool, 1> debug_vals = {true};


	template <typename F, std::size_t... I>
	constexpr void for_each_index(std::index_sequence<I...>, F&& f) {
		(f(std::integral_constant<std::size_t, I>{}), ...);
	}

public:
	template <typename Func>
	void for_all_combinations(Func&& func) {
		for_each_index(std::make_index_sequence<dimensions_vals.size()>(), [&](auto i_dim) {
		for_each_index(std::make_index_sequence<trials_vals.size()>(), [&](auto i_trials) {
		for_each_index(std::make_index_sequence<selection_vals.size()>(), [&](auto i_sel) {
		for_each_index(std::make_index_sequence<roulette_vals.size()>(), [&](auto i_roul) {
		for_each_index(std::make_index_sequence<rank_vals.size()>(), [&](auto i_rank) {
		for_each_index(std::make_index_sequence<tourn_vals.size()>(), [&](auto i_tourn) {
		for_each_index(std::make_index_sequence<tourn_size_vals.size()>(), [&](auto i_tsize) {
		for_each_index(std::make_index_sequence<tourn_num_vals.size()>(), [&](auto i_tnum) {
		for_each_index(std::make_index_sequence<block_vals.size()>(), [&](auto i_block) {
		for_each_index(std::make_index_sequence<grid_vals.size()>(), [&](auto i_grid) {
		for_each_index(std::make_index_sequence<step_vals.size()>(), [&](auto i_step) {
		for_each_index(std::make_index_sequence<iter_vals.size()>(), [&](auto i_iter) {
		for_each_index(std::make_index_sequence<debug_vals.size()>(), [&](auto i_debug) {

			constexpr uint32_t dim  = dimensions_vals[i_dim];
			constexpr uint32_t trial = trials_vals[i_trials];
			constexpr Selection sel = selection_vals[i_sel];
			constexpr Roulette roul = roulette_vals[i_roul];
			constexpr Rank rank = rank_vals[i_rank];
			constexpr Tourn tourn = tourn_vals[i_tourn];
			constexpr uint32_t tsize = tourn_size_vals[i_tsize];
			constexpr uint32_t tnum = tourn_num_vals[i_tnum];
			constexpr uint32_t block = block_vals[i_block];
			constexpr uint32_t grid = grid_vals[i_grid];
			constexpr uint32_t step = step_vals[i_step];
			constexpr uint32_t iter = iter_vals[i_iter];
			constexpr bool dbg = debug_vals[i_debug];

			// Declare dummy bounds for compile-time size
			float lower_bounds[dim] = {-10, -10};
			float upper_bounds[dim] = {+10, +10};
			opt_func f = problems::gpu::rosenbrock; // supply actual function

			abc_gpu_test<
				dim, trial, sel, roul, rank, tourn,
				tsize, tnum, block, grid, step, iter, dbg
			>(f, lower_bounds, upper_bounds);

		}); }); }); }); }); }); }); }); }); }); }); }); });
	}
};

int main()
{
	float lower_bounds[] = {-10, -10};
	float upper_bounds[] = { 10,  10};

	const int bees_count   = 1000;
	const int trials_limit =   20;
	const int dimensions   =    2;
	//abc_cpu_test<
	//	dimensions,
	//	bees_count,
	//	bees_count/10,
	//	trials_limit,
	//	//ROULETTE_WHEEL,
	//	RANK,
	//	CUSTOM,
	//	PARTIAL_SORT,
	//	//LINEAR_ARRAY,
	//	CONSTANT_LINEAR,
	//	SINGLE,
	//	bees_count/10,
	//	3,
	//	100,
	//	true
	//>(
	//	problems::cpu::rosenbrock,
	//	lower_bounds,
	//	upper_bounds,
	//	1000
	//);
	//abc_gpu_test<
	//	dimensions,
	//	trials_limit,
	//	ROULETTE_WHEEL,
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
	//for_all_combinations([]{});
	RouletteWheelGpuTest test = RouletteWheelGpuTest();
	test.for_all_combinations([]{});
	return 0;
}
