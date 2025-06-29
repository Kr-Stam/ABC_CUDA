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
	float    lower_bound[dimensions],
	float    upper_bound[dimensions]
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
		for(int j = 1; j < bees_count; j++)
		{
			if(values[j] < min_value)
			{
				min_value = values[j];
				min_idx = j;
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
			for(int j = 0; j < 10; j++)
			{
				float min = values[j];
				int min_idx;
				for(int k = j; k < bees_count; k++)
				{
					if(min > values[k])
					{
						min_idx = k;
						min = values[k];
					}
				}
				float tmp = values[j];
				values[j] = values[min_idx];
				values[min_idx] = tmp;

				for(int k = 0; k < dimensions; k++)
				{
					tmp = cords[j*dimensions + k];
					cords[j*dimensions + k] = cords[min_idx*dimensions + k];
					cords[min_idx*dimensions + k] = tmp;
				}

			}
			for(int j = 0; j < 10; j++)
			{
				printf(
					"Bee%03d: x=%f y=%f value=%f\n",
					j,
					cords[j*dimensions + 0],
					cords[j*dimensions + 1],
					values[j]
				);
			}
		}
	}
}

template<uint32_t dim>
class RouletteWheelGpuTest
{
	static constexpr std::array<uint32_t, 1> trials_vals = {10};
	static constexpr std::array<Roulette, 3> roulette_vals = {SUM, CUSTOM, MIN_MAX};

	static constexpr std::array<uint32_t, 1> block_vals = {512};
	static constexpr std::array<uint32_t, 1> grid_vals = {100};
	static constexpr std::array<uint32_t, 1> step_vals = {100};
	static constexpr std::array<uint32_t, 1> iter_vals = {1000};

	template <typename F, std::size_t... I>
	constexpr void for_each_index(std::index_sequence<I...>, F&& f) {
		(f(std::integral_constant<std::size_t, I>{}), ...);
	}

public:
	void for_all_combinations(
			opt_func function,
			float lower_bounds[dim], float upper_bounds[dim]
	) {
		for_each_index(std::make_index_sequence<trials_vals.size()>(), [&](auto i_trials) {
		for_each_index(std::make_index_sequence<roulette_vals.size()>(), [&](auto i_roul) {
		for_each_index(std::make_index_sequence<block_vals.size()>(), [&](auto i_block) {
		for_each_index(std::make_index_sequence<grid_vals.size()>(), [&](auto i_grid) {
		for_each_index(std::make_index_sequence<step_vals.size()>(), [&](auto i_step) {
		for_each_index(std::make_index_sequence<iter_vals.size()>(), [&](auto i_iter) {
			constexpr uint32_t trial = trials_vals[i_trials];
			constexpr Roulette roul = roulette_vals[i_roul];
			constexpr uint32_t block = block_vals[i_block];
			constexpr uint32_t grid = grid_vals[i_grid];
			constexpr uint32_t step = step_vals[i_step];
			constexpr uint32_t iter = iter_vals[i_iter];

			abc_gpu_test<
				dim, trial, ROULETTE_WHEEL, roul, LINEAR_ARRAY, SINGLE,
				0, 0, block, grid, step, iter, true
			>(function, lower_bounds, upper_bounds);

		}); }); }); }); }); });
	}
};

template<uint32_t dim>
class RankGpuTest
{
	static constexpr std::array<uint32_t, 1> trials_vals = {10};
	static constexpr std::array<Rank, 7> rank_vals = {
		LINEAR_ARRAY,
		EXPONENTIAL_ARRAY,
		LINEAR_SIMPLE_ARRAY,
		EXPONENTIAL_SIMPLE_ARRAY,  
		CONSTANT_LINEAR,
		CONSTANT_EXPONENTIAL,
		CONSTANT_EXPONENTIAL_2
	};

	static constexpr std::array<uint32_t, 1> block_vals = {512};
	static constexpr std::array<uint32_t, 1> grid_vals = {100};
	static constexpr std::array<uint32_t, 1> step_vals = {100};
	static constexpr std::array<uint32_t, 1> iter_vals = {1000};

	template <typename F, std::size_t... I>
	constexpr void for_each_index(std::index_sequence<I...>, F&& f) {
		(f(std::integral_constant<std::size_t, I>{}), ...);
	}

public:
	void for_all_combinations(
			opt_func function,
			float lower_bounds[dim], float upper_bounds[dim]
	) {
		for_each_index(std::make_index_sequence<trials_vals.size()>(), [&](auto i_trials) {
		for_each_index(std::make_index_sequence<rank_vals.size()>(), [&](auto i_rank) {
		for_each_index(std::make_index_sequence<block_vals.size()>(), [&](auto i_block) {
		for_each_index(std::make_index_sequence<grid_vals.size()>(), [&](auto i_grid) {
		for_each_index(std::make_index_sequence<step_vals.size()>(), [&](auto i_step) {
		for_each_index(std::make_index_sequence<iter_vals.size()>(), [&](auto i_iter) {
			constexpr uint32_t trial = trials_vals[i_trials];
			constexpr Rank rank = rank_vals[i_rank];
			constexpr uint32_t block = block_vals[i_block];
			constexpr uint32_t grid = grid_vals[i_grid];
			constexpr uint32_t step = step_vals[i_step];
			constexpr uint32_t iter = iter_vals[i_iter];

			abc_gpu_test<
				dim, trial, RANK, SUM, rank, SINGLE,
				0, 0, block, grid, step, iter, true
			>(function, lower_bounds, upper_bounds);

		}); }); }); }); }); });
	}
};

template<uint32_t dim>
class TournamentGpuTestSingle
{
	static constexpr std::array<uint32_t, 4> tournament_sizes = {4, 8, 16, 32};

	static constexpr std::array<uint32_t, 1> trials_vals = {10};
	static constexpr std::array<uint32_t, 1> block_vals = {512};
	static constexpr std::array<uint32_t, 1> grid_vals = {100};
	static constexpr std::array<uint32_t, 1> step_vals = {100};
	static constexpr std::array<uint32_t, 1> iter_vals = {1000};

	template <typename F, std::size_t... I>
	constexpr void for_each_index(std::index_sequence<I...>, F&& f) {
		(f(std::integral_constant<std::size_t, I>{}), ...);
	}

public:
	void for_all_combinations(
			opt_func function,
			float lower_bounds[dim], float upper_bounds[dim]
	) {
		for_each_index(std::make_index_sequence<tournament_sizes.size()>(), [&](auto i_tourn_size) {
		for_each_index(std::make_index_sequence<trials_vals.size()>(), [&](auto i_trials) {
		for_each_index(std::make_index_sequence<block_vals.size()>(), [&](auto i_block) {
		for_each_index(std::make_index_sequence<grid_vals.size()>(), [&](auto i_grid) {
		for_each_index(std::make_index_sequence<step_vals.size()>(), [&](auto i_step) {
		for_each_index(std::make_index_sequence<iter_vals.size()>(), [&](auto i_iter) {

			constexpr uint32_t tournament_size = tournament_sizes[i_tourn_size];

			constexpr uint32_t trial = trials_vals[i_trials];
			constexpr uint32_t block = block_vals[i_block];
			constexpr uint32_t grid  = grid_vals[i_grid];
			constexpr uint32_t step  = step_vals[i_step];
			constexpr uint32_t iter  = iter_vals[i_iter];

			abc_gpu_test<
				dim, trial, TOURNAMENT, SUM, LINEAR_ARRAY, SINGLE,
				tournament_size, 0, block, grid, step, iter, true
			>(function, lower_bounds, upper_bounds);

		}); }); }); }); }); });
	}
};

template<uint32_t dim>
class TournamentGpuTestMultiple
{
	static constexpr std::array<uint32_t, 4> tournament_sizes = {4, 8, 16, 32};
	static constexpr std::array<uint32_t, 4> tournament_nums  = {2, 4, 8, 16};

	static constexpr std::array<uint32_t, 1> trials_vals = {10};
	static constexpr std::array<uint32_t, 1> block_vals = {512};
	static constexpr std::array<uint32_t, 1> grid_vals = {100};
	static constexpr std::array<uint32_t, 1> step_vals = {100};
	static constexpr std::array<uint32_t, 1> iter_vals = {1000};

	template <typename F, std::size_t... I>
	constexpr void for_each_index(std::index_sequence<I...>, F&& f) {
		(f(std::integral_constant<std::size_t, I>{}), ...);
	}

public:
	void for_all_combinations(
			opt_func function,
			float lower_bounds[dim], float upper_bounds[dim]
	) {
		for_each_index(std::make_index_sequence<tournament_sizes.size()>(), [&](auto i_tourn_size) {
		for_each_index(std::make_index_sequence<tournament_nums.size()>(), [&](auto i_tourn_num) {
		for_each_index(std::make_index_sequence<trials_vals.size()>(), [&](auto i_trials) {
		for_each_index(std::make_index_sequence<block_vals.size()>(), [&](auto i_block) {
		for_each_index(std::make_index_sequence<grid_vals.size()>(), [&](auto i_grid) {
		for_each_index(std::make_index_sequence<step_vals.size()>(), [&](auto i_step) {
		for_each_index(std::make_index_sequence<iter_vals.size()>(), [&](auto i_iter) {

			constexpr uint32_t tournament_size = tournament_sizes[i_tourn_size];
			constexpr uint32_t tournament_num  = tournament_nums[i_tourn_num];

			constexpr uint32_t trial = trials_vals[i_trials];
			constexpr uint32_t block = block_vals[i_block];
			constexpr uint32_t grid  = grid_vals[i_grid];
			constexpr uint32_t step  = step_vals[i_step];
			constexpr uint32_t iter  = iter_vals[i_iter];

			abc_gpu_test<
				dim, trial, TOURNAMENT, SUM, LINEAR_ARRAY, MULTIPLE,
				tournament_size, tournament_num, block, grid, step, iter, true
			>(function, lower_bounds, upper_bounds);

		}); }); }); }); }); }); }); 
	}
};

template<uint32_t dim>
class RouletteCpuTest
{
	static constexpr std::array<uint32_t, 1> trials_vals = {10};
	static constexpr std::array<Roulette, 3> roul_vals = {SUM, CUSTOM, MIN_MAX};
	static constexpr std::array<RouletteCpu, 2> roul_sorting_vals = {FULL, PARTIAL_SORT};

	static constexpr std::array<uint32_t, 1> bees_counts = {1000};
	static constexpr std::array<uint32_t, 1> step_vals = {100};
	static constexpr std::array<uint32_t, 1> iter_vals = {1000};

	template <typename F, std::size_t... I>
	constexpr void for_each_index(std::index_sequence<I...>, F&& f) {
		(f(std::integral_constant<std::size_t, I>{}), ...);
	}

public:
	void for_all_combinations(
			opt_func function,
			float lower_bounds[dim], float upper_bounds[dim]
	) {
		for_each_index(std::make_index_sequence<trials_vals.size()>(), [&](auto i_trials) {
		for_each_index(std::make_index_sequence<roul_vals.size()>(), [&](auto i_roul) {
		for_each_index(std::make_index_sequence<roul_sorting_vals.size()>(), [&](auto i_roul_sort) {
		for_each_index(std::make_index_sequence<bees_counts.size()>(), [&](auto i_bees_count) {
		for_each_index(std::make_index_sequence<step_vals.size()>(), [&](auto i_step) {
		for_each_index(std::make_index_sequence<iter_vals.size()>(), [&](auto i_iter) {
			constexpr uint32_t trial = trials_vals[i_trials];
			constexpr Roulette roul = roul_vals[i_roul];
			constexpr RouletteCpu roul_sort = roul_sorting_vals[i_roul_sort];
			constexpr uint32_t bees_count = bees_counts[i_bees_count];
			constexpr uint32_t step = step_vals[i_step];
			constexpr uint32_t iter = iter_vals[i_iter];

			abc_cpu_test<
				dim, bees_count, bees_count/10, trial, ROULETTE_WHEEL, roul, roul_sort, LINEAR_ARRAY, SINGLE,
				0, 0, iter, step, true
			>(function, lower_bounds, upper_bounds);

		}); }); }); }); }); });
	}
};

template<uint32_t dim>
class RankCpuTest
{
	static constexpr std::array<uint32_t, 1> trials_vals = {10};
	static constexpr std::array<Rank, 7> rank_vals = {
		LINEAR_ARRAY,
		EXPONENTIAL_ARRAY,
		LINEAR_SIMPLE_ARRAY,
		EXPONENTIAL_SIMPLE_ARRAY,  
		CONSTANT_LINEAR,
		CONSTANT_EXPONENTIAL,
		CONSTANT_EXPONENTIAL_2
	};

	static constexpr std::array<uint32_t, 1> bees_counts = {1000};
	static constexpr std::array<uint32_t, 1> step_vals = {100};
	static constexpr std::array<uint32_t, 1> iter_vals = {1000};

	template <typename F, std::size_t... I>
	constexpr void for_each_index(std::index_sequence<I...>, F&& f) {
		(f(std::integral_constant<std::size_t, I>{}), ...);
	}

public:
	void for_all_combinations(
			opt_func function,
			float lower_bounds[dim], float upper_bounds[dim]
	) {
		for_each_index(std::make_index_sequence<trials_vals.size()>(), [&](auto i_trials) {
		for_each_index(std::make_index_sequence<rank_vals.size()>(), [&](auto i_rank) {
		for_each_index(std::make_index_sequence<bees_counts.size()>(), [&](auto i_bees_count) {
		for_each_index(std::make_index_sequence<step_vals.size()>(), [&](auto i_step) {
		for_each_index(std::make_index_sequence<iter_vals.size()>(), [&](auto i_iter) {
			constexpr uint32_t trial = trials_vals[i_trials];
			constexpr Rank rank = rank_vals[i_rank];
			constexpr uint32_t bees_count = bees_counts[i_bees_count];
			constexpr uint32_t step = step_vals[i_step];
			constexpr uint32_t iter = iter_vals[i_iter];

			abc_cpu_test<
				dim, bees_count, bees_count/10, trial, RANK, SUM, FULL, rank, SINGLE,
				0, 0, iter, step, true
			>(function, lower_bounds, upper_bounds);

		}); }); }); }); });
	}
};

template<uint32_t dim>
class TournamentCpuTestSingle
{
	static constexpr std::array<uint32_t, 3> tournament_sizes = {4, 8, 16};

	static constexpr std::array<uint32_t, 1> trials_vals = {10};
	static constexpr std::array<uint32_t, 1> bees_counts = {1000};
	static constexpr std::array<uint32_t, 1> step_vals = {100};
	static constexpr std::array<uint32_t, 1> iter_vals = {1000};

	template <typename F, std::size_t... I>
	constexpr void for_each_index(std::index_sequence<I...>, F&& f) {
		(f(std::integral_constant<std::size_t, I>{}), ...);
	}

public:
	void for_all_combinations(
			opt_func function,
			float lower_bounds[dim], float upper_bounds[dim]
	) {
		for_each_index(std::make_index_sequence<trials_vals.size()>(), [&](auto i_trials) {
		for_each_index(std::make_index_sequence<tournament_sizes.size()>(), [&](auto i_tourn_sizes) {
		for_each_index(std::make_index_sequence<bees_counts.size()>(), [&](auto i_bees_count) {
		for_each_index(std::make_index_sequence<step_vals.size()>(), [&](auto i_step) {
		for_each_index(std::make_index_sequence<iter_vals.size()>(), [&](auto i_iter) {
			constexpr uint32_t trial = trials_vals[i_trials];
			constexpr uint32_t tournament_size = tournament_sizes[i_tourn_sizes];
			constexpr uint32_t bees_count = bees_counts[i_bees_count];
			constexpr uint32_t step = step_vals[i_step];
			constexpr uint32_t iter = iter_vals[i_iter];

			abc_cpu_test<
				dim, bees_count, bees_count/10, trial, TOURNAMENT, SUM, FULL, LINEAR_ARRAY, SINGLE,
				tournament_size, 0, iter, step, true
			>(function, lower_bounds, upper_bounds);

		}); }); }); }); });
	}
};

template<uint32_t dim>
class TournamentCpuTestMultiple
{
	static constexpr std::array<uint32_t, 3> tournament_sizes = {4, 8, 16};
	static constexpr std::array<uint32_t, 3> tournament_nums  = {2, 4, 8};

	static constexpr std::array<uint32_t, 1> trials_vals = {10};
	static constexpr std::array<uint32_t, 1> bees_counts = {1000};
	static constexpr std::array<uint32_t, 1> step_vals = {100};
	static constexpr std::array<uint32_t, 1> iter_vals = {1000};

	template <typename F, std::size_t... I>
	constexpr void for_each_index(std::index_sequence<I...>, F&& f) {
		(f(std::integral_constant<std::size_t, I>{}), ...);
	}

public:
	void for_all_combinations(
			opt_func function,
			float lower_bounds[dim], float upper_bounds[dim]
	) {
		for_each_index(std::make_index_sequence<trials_vals.size()>(), [&](auto i_trials) {
		for_each_index(std::make_index_sequence<tournament_sizes.size()>(), [&](auto i_tourn_sizes) {
		for_each_index(std::make_index_sequence<tournament_nums.size()>(), [&](auto i_tourn_nums) {
		for_each_index(std::make_index_sequence<bees_counts.size()>(), [&](auto i_bees_count) {
		for_each_index(std::make_index_sequence<step_vals.size()>(), [&](auto i_step) {
		for_each_index(std::make_index_sequence<iter_vals.size()>(), [&](auto i_iter) {
			constexpr uint32_t trial = trials_vals[i_trials];
			constexpr uint32_t tournament_size = tournament_sizes[i_tourn_sizes];
			constexpr uint32_t tournament_num = tournament_nums[i_tourn_nums];
			constexpr uint32_t bees_count = bees_counts[i_bees_count];
			constexpr uint32_t step = step_vals[i_step];
			constexpr uint32_t iter = iter_vals[i_iter];

			abc_cpu_test<
				dim, bees_count, bees_count/10, trial, TOURNAMENT, SUM, FULL, LINEAR_ARRAY, MULTIPLE,
				tournament_size, tournament_num, iter, step, true
			>(function, lower_bounds, upper_bounds);

		}); }); }); }); }); });
	}
};

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
	abc_gpu_test<
		2,
		20,
		RANK,
		CUSTOM,
		CONSTANT_LINEAR,
		SINGLE,
		10,
		3,
		512,
		100,
		100,
		1000,
		true
	>(
		problems::gpu::rosenbrock,
		lower_bounds,
		upper_bounds
	); 
	//RouletteWheelGpuTest roulette_test = RouletteWheelGpuTest<2>();
	//roulette_test.for_all_combinations(problems::gpu::rosenbrock, lower_bounds, upper_bounds);
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
