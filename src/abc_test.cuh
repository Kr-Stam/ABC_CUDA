/******************************************************************************
 * @file abc_test.cuh                                                         *
 * @brief Templated testing functions that make csv files with execution data *
 * @details Using templates all possible combinations of template parameters  *
 *          are instantiated and are then timed using chrono                  *
 *                                                                            *
 * @author Kristijan Stameski                                                 *
 *****************************************************************************/
#pragma once

#include <stdio.h>
#include <math.h>
#include <array>
#include <filesystem>
#include "abc_cpu.cuh"
#include "abc_gpu.cuh"
#include "problems/problems.h"
#include "problems/problems_gpu.cuh"
#include "timer.cuh"

using namespace abc_shared;
constexpr int MAX_TITLE_SIZE = 1000;

constexpr std::array<char, MAX_TITLE_SIZE> create_title_cpu(
	Selection   selection_type,
	Roulette    roulette_type,
	RouletteCpu roulette_sorting,
	Rank        rank_type,
	Tourn       tournament_type,
	uint32_t    tournament_size,
	uint32_t    tournament_num
)
{
	std::array<char, MAX_TITLE_SIZE> title = { };

	snprintf(
		title.data(),
		MAX_TITLE_SIZE,
		"output/cpu/%s_%s_%s_%s_%s(%d_%d).csv",
		SelectionTypeToString(selection_type).data(),
		RouletteTypeToString(roulette_type).data(),
		RouletteCpuToString(roulette_sorting).data(),
		RankTypeToString(rank_type).data(),
		TournamentTypeToString(tournament_type).data(),
		tournament_size,
		tournament_num
	);

	return title;
}

constexpr std::array<char, MAX_TITLE_SIZE> create_title_gpu(
	uint32_t  trials_limit,
	Selection selection_type   = ROULETTE_WHEEL,
	Roulette  roulette_type    = CUSTOM,
	Rank      rank_type        = LINEAR_ARRAY,
	Tourn     tournament_type  = SINGLE,
	uint32_t  tournament_size  = 10,
	uint32_t  tournament_num   = 2,
	uint32_t  grid_size        = 0,
	uint32_t  block_size       = 0
)
{
	std::array<char, MAX_TITLE_SIZE> title = { };

	switch(selection_type)
	{
	case ROULETTE_WHEEL:
		snprintf(
			title.data(),
			MAX_TITLE_SIZE,
			"%s_[%d]_(%dx%d).csv",
			RouletteTypeToString(roulette_type).data(),
			trials_limit,
			grid_size,
			block_size
		);
		break;
	case RANK:
		snprintf(
			title.data(),
			MAX_TITLE_SIZE,
			"%s_[%d]_(%dx%d).csv",
			RankTypeToString(rank_type).data(),
			trials_limit,
			grid_size,
			block_size
		);
		break;
	case TOURNAMENT:
		snprintf(
			title.data(),
			MAX_TITLE_SIZE,
			"%s_[%d]_(%dx%d)_(%dx%d).csv",
			TournamentTypeToString(tournament_type).data(),
			trials_limit,
			tournament_size,
			tournament_num,
			grid_size,
			block_size
		);
		break;
	}

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
	
	std::array<char, MAX_TITLE_SIZE> title = create_title_cpu(
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
		"duration;iterations;min_value;bees_count;scouts_count\n"
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

		if(debug)
		{
			printf("Iteration #%03d Time %ld: ", i*step_size, duration);
			for(int j = 0; j <= dimensions; j++)
				printf("x%d: %.2f ", j, min_bee.coordinates[j]);

			printf("value: %.2f\n", min_bee.value);
		}
	}

	std::fclose(file);

	if(debug)
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

void ensure_created(std::string dir)
{
	if (!std::filesystem::exists(dir))
		std::filesystem::create_directory(dir);
}

inline float calc_error(float global_min_value, float min_value)
{
	return abs(global_min_value - min_value);
}

template<unsigned int dim>
inline float calc_param_convergence(
	SolutionParams<dim> global_min_params,
	float min_value[dim]
)
{
	float result = INFINITY;
	for(int i = 0; i < global_min_params.num_of_solutions; i++)
	{
		float tmp_result = 0;
		for(int j = 0; j < dim; j++)
		{
			float tmp = global_min_params.solutions[i][j] - min_value[j];
			tmp_result += tmp*tmp;
		}
		if (tmp_result < result) result = tmp_result;
	}
	return result;
}

inline float calc_grad_norm(float min_value, float prev_min_value)
{
	float tmp = (prev_min_value - min_value); 
	return tmp*tmp;
}

template<
	uint32_t  dim,
	uint32_t  trials_limit, 
	Selection selection_type,
	Roulette  roulette_type,
	Rank      rank_type,
	Tourn     tournament_type,
	uint32_t  tournament_size,
	uint32_t  tournament_num,
	//gpu
	uint32_t  block_size,
	uint32_t  grid_size,
	uint32_t  step_size,
	uint32_t  iterations,
	bool      debug
>
void abc_gpu_test(
	TestFunc  test_function
)
{
	std::string out_path;
	{ // create out_path
		out_path = "output";
		ensure_created(out_path);
		out_path.append("/gpu");
		ensure_created(out_path);
		std::string func_type_str = TestFuncToString(test_function).data();
		out_path.append("/");
		out_path.append(func_type_str);
		ensure_created(out_path);
		std::string sel_type_str = SelectionTypeToString(selection_type).data();
		out_path.append("/");
		out_path.append(sel_type_str);
		ensure_created(out_path);
		out_path.append("/");
		std::array<char, MAX_TITLE_SIZE> title = create_title_gpu(
			trials_limit,
			selection_type,
			roulette_type,
			rank_type,
			tournament_type,
			tournament_size,
			tournament_num,
			grid_size,
			block_size
		);
		out_path.append(title.data());
	}

	FILE* file = std::fopen(out_path.data(), "w");
	std::fprintf(
		file,
		"duration;iterations;min_value;error;param_convergence;grad_norm\n"
	);
	
	//variable initialization
	int bees_count = grid_size*block_size;

	float lower_bounds[dim] = { };
	float upper_bounds[dim] = { };
	GetBounds<dim >(test_function, lower_bounds, upper_bounds);

	float* cords  = (float*) malloc(dim*bees_count*sizeof(float));
	float* values = (float*) malloc(           bees_count*sizeof(float));

	int num_of_steps = iterations / step_size;
	float prev_min_value;

	float global_min_value = GetGlobalMinValue(test_function);
	SolutionParams global_min_params = GetGlobalMinParams<dim>(test_function);

	for(int i = 0; i < num_of_steps; i++)
	{
		uint64_t duration;
		gpu::launch_abc
		<
			dim,
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
			test_function,
			lower_bounds,
			upper_bounds,
			&duration
		);


		//Find minimum
		int min_idx = 0;
		float min_value = values[0];
		for(int j = 1; j < bees_count; j++)
		{
			if(min_value > values[j])
			{
				min_value = values[j];
				min_idx = j;
			}
		}

		float error = calc_error(global_min_value, min_value);
		float param_convergence = calc_param_convergence<dim>(
			global_min_params,
			&cords[min_idx]
		);
		float grad_norm = calc_grad_norm(min_value, prev_min_value); 
		std::fprintf(
			file,
			"%ld;%d;%f;%f;%f;%f\n",
			duration,
			i*step_size,
			min_value,
			error,
			param_convergence,
			grad_norm
		);
		prev_min_value = min_value;

		{ // write to stdout
			if(debug)
			{
				for(int j = 0; j < 10; j++)
				{
					float min = values[j];
					int min_idx = j;
					for(int k = j + 1; k < bees_count; k++)
					{
						if(min > values[k])
						{
							min_idx = k;
							min = values[k];
						}
					}

					for(int k = 0; k < dim; k++)
					{
						float tmp = cords[min_idx*dim + k];
						cords[min_idx*dim + k] = cords[j*dim + k];
						cords[j*dim + k] = tmp;
					}
					values[min_idx] = values[j];
					values[j] = min;
				}
				for(int j = 0; j < 10; j++)
				{
					printf(
						"Bee%03d: x=%f y=%f value=%f\n",
						j,
						cords[j*dim + 0],
						cords[j*dim + 1],
						values[j]
					);
				}
			}
		}

		#define EPS 1.0E-8
		if (error < EPS && param_convergence < EPS && grad_norm < EPS) break;
	}

	free(cords);
	free(values);
}

class GpuTestBase
{
public:
	static constexpr std::array<uint32_t, 3> trials_vals = {
		10, 20, 30
	};
	static constexpr std::array<uint32_t, 1> step_vals   = {
		10 
	};

	static constexpr std::array<uint32_t, 5> block_vals = {
		32, 64, 128, 256, 512 
	};
	static constexpr std::array<uint32_t, 3> grid_vals = { 5, 10, 100 };
};

template<
	uint32_t dim,
	uint32_t max_iterations
>
class RouletteWheelGpuTest : private GpuTestBase
{
	static constexpr std::array<Roulette, 3> roulette_vals = {SUM, CUSTOM, MIN_MAX};

	template <typename F, std::size_t... I>
	constexpr void for_each_index(std::index_sequence<I...>, F&& f) {
		(f(std::integral_constant<std::size_t, I>{}), ...);
	}

public:
	void for_all_combinations(
			TestFunc test_function,
			float lower_bounds[dim], float upper_bounds[dim]
	) {
		for_each_index(std::make_index_sequence<trials_vals.size()>(), [&](auto i_trials) {
		for_each_index(std::make_index_sequence<roulette_vals.size()>(), [&](auto i_roul) {
		for_each_index(std::make_index_sequence<block_vals.size()>(), [&](auto i_block) {
		for_each_index(std::make_index_sequence<grid_vals.size()>(), [&](auto i_grid) {
		for_each_index(std::make_index_sequence<step_vals.size()>(), [&](auto i_step) {
			constexpr uint32_t trial = trials_vals[i_trials];
			constexpr Roulette roul = roulette_vals[i_roul];
			constexpr uint32_t block = block_vals[i_block];
			constexpr uint32_t grid = grid_vals[i_grid];
			constexpr uint32_t step = step_vals[i_step];

			abc_gpu_test<
				dim, trial, ROULETTE_WHEEL, roul, LINEAR_ARRAY, SINGLE,
				0, 0, block, grid, step, max_iterations, true
			>(test_function);

		}); }); }); }); });
	}
};

template<
	uint32_t dim,
	uint32_t max_iterations,
	TestFunc test_function
>
class RankGpuTest : private GpuTestBase
{
	static constexpr std::array<Rank, 7> rank_vals = {
		LINEAR_ARRAY,
		EXPONENTIAL_ARRAY,
		LINEAR_SIMPLE_ARRAY,
		EXPONENTIAL_SIMPLE_ARRAY,  
		CONSTANT_LINEAR,
		CONSTANT_EXPONENTIAL,
		CONSTANT_EXPONENTIAL_2
	};

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
			constexpr uint32_t trial = trials_vals[i_trials];
			constexpr Rank rank = rank_vals[i_rank];
			constexpr uint32_t block = block_vals[i_block];
			constexpr uint32_t grid = grid_vals[i_grid];
			constexpr uint32_t step = step_vals[i_step];

			abc_gpu_test<
				dim, trial, RANK, SUM, rank, SINGLE,
				0, 0, block, grid, step, max_iterations, true
			>(test_function);

		}); }); }); }); });
	}
};

template<
	uint32_t dim,
	uint32_t max_iterations,
	TestFunc test_function
>
class TournamentGpuTestSingle : private GpuTestBase
{
	static constexpr std::array<uint32_t, 4> tournament_sizes = {4, 8, 16, 32};

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

			constexpr uint32_t tournament_size = tournament_sizes[i_tourn_size];

			constexpr uint32_t trial = trials_vals[i_trials];
			constexpr uint32_t block = block_vals[i_block];
			constexpr uint32_t grid  = grid_vals[i_grid];
			constexpr uint32_t step  = step_vals[i_step];

			abc_gpu_test<
				dim, trial, TOURNAMENT, SUM, LINEAR_ARRAY, SINGLE,
				tournament_size, 0, block, grid, step, max_iterations,
				true
			>(test_function);

		}); }); }); }); });
	}
};

template<
	uint32_t dim,
	uint32_t max_iterations,
	TestFunc test_function
>
class TournamentGpuTestMultiple : private GpuTestBase
{
	static constexpr std::array<uint32_t, 4> tournament_sizes = {4, 8, 16, 32};
	static constexpr std::array<uint32_t, 4> tournament_nums  = {2, 4, 8, 16};

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

			constexpr uint32_t tournament_size = tournament_sizes[i_tourn_size];
			constexpr uint32_t tournament_num  = tournament_nums[i_tourn_num];

			constexpr uint32_t trial = trials_vals[i_trials];
			constexpr uint32_t block = block_vals[i_block];
			constexpr uint32_t grid  = grid_vals[i_grid];
			constexpr uint32_t step  = step_vals[i_step];

			abc_gpu_test<
				dim, trial, TOURNAMENT, SUM, LINEAR_ARRAY, MULTIPLE,
				tournament_size, tournament_num, block, grid, step, 
				max_iterations, test_function, true
			>(test_function);

		}); }); }); }); }); });
	}
};

class CpuTestBase
{
public:
	static constexpr std::array<uint32_t, 1> trials_vals = {10};
	static constexpr std::array<uint32_t, 1> bees_counts = {1000};
	static constexpr std::array<uint32_t, 1> step_vals = {100};
};

template<uint32_t dim, uint32_t max_iterations>
class RouletteCpuTest : private CpuTestBase
{
	static constexpr std::array<Roulette, 3> roul_vals = {SUM, CUSTOM, MIN_MAX};
	static constexpr std::array<RouletteCpu, 2> roul_sorting_vals = {FULL, PARTIAL_SORT};


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
			constexpr uint32_t trial = trials_vals[i_trials];
			constexpr Roulette roul = roul_vals[i_roul];
			constexpr RouletteCpu roul_sort = roul_sorting_vals[i_roul_sort];
			constexpr uint32_t bees_count = bees_counts[i_bees_count];
			constexpr uint32_t step = step_vals[i_step];

			abc_cpu_test<
				dim, bees_count, bees_count/10, trial, ROULETTE_WHEEL, roul,
				roul_sort, LINEAR_ARRAY, SINGLE, 0, 0, max_iterations, step, true
			>(function, lower_bounds, upper_bounds);

		}); }); }); }); });
	}
};

template<uint32_t dim, uint32_t max_iterations>
class RankCpuTest : private CpuTestBase
{
	static constexpr std::array<Rank, 7> rank_vals = {
		LINEAR_ARRAY,
		EXPONENTIAL_ARRAY,
		LINEAR_SIMPLE_ARRAY,
		EXPONENTIAL_SIMPLE_ARRAY,  
		CONSTANT_LINEAR,
		CONSTANT_EXPONENTIAL,
		CONSTANT_EXPONENTIAL_2
	};

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
			constexpr uint32_t trial = trials_vals[i_trials];
			constexpr Rank rank = rank_vals[i_rank];
			constexpr uint32_t bees_count = bees_counts[i_bees_count];
			constexpr uint32_t step = step_vals[i_step];

			abc_cpu_test<
				dim, bees_count, bees_count/10, trial, RANK, SUM, FULL, rank,
				SINGLE,	0, 0, max_iterations, step, true
			>(function, lower_bounds, upper_bounds);

		}); }); }); });
	}
};

template<uint32_t dim, uint32_t max_iterations>
class TournamentCpuTestSingle : private CpuTestBase
{
	static constexpr std::array<uint32_t, 3> tournament_sizes = {4, 8, 16};

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
			constexpr uint32_t trial = trials_vals[i_trials];
			constexpr uint32_t tournament_size = tournament_sizes[i_tourn_sizes];
			constexpr uint32_t bees_count = bees_counts[i_bees_count];
			constexpr uint32_t step = step_vals[i_step];

			abc_cpu_test<
				dim, bees_count, bees_count/10, trial, TOURNAMENT, SUM, FULL, LINEAR_ARRAY, SINGLE,
				tournament_size, 0, max_iterations, step, true
			>(function, lower_bounds, upper_bounds);

		}); }); }); });
	}
};

template<uint32_t dim, uint32_t max_iterations>
class TournamentCpuTestMultiple : private CpuTestBase
{
	static constexpr std::array<uint32_t, 3> tournament_sizes = {4, 8, 16};
	static constexpr std::array<uint32_t, 3> tournament_nums  = {2, 4, 8};

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
			constexpr uint32_t trial = trials_vals[i_trials];
			constexpr uint32_t tournament_size = tournament_sizes[i_tourn_sizes];
			constexpr uint32_t tournament_num = tournament_nums[i_tourn_nums];
			constexpr uint32_t bees_count = bees_counts[i_bees_count];
			constexpr uint32_t step = step_vals[i_step];

			abc_cpu_test<
				dim, bees_count, bees_count/10, trial, TOURNAMENT, SUM, FULL,
				LINEAR_ARRAY, MULTIPLE, tournament_size, tournament_num, max_iterations,
				step, true
			>(function, lower_bounds, upper_bounds);

		}); }); }); }); });
	}
};
