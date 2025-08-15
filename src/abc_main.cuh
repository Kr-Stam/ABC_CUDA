#pragma once

#include <array>

namespace abc_shared
{
	#include "utils/array.hpp" 
	#define STR_SIZE 20

	///@brief Choice of selection type
	enum Device {
		CPU = 0,
		GPU = 1
	};

	///@brief Choice of selection type
	enum Selection {
		ROULETTE_WHEEL = 0,
		RANK           = 1,
		TOURNAMENT     = 2
	};

	constexpr std::array<char, STR_SIZE> SelectionTypeToString(
		Selection selection_type
	)
	{
		switch(selection_type)
		{
		case ROULETTE_WHEEL:
			return utils::arr::str_to_arr<STR_SIZE>("roulette_wheel");
		case RANK:
			return utils::arr::str_to_arr<STR_SIZE>("rank");
		case TOURNAMENT:
			return utils::arr::str_to_arr<STR_SIZE>("tournament");
		}
		return utils::arr::str_to_arr<STR_SIZE>("ERROR");
	}

	///@brief Choice of roulette type
	enum Roulette {
		SUM     = 0,
		CUSTOM  = 1,
		MIN_MAX = 2
	};

	constexpr std::array<char, STR_SIZE> RouletteTypeToString(
		Roulette roulette_type
	)
	{
		switch(roulette_type)
		{
		case SUM:
			return utils::arr::str_to_arr<STR_SIZE>("sum");
		case CUSTOM:
			return utils::arr::str_to_arr<STR_SIZE>("custom");
		case MIN_MAX:
			return utils::arr::str_to_arr<STR_SIZE>("min_max");
		}
		return utils::arr::str_to_arr<STR_SIZE>("ERROR");
	};

	///@brief Choice of roulette type
	enum RouletteCpu {
		FULL         = 0,
		PARTIAL_SORT = 1,
		NONE         = 2
	};

	constexpr std::array<char, STR_SIZE> RouletteCpuToString(
		RouletteCpu roulette_sorting
	)
	{
		switch(roulette_sorting)
		{
		case FULL:
			return utils::arr::str_to_arr<STR_SIZE>("full");
		case PARTIAL_SORT:
			return utils::arr::str_to_arr<STR_SIZE>("partial_sort");
		case NONE:
			return utils::arr::str_to_arr<STR_SIZE>("none");
		}
		return utils::arr::str_to_arr<STR_SIZE>("ERROR");
	}

	///@brief Choice of rank selection type
	enum Rank {
		LINEAR_ARRAY             = 0,
		EXPONENTIAL_ARRAY        = 1,
		LINEAR_SIMPLE_ARRAY      = 2,
		EXPONENTIAL_SIMPLE_ARRAY = 3,
		CONSTANT_LINEAR          = 4,
		CONSTANT_EXPONENTIAL     = 5,
		CONSTANT_EXPONENTIAL_2   = 6
	};

	constexpr std::array<char, 20> RankTypeToString(
		Rank rank_type
	)
	{
		switch(rank_type)
		{
		case LINEAR_ARRAY:
			return utils::arr::str_to_arr<STR_SIZE>("linear_array");
		case EXPONENTIAL_ARRAY:
			return utils::arr::str_to_arr<STR_SIZE>("exponential_array");
		case LINEAR_SIMPLE_ARRAY:
			return utils::arr::str_to_arr<STR_SIZE>("linear_simple_array");
		case EXPONENTIAL_SIMPLE_ARRAY:
			return utils::arr::str_to_arr<STR_SIZE>("exponential_simple_array");
		case CONSTANT_LINEAR:
			return utils::arr::str_to_arr<STR_SIZE>("constant_linear");
		case CONSTANT_EXPONENTIAL:
			return utils::arr::str_to_arr<STR_SIZE>("constant_exponential");
		case CONSTANT_EXPONENTIAL_2:
			return utils::arr::str_to_arr<STR_SIZE>("constant_exponential_2");
		}
		return utils::arr::str_to_arr<STR_SIZE>("ERROR");
	}

	///@brief Choice of tournament selection type
	enum Tourn {
		SINGLE   = 0,
		MULTIPLE = 1
	};

	constexpr std::array<char, 20> TournamentTypeToString(
		Tourn tournament_type
	)
	{
		switch(tournament_type)
		{
		case SINGLE:
			return utils::arr::str_to_arr<STR_SIZE>("single");
		case MULTIPLE:
			return utils::arr::str_to_arr<STR_SIZE>("multiple");
		}
		return utils::arr::str_to_arr<STR_SIZE>("ERROR");
	}

	///@brief Function to optimize
	enum TestFunc {
		ROSENBROCK    = 0,
		CROSS_IN_TRAY = 1,
		SCHAFFER_2    = 2,
		SCHAFFER_4    = 3,
		BOHACHEVSKY_1 = 4,
		BOHACHEVSKY_2 = 5,
		BOHACHEVSKY_3 = 6,
		SCHWEFEL      = 7,
		INVALID       = 8
	};

	constexpr std::array<char, 20> TestFuncToString(
		TestFunc test_function
	)
	{
		switch(test_function)
		{
		case ROSENBROCK:
			return utils::arr::str_to_arr<STR_SIZE>("rosenbrock");
		case CROSS_IN_TRAY:
			return utils::arr::str_to_arr<STR_SIZE>("cross_in_tray");
		case SCHAFFER_2:
			return utils::arr::str_to_arr<STR_SIZE>("schaffer_2");
		case SCHAFFER_4:
			return utils::arr::str_to_arr<STR_SIZE>("schaffer_4");
		case BOHACHEVSKY_1:
			return utils::arr::str_to_arr<STR_SIZE>("bohachevsky_1");
		case BOHACHEVSKY_2:
			return utils::arr::str_to_arr<STR_SIZE>("bohachevsky_2");
		case BOHACHEVSKY_3:
			return utils::arr::str_to_arr<STR_SIZE>("bohachevsky_3");
		case SCHWEFEL:
			return utils::arr::str_to_arr<STR_SIZE>("schwefel");
		case INVALID:
			return utils::arr::str_to_arr<STR_SIZE>("INVALID");
		}
		return utils::arr::str_to_arr<STR_SIZE>("ERROR");
	}

	constexpr float GetGlobalMinValue(
		TestFunc test_function
	)
	{
		switch(test_function)
		{
		case ROSENBROCK:    return 0;
		case CROSS_IN_TRAY: return -2.06261;
		case SCHAFFER_2:    return 0;
		case SCHAFFER_4:    return 0.292579;
		case BOHACHEVSKY_1: return 0;
		case BOHACHEVSKY_2: return 0;
		case BOHACHEVSKY_3: return 0;
		case SCHWEFEL:      return 0;
		case INVALID:       return 0;
		}
		return 0;
	}

	template<unsigned int dim>
	struct SolutionParams
	{
		int num_of_solutions;
		std::array<std::array<float, dim>, 10> solutions;
	};

	template<unsigned int dim>
	constexpr void GetBounds(
		TestFunc test_function,
		float lower_bounds[dim],
		float upper_bounds[dim]
	)
	{
		switch(test_function)
		{
		case ROSENBROCK:
			lower_bounds[1] = -5;
			lower_bounds[0] = -5;
			upper_bounds[1] = 10;
			upper_bounds[0] = 10;
			return;
		case CROSS_IN_TRAY:
			lower_bounds[1] = -10;
			lower_bounds[0] = -10;
			upper_bounds[1] =  10;
			upper_bounds[0] =  10;
			return;
		case SCHAFFER_2:
		case SCHAFFER_4:
		case BOHACHEVSKY_1:
		case BOHACHEVSKY_2:
		case BOHACHEVSKY_3:
			lower_bounds[1] = -100;
			lower_bounds[0] = -100;
			upper_bounds[1] =  100;
			upper_bounds[0] =  100;
			return;
		case  SCHWEFEL:
			lower_bounds[1] = -500;
			lower_bounds[0] = -500;
			upper_bounds[1] =  500;
			upper_bounds[0] =  500;
			return;
		case INVALID:
			return;
		}
		return;
	}

	template<unsigned int dim>
	constexpr SolutionParams<dim> GetGlobalMinParams(
		TestFunc test_function
	)
	{
		std::array<std::array<float, dim>, 10> params = { };
		switch(test_function)
		{
		case ROSENBROCK:
			params[0] = { 1, 1 };
			return { 1, params };
		case CROSS_IN_TRAY:
			params[0] = {  1.34941,  1.34941 };
			params[1] = { -1.34941,  1.34941 };
			params[2] = {  1.34941, -1.34941 };
			params[3] = { -1.34941, -1.34941 };
			return { 4, params };
		case SCHAFFER_2:
			return { 1, { { 0, 0 } } };
		case SCHAFFER_4:
			params[0] = { 0,  1.25313 };
			params[1] = {  1.25313, 0 };
			params[2] = { 0, -1.25313 };
			params[3] = {  1.25313, 0 };
			return { 4, params };
		case BOHACHEVSKY_1:
		case BOHACHEVSKY_2:
		case BOHACHEVSKY_3:
			params[0] = { 0,  0 };
			return { 1, params };
		case  SCHWEFEL:
			params[0] = { 420.9687, 420.9687}; 
			return { 1, params };
		case INVALID:
			return { };
		}
		return { };
	}
}
