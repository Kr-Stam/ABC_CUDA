#pragma once

namespace abc_shared
{
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

	const char* SelectionTypeToString(Selection selection_type)
	{
		switch(selection_type)
		{
		case ROULETTE_WHEEL: return "ROULETTE_WHEEL";
		case RANK:           return "RANK";
		case TOURNAMENT:     return "TOURNAMENT";
		}
		return "ERROR";
	}

	///@brief Choice of roulette type
	enum Roulette {
		SUM            = 0,
		CUSTOM         = 1,
		MIN_MAX        = 2,
		//! ova ne e implementirano
		MIN_MAX_SCALED = 3
	};

	const char* RouletteTypeToString(Roulette roulette_type)
	{
		switch(roulette_type)
		{
		case SUM:            return "SUM";
		case CUSTOM:         return "CUSTOM";
		case MIN_MAX:        return "MIN_MAX";
		case MIN_MAX_SCALED: return "MIN_MAX_SCALED";
		}
		return "ERROR";
	};


	///@brief Choice of roulette type
	enum RouletteCpu {
		FULL         = 0,
		PARTIAL_SORT = 1,
		NONE         = 2
	};

	const char* RouletteCpuToString(RouletteCpu roulette_sorting)
	{
		switch(roulette_sorting)
		{
		case FULL:         return "FULL";
		case PARTIAL_SORT: return "PARTIAL_SORT";
		}
		return "ERROR";
	}

	///@brief Choice of rank selection type
	enum Rank {
		LINEAR_ARRAY                   = 0,
		EXPONENTIAL_ARRAY              = 1,
		LINEAR_SIMPLE_ARRAY            = 2,
		EXPONENTIAL_SIMPLE_ARRAY       = 3,
		CONSTANT_LINEAR                = 4,
		CONSTANT_EXPONENTIAL           = 5,
		CONSTANT_EXPONENTIAL_2         = 6
	};

	const char* RankTypeToString(Rank rank_type)
	{
		switch(rank_type)
		{
		case LINEAR_ARRAY:             return "LINEAR_ARRAY";
		case EXPONENTIAL_ARRAY:        return "EXPONENTIAL_ARRAY";
		case LINEAR_SIMPLE_ARRAY:      return "LINEAR_SIMPLE_ARRAY";
		case EXPONENTIAL_SIMPLE_ARRAY: return "EXPONENTIAL_SIMPLE_ARRAY";
		case CONSTANT_LINEAR:          return "CONSTANT_LINEAR";
		case CONSTANT_EXPONENTIAL:     return "CONSTANT_EXPONENTIAL";
		case CONSTANT_EXPONENTIAL_2:   return "CONSTANT_EXPONENTIAL_2";
		}
		return "ERROR";
	}

	///@brief Choice of tournament selection type
	enum Tourn {
		SINGLE   = 0,
		MULTIPLE = 1
	};

	const char* TournamentTypeToString(Tourn tournament_type)
	{
		switch(tournament_type)
		{
		case SINGLE:   return "SINGLE";
		case MULTIPLE: return "MULTIPLE";
		}
		return "ERROR";
	}

}
