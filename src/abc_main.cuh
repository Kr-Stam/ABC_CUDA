#pragma once

namespace abc_shared
{
	///@brief Choice of selection type
	enum Selection {
		ROULETTE_WHEEL = 0,
		RANK           = 1,
		TOURNAMENT     = 2
	};

	///@brief Choice of roulette type
	enum Roulette {
		SUM            = 0,
		CUSTOM         = 1,
		MIN_MAX        = 2,
		//! ova ne e implementirano
		MIN_MAX_SCALED = 3
	};

	///@brief Choice of roulette type
	enum RouletteCpu {
		FULL         = 0,
		PARTIAL_SORT = 1
	};

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

	///@brief Choice of tournament selection type
	enum Tourn {
		SINGLE   = 0,
		MULTIPLE = 1
	};
}
