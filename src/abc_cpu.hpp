#include "problems/problems.hpp"
#include "vector"

namespace cpu
{
	typedef struct
	{
		double coordinates[2];
		double value;
		int    trials;
	} Bee;

	void abc(
		std::vector<Bee>*   bees,
		int                 num_of_bees, 
		int                 max_generations, 
		int                 trials_limit, 
		double              ratio_of_scouts,
		double              ratio_of_onlookers,
		OptimizationProblem optimization_problem,
		double              lower_bounds[],
		double              upper_bounds[]
	);
}
