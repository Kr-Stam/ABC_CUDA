#include "problems/problems.cuh"
#include "vector"

namespace cpu
{
	typedef struct
	{
		double coordinates[2];
		double value;
		int    trials;
	} Bee;

	void init_bees(
			std::vector<Bee>*   bees,
			int                 num_of_bees,
			OptimizationProblem optimization_problem,
			double              lower_bounds[],
			double              upper_bounds[]
	);

	void abc(
		std::vector<Bee>*   bees,
		int                 num_of_bees, 
		int                 max_generations, 
		int                 trials_limit, 
		double              ratio_of_scouts,
		OptimizationProblem optimization_problem,
		double              lower_bounds[],
		double              upper_bounds[]
	);

	Bee min_bee(
		std::vector<Bee>*   bees,
		int                 num_of_bees
	);
}
