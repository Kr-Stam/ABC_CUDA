#include "abc_cpu.hpp"
#include "problems/problems.hpp"
#include "utils/utils.hpp"
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <time.h>
#include <stdio.h>

using namespace cpu;

//TODO: dobro prashanje e za dali treba generichno da definiram
//      za sekoja golemina na problem (mnogu for loops)

struct
{
	bool operator()(Bee a, Bee b) const 
	{
		return a.value < b.value; 
	}
} BeeCompare;

//TODO: treba da se testira dali e soodvetno da imam inline 
void inline generate_random_solution(
		Bee* bee,
		OptimizationProblem* optimization_problem,
		double lower_bounds[],
		double upper_bounds[]
)
{
	#pragma unroll
	for(int dim = 0; dim < (*optimization_problem).n; dim++)
	{
		(*bee).coordinates[dim] = 
			utils::random::rand_bounded_double(
			lower_bounds[dim],
			upper_bounds[dim]
		);
	}

	(*bee).value = 
		(*optimization_problem).function(
			(*bee).coordinates,
			(*optimization_problem).n
		);
	(*bee).trials = 0;
}

void inline local_optimization(
		Bee* bee,
		OptimizationProblem* optimization_problem,
		double lower_bounds[],
		double upper_bounds[]
)
{
	Bee tmp = (*bee);

	#pragma unroll
	for(int dim = 0; dim < (*optimization_problem).n; dim++)
	{
		double step = utils::random::rand_bounded_double(
			lower_bounds[dim] / 10,
			upper_bounds[dim] / 10
		);
		tmp.coordinates[dim] += step;
		tmp.coordinates[dim] = 
			utils::fast_clip(
				tmp.coordinates[dim],
				lower_bounds[dim],
				upper_bounds[dim]
			);
	}

	tmp.value = 
		(*optimization_problem).function(
				tmp.coordinates,
				(*optimization_problem).n
		);

	if(tmp.value < (*bee).value)
	{
		(*bee) = tmp;
		(*bee).trials = 0;
	}
	else
	{
		(*bee).trials++;
	}
}

void create_roulette_wheel(
	std::vector<Bee>*     bees,
	std::vector<double>* roulette,
	int num_of_candidates,
	int total_num
)
{
	//sum reduction
	double sum = 0;
	#pragma unroll
	for(int bee_idx = 0; bee_idx < num_of_candidates; bee_idx++)
	{
		//tuka go koristam inverznoto deka go baram globalniot minimum
		(*roulette)[bee_idx] = 1.00/(*bees)[bee_idx].value;
		sum += (*roulette)[bee_idx];
	}

	//cumulative distribution
	(*roulette)[0] = (*roulette)[0] / sum;
	#pragma unroll
	for(int bee_idx = 1; bee_idx < num_of_candidates; bee_idx++)
	{
		(*roulette)[bee_idx] = (*roulette)[bee_idx] / sum + (*roulette)[bee_idx - 1];
	}
}

int spin_roulette(
		std::vector<double> roulette
)
{
	double choice = utils::random::rand_bounded_double(0, 1);
	for(int idx = 0; idx < roulette.size(); idx++)
	{
		if(choice <= roulette[idx])
		{
			return idx;
		}
	}
	return 0;
}

void cpu::init_bees(
		std::vector<Bee>*   bees,
		int                 num_of_bees,
		OptimizationProblem optimization_problem,
		double              lower_bounds[],
		double              upper_bounds[]
)
{
	utils::random::seed_random();
	//Initialize initial food sources 
	for(int bee_idx = 0; bee_idx < num_of_bees; bee_idx++)
	{
		generate_random_solution(
			&(*bees)[bee_idx],
			&optimization_problem,
			lower_bounds,
			upper_bounds
		);
	}
}

Bee cpu::min_bee(
	std::vector<Bee>*   bees,
	int                 num_of_bees
)
{
	Bee minimum = (*bees)[0];
	for(int bee_idx = 1; bee_idx < num_of_bees; bee_idx++)
	{
		if(minimum.value > (*bees)[bee_idx].value)
			minimum = (*bees)[bee_idx];
	}
	return minimum;
}

/**
 * @brief An optimized sequential implementation of the artificial bee colony algorithm
 * 
 * @warning All memory used by this function is to be managed by the user
 * @warning Concurrent modfication of the bees vector can lead to unforeseen problems
 *          as much of the iteration presuposes the bees vector to be a continous
 *          immutable array
 *
 * @param bees[inout] Bees represent solutions that are to be optimized
 * @param num_of_bees[in]
 * @param max_generation[in] The number of iterations the algorithm will run
 * @param trials_limit[in] The number of optimization attempts after which
 *        a candidate solution will be discarded by a bee
 * @param ratio_of_scouts[in] Determines the number of candidate solutions
 *        that onlooker bees can choose from. If set to 1.0 all bees are
 *        considered as potential candidates, if set 0.2 only the top 20%
 *        are considered.
 * @param optimization_problem[in] defines a function pointer to a
 *        function of the type: array(double) -> double
 * @param lower_bounds[in] Determines the lower bounds for the search space
 * @param upper_bounds[in] Determines the upper bounds for the search space
 *
 * @return void
 */
void cpu::abc(
	std::vector<Bee>*   bees,
	int                 num_of_bees, 
	int                 max_generations, 
	int                 trials_limit, 
	double              ratio_of_scouts,
	OptimizationProblem optimization_problem,
	double              lower_bounds[],
	double              upper_bounds[]
)
{
	int num_of_scouts    = (int) (((double) num_of_bees) * ratio_of_scouts);

	//Main Loop
	for(int i = 0; i < max_generations; i++)
	{
		//Employed Bee Local Optimization
		for(int bee_idx = 0; bee_idx < num_of_bees; bee_idx++)
		{
			local_optimization(
				&(*bees)[bee_idx], 
				&optimization_problem, 
				lower_bounds,
				upper_bounds
			);
		}
		//Sort
		//std::sort((*bees).begin(), (*bees).end(), BeeCompare);
		std::partial_sort(
			(*bees).begin(),
			(*bees).begin() + num_of_scouts,
			(*bees).end(),
			BeeCompare
		);

		//Roulette selection
		//initialize roulette
		std::vector<double> roulette(num_of_scouts);
		create_roulette_wheel(
			&(*bees),
			&roulette,
			num_of_scouts,
			num_of_bees
		);
		//do selection
		for(int bee_idx = num_of_scouts; bee_idx < num_of_bees; bee_idx++)
		{
			int spin = spin_roulette(roulette);
			if((*bees)[bee_idx].value > (*bees)[spin].value)
			{
				(*bees)[bee_idx] = (*bees)[spin];
				(*bees)[bee_idx].trials = 0;
			}
			else
			{
				(*bees)[bee_idx] = (*bees)[spin];
				(*bees)[bee_idx].trials++;
			}
		}

		//Search for new solutions
		for(int bee_idx = 0; bee_idx < num_of_bees; bee_idx++)
		{
			if((*bees)[bee_idx].trials > trials_limit)
			{
				generate_random_solution(
						&(*bees)[bee_idx],
						&optimization_problem,
						lower_bounds,
						upper_bounds
				);
			}
		}
	}

}

