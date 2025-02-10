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

double clip(double value, double lower_bound, double upper_bound)
{
	if (value < lower_bound)
		return lower_bound;
	else if (value > upper_bound)
		return upper_bound;
	else
		return value;
}

struct
{
	bool operator()(Bee a, Bee b) const 
	{
		return a.value < b.value; 
	}
} BeeCompare;

void inline generate_random_solution(
		Bee* bee,
		OptimizationProblem* optimization_problem,
		double lower_bounds[],
		double upper_bounds[]
)
{
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
	for(int dim = 0; dim < (*optimization_problem).n; dim++)
	{
		double step = utils::random::rand_bounded_double(
			lower_bounds[dim] / 10,
			upper_bounds[dim] / 10
		);
		tmp.coordinates[dim] += step;
		tmp.coordinates[dim] = 
			clip(
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
	double sum = 0;
	for(int bee_idx = 0; bee_idx < num_of_candidates; bee_idx++)
	{
		(*roulette)[bee_idx] = 1.00/(*bees)[bee_idx].value;
		sum += (*roulette)[bee_idx];
	}
	for(int bee_idx = 0; bee_idx < num_of_candidates; bee_idx++)
	{
		//printf("roulette %.2f after %.2f", roulette[bee_idx], 100 * roulette[bee_idx] / sum);
		(*roulette)[bee_idx] = (*roulette)[bee_idx] / sum;
	}
	for(int bee_idx = 1; bee_idx < num_of_candidates; bee_idx++)
	{
		(*roulette)[bee_idx] += (*roulette)[bee_idx - 1];
	}
	//printf("roulette: ");
	//for(int i = 0; i < 10; i++)
	//	printf("%.2f ", (*roulette)[i]);
	//printf("\n");
}

///\brief This is a sequential implementation of the
///       artificial bee colony algorithm
///\details 
void cpu::abc(
	std::vector<Bee>*   bees,
	int                 num_of_bees, 
	int                 max_generations, 
	int                 trials_limit, 
	double              ratio_of_scouts,
	double              ratio_of_onlookers,
	OptimizationProblem optimization_problem,
	double              lower_bounds[],
	double              upper_bounds[]
)
{
	utils::random::seed_random();

	//this array contains all choice params of the (*bees)
	//Bee* (*bees) = (Bee*) malloc(
	//	    sizeof(double) *
	//		num_of_(*bees) *
	//		(optimization_problem.n + 1)
	//);

	int num_of_scouts    = (int) (((double) num_of_bees) * ratio_of_scouts);
	//ova sakam da go koristam za da imam granica na 90ti percentil
	//int num_of_onlookers = (int) (((double) num_of_(*bees)) * ratio_of_scouts);

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
			double choice = utils::random::rand_bounded_double(0, 1);
			for(int j = 0; j < num_of_scouts; j++)
			{
				if(choice <= roulette[j])
				{
					if((*bees)[j].value < (*bees)[bee_idx].value)
					{
						(*bees)[bee_idx] = (*bees)[j];
						(*bees)[bee_idx].trials = 0;
					}
					else
					{
						(*bees)[bee_idx].trials++;
					}
				}
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
		//printf("After selection\n");
		//for(int bee_idx = num_of_scouts; bee_idx < num_of_scouts + 10; bee_idx++)
		//{
		//	printf("x: %.2f y: %.2f value: %.2f\n", 
		//			(*bees)[bee_idx].coordinates[0],
		//			(*bees)[bee_idx].coordinates[1],
		//			(*bees)[bee_idx].value);
		//}
		//while(true){}

	}

}

