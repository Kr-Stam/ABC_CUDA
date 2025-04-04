/******************************************************************************
 * @file abc_cpu.h                                                            *
 * @brief Optimized sequential cpu implementation of the ABC algorihtm        *
 * @details Further optimization is possible although deemed unnecessary as   *
 *          the projects's main goal is the parallelization of this algorihtm *
 *                                                                            *
 * @note Almost all of the functions are declared within the header file      *
 *       because they were templated in order to optimize performance for     *
 *       each dimension size                                                  *
 * @author Kristijan Stameski                                                 *
 ******************************************************************************/

#pragma once

#include "problems/problems.h"
#include "utils/utils.hpp"
#include <vector>
#include <algorithm>

namespace cpu
{
	/**
	 * @brief A struct representing an individual Bee
	 * @details Each bee has a solution/food source it is working on,
	 *          this solution has certain coordinates in the problem space
	 *          and has a certain fitness value. An internal variable
	 *          representing the number of trials is retained across 
	 *          itertation in order to explore the viability of the solution.
	 *          If the solution cannot be improved after a certain number
	 *          of trial attempts then the solution is abandoned
	 * */
	typedef struct
	{
		double coordinates[2];
		double value;
		int    trials;
	} Bee;

	///@brief comparator for sorting Bees
	const struct
	{
		bool operator()(Bee a, Bee b) const 
		{
			return a.value < b.value; 
		}
	} BeeCompare;

	/**
	 * @brief Find the minimum bee in the vector of bees
	 * @param[in] bees
	 * @param[in] num_of_bees
	 * @return minimum Bee
	 *
	 * @note This function is used for debugging and is not used in the main
	 *       algorithm
	 * */
	inline Bee min_bee(
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
	 * @brief Generate a random solution within a bounded hypercube
	 * @param[out] bee
	 * @param[in]  function
	 * @param[in]  lower_bounds
	 * @param[in]  upper_bounds
	 * */
	template<unsigned int dimensions>
	void inline generate_random_solution(
		Bee*     bee,
		opt_func function,
		double   lower_bounds[],
		double   upper_bounds[]
	)
	{
		#pragma unroll
		for(int dim = 0; dim < dimensions; dim++)
		{
			(*bee).coordinates[dim] = 
				utils::random::rand_bounded_double(
				lower_bounds[dim],
				upper_bounds[dim]
			);
		}

		(*bee).value = function((*bee).coordinates, dimensions);
		(*bee).trials = 0;
	}

	/**
	 * @brief Local optimization around the existing food source
	 * @param[inout] bee
	 * @param[in]     function
	 * @param[in]     lower_bounds
	 * @param[in]     upper_bounds
	 * @details Randomly select another bee and merge the solutions with
	 *          a stochastic step
	 * */
	template<unsigned int dimensions>
	void inline local_optimization(
			std::vector<Bee>* bees,
			int               num_of_bees,
			opt_func          function,
			double            lower_bounds[],
			double            upper_bounds[]
	)
	{
		Bee tmp_bee;
		for(int i = 0; i < num_of_bees; i++)
		{
			int	choice  = utils::random::rand_bounded_int(0, num_of_bees);
			double step = utils::random::rand_bounded_double(0, 1);
			#pragma unroll
			for(int dim = 0; dim < dimensions; dim++)
			{
				tmp_bee.coordinates[dim] = 
					utils::fast_clip(
						(*bees)[i].coordinates[dim] + 
						step * (
							(*bees)[choice].coordinates[dim] +
							(*bees)[choice].coordinates[dim]
						),
						lower_bounds[dim],
						upper_bounds[dim]
					);
			}
			tmp_bee.value = function(tmp_bee.coordinates, dimensions);
			if(tmp_bee.value < (*bees)[i].value)
			{
				(*bees)[i] = tmp_bee;
				(*bees)[i].trials = 0;
			}
			else
			{
				(*bees)[i].trials++;
			}
		}
	}

	//TODO: Treba da se napravi posebna verzija za drugi tipovi na selekcija

	/**
	 * @brief Creates a roulette wheel with a cumulative distribution over 
	 *        the first num_of_candidates bees
	 *
	 * @param[in]  bees              A vector of candidate Bee pointers
	 * @param[out] roulette          The resulting roulette wheel
	 * @param[in]  num_of_candidates The number of candidates to be considered
	 *
	 * @note This roulette wheel is not sorted, if you want to select 
	 *       the top/best num_of_candidates bees then the Bee pointers
	 *       must be sorted beforehand
	 * */
	inline void create_roulette_wheel(
		std::vector<Bee>*    bees,
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
			//? ne znam dali ova e najpametnata opcija deka vo gpu delot go
			//? pravam na razlichen nachin
			
			//the inverse is used in order to select for the global minimum
			(*roulette)[bee_idx] = 1.00/(*bees)[bee_idx].value;
			sum += (*roulette)[bee_idx];
		}

		//cumulative distribution
		(*roulette)[0] = (*roulette)[0] / sum;
		#pragma unroll
		for(int bee_idx = 1; bee_idx < num_of_candidates; bee_idx++)
		{
			(*roulette)[bee_idx] = (*roulette)[bee_idx] / sum +
			                       (*roulette)[bee_idx - 1];
		}
	}

	/**
	 * @brief Selects a random item from a roulette wheel according to weighted
	 *        roulette wheel selection
	 *
	 * @param[in] roulette A vector of doubles containing a cumulative probabilty
	 *            distribution
	 *
	 * @warning This function expects a cumulative distribution, all items
	 *          of the roulette should sum up to 1.0
	 * */
	inline int spin_roulette(
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

	/**
	 * @brief Initialized a vector of Bees with initial food sources within
	 *        the hypercube enclosed by the given bounds
	 *
	 * @param[out] bees
	 * @param[in]  num_of_bees
	 * @param[in]  function
	 * @param[in]  lower_bounds
	 * @param[in]  upper_bounds
	 *
	 * @note This function is separated from the main ABC function so that
	 *       the ABC function can be called repeatedly and be clearly separated
	 *       into steps or batches. This was done for benchmarking purposes
	 * */
	template<unsigned int dimensions>
	void init_bees(
			std::vector<Bee>* bees,
			int               num_of_bees,
			opt_func          function,
			double            lower_bounds[],
			double            upper_bounds[]
	)
	{
		utils::random::seed_random();

		//? ne mi se dopagja toa shto ova e nested funkcija ama mislam deka
		//? ova e najdobriot nachin na koj da bide strukturirano?
		//Initialize initial food sources 
		for(int bee_idx = 0; bee_idx < num_of_bees; bee_idx++)
		{
			generate_random_solution<dimensions>(
				&(*bees)[bee_idx],
				function,
				lower_bounds,
				upper_bounds
			);
		}
	}

	/**
	 * @brief An optimized sequential CPU implementation of the ABC algorithm
	 * 
	 * @warning All memory used by this function is to be managed by the user
	 * @warning Concurrent modfication of the bees vector can lead to
	 *          unforeseen problems	as much of the iteration presuposes the 
	 *          bees vector to be a continous immutable array
	 *
	 * @param bees[inout]         Bees are potential solutions to be optimized
	 * @param num_of_bees[in]     The total number of bees
	 * @param max_generation[in]  The maximum number of iterations
	 * @param trials_limit[in]    The number of optimization attempts after
	 *                            which a candidate solution will be discarded
	 * @param ratio_of_scouts[in] Determines the number of candidate solutions
	 *                            that onlooker bees can choose from.
	 *                            If set to 1.0 all bees are considered as
	 *                            potential candidates, if set to 0.2 only
	 *                            the top 20% are considered.
	 * @param function[in]        pointer to a function to be optimized
	 *                            of the type: (array(double), int) -> double
	 * @param lower_bounds[in] Determines the lower bounds for the search space
	 * @param upper_bounds[in] Determines the upper bounds for the search space
	 *
	 * @return void
	 */
	template<unsigned int dimensions>
	void abc(
		std::vector<Bee>*  bees,
		int                num_of_bees, 
		int                max_generations, 
		int                trials_limit, 
		double             ratio_of_scouts,
		opt_func           function,
		double             lower_bounds[],
		double             upper_bounds[]
	)
	{
		int num_of_scouts = (int) (((double) num_of_bees) * ratio_of_scouts);
		std::vector<double> roulette(num_of_scouts);

		//---------------------------MAIN-LOOP--------------------------------//
		for(int i = 0; i < max_generations; i++)
		{
			//----------------EMPLOYED-BEE-LOCAL-OPTIMIZATION-----------------//
			local_optimization<dimensions>(
				bees,
				num_of_bees,
				function, 
				lower_bounds,
				upper_bounds
			);
			//----------------------------------------------------------------//

			//----------------ONLOOKER-BEE-GLOBAL-OPTIMIZATION----------------//
			//Vo ovaa implementacija na selekcija mora da se 
			//sortiraat rezultatite
			
			//std::sort((*bees).begin(), (*bees).end(), BeeCompare);
			//? se koristi std::partial_sort poradi toa shto nadvor od opsegot
			//? beshe da napravam optimizirana implementacija na quicksort
			//? ili merge_sort specifichno za pcheli (voglavno poradi odlukata
			//? za abstrakcija da bide  prikazhano kako struktura i problemi
			//? okolu internoto chuvanje na pochetniot indeks)
			//! TODO: Tuka mora da se pretstavat i alternativi na ovaa selekcija
			std::partial_sort(
				(*bees).begin(),
				(*bees).begin() + num_of_scouts,
				(*bees).end(),
				BeeCompare
			);
			//TODO: Dodadi i drugi tipovi na selekcija
			//Roulette selection
			//initialize roulette
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
			//----------------------------------------------------------------//

			//-----------------------TRIAL-LIMIT-CHECK------------------------//
			for(int bee_idx = 0; bee_idx < num_of_bees; bee_idx++)
			{
				if((*bees)[bee_idx].trials > trials_limit)
				{
					generate_random_solution<dimensions>(
							&(*bees)[bee_idx],
							function,
							lower_bounds,
							upper_bounds
					);
				}
			}
			//----------------------------------------------------------------//
		}
		//--------------------------------------------------------------------//
	}
}
