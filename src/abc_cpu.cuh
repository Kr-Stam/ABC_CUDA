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
 *****************************************************************************/

#pragma once

#include "problems/problems.h"
#include "utils/utils.hpp"
#include "rank_array.cuh"
#include "abc_main.cuh"
#include <vector>
#include <algorithm>
#include <math.h>

namespace cpu
{
	using namespace abc_shared;

	/*
	 * @brief A struct representing an individual Bee
	 * @details
	 * Each bee has a solution/food source it is working on,
	 * this solution has certain coordinates in the problem space
	 * and has a certain fitness value. An internal variable
	 * representing the number of trials is retained across 
	 * itertation in order to explore the viability of the solution.
	 * If the solution cannot be improved after a certain number
	 * of trial attempts then the solution is abandoned
	 */
	typedef struct
	{
		float coordinates[2];
		float value;
		int   trials;
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
	 * @param[in] bees_count
	 * @return minimum Bee
	 *
	 * @note This function is used for debugging and is not used in the main
	 *       algorithm
	 * */
	inline Bee min_bee(
		std::vector<Bee>*   bees,
		int                 bees_count
	)
	{
		Bee minimum = (*bees)[0];

		for(int idx = 1; idx < bees_count; idx++)
			if(minimum.value > (*bees)[idx].value)
				minimum = (*bees)[idx];

		return minimum;
	}

	/**
	 * @brief Generate a random solution within a bounded hypercube
	 * @param[out] bee
	 * @param[in]  function
	 * @param[in]  lower_bounds
	 * @param[in]  upper_bounds
	 * */
	template<uint32_t dimensions>
	void inline generate_random_solution(
		Bee*     bee,
		opt_func function,
		float    lower_bounds[],
		float    upper_bounds[]
	)
	{
		#pragma unroll
		for(int dim = 0; dim < dimensions; dim++)
		{
			(*bee).coordinates[dim] = 
				utils::random::rand_bounded_float(
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
	 * @param[in]    function
	 * @param[in]    lower_bounds
	 * @param[in]    upper_bounds
	 * @details Randomly select another bee and merge the solutions with
	 *          a stochastic step
	 * */
	template<uint32_t dimensions, uint32_t bees_count>
	void inline local_optimization(
		std::vector<Bee>* bees,
		opt_func          function,
		float             lower_bounds[],
		float             upper_bounds[]
	)
	{
		Bee tmp_bee;
		for(int i = 0; i < bees_count; i++)
		{
			int  choice = utils::random::rand_bounded_int(0, bees_count);
			float step  = utils::random::rand_bounded_float(0, 1);

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

	/*
	 * @brief Creates a roulette wheel with a cumulative distribution over 
	 *        the first num_of_candidates bees
	 *
	 * @param[in]  bees              A vector of candidate Bee pointers
	 * @param[out] roulette          The resulting roulette wheel
	 * @param[in]  num_of_candidates The number of candidates considered
	 *
	 * @note This roulette wheel is not sorted, if you want to select 
	 *       the top/best num_of_candidates bees then the Bee pointers
	 *       must be sorted beforehand
	 */
	template<
		uint32_t size,
		Roulette roulette_type
	>
	inline void create_roulette_wheel(
		std::vector<Bee>*       bees,
		std::array<float, size> roulette
	)
	{
		float min, max, sum;

		if constexpr (roulette_type == SUM)
		{
			sum = roulette[0];
			#pragma unroll
			for(int idx = 1; idx < size; idx++)
			{
				roulette[idx] = (*bees)[idx].value;
				sum += roulette[idx];
			}

			//cumulative distribution
			roulette[0] = roulette[0] / sum;
		}
		else if constexpr (roulette_type == CUSTOM)
		{
			sum = roulette[0];
			max = roulette[0];
			#pragma unroll
			for(int idx = 1; idx < size; idx++)
			{
				// the inverse is used in order to select for the 
				// global minimum
				//roulette[idx] = 1.00 / (*bees)[idx].value;
				roulette[idx] = (*bees)[idx].value;
				sum += roulette[idx];
				max = roulette[idx] > max ? roulette[idx] : max;
			}
			
			//cumulative distribution
			sum = max * size - sum;
			roulette[0] = (max - roulette[0]) / (size*max - sum);
		}
		else if constexpr (roulette_type == MIN_MAX)
		{
			max = roulette[0];
			min = roulette[0];
			#pragma unroll
			for(int idx = 1; idx < size; idx++)
			{
				roulette[idx] = (*bees)[idx].value;
				min = roulette[idx] < min ? roulette[idx] : min;
				max = roulette[idx] > max ? roulette[idx] : max;
			}
			
			//cumulative distribution
			sum = max * size - sum;
			roulette[0] = (roulette[0] - min) / (max - min);
		}

		#pragma unroll
		for(int idx = 1; idx < size; idx++)
		{
			if constexpr (roulette_type == SUM)
				roulette[idx] = roulette[idx] / sum;
			else if constexpr (roulette_type == CUSTOM)
				roulette[idx] = (max - roulette[idx]) / (size*max - sum);
			else if constexpr (roulette_type == MIN_MAX)
				roulette[idx] = (roulette[idx] - min) / (max - min);
			
			roulette[idx] += roulette[idx - 1];

			if(roulette[idx] < 0)
			{
				roulette[idx] = 1;
				break;
			}
		}
	}

	/*
	 * @brief Selects a random item from a roulette wheel according to
	 *        weighted roulette wheel selection
	 *
	 * @param[in] roulette vector of floats containing a cumulative
	 *                     probabilty distribution
	 * @param[in] rand     random float in the range [0, 1]
	 *
	 * @warning This function expects a cumulative distribution, all items
	 *          of the roulette should sum up to 1.0
	 */
	template<uint32_t size>
	inline int spin_roulette(std::array<float, size> roulette, float rand)
	{
		#pragma unroll
		for(int idx = 0; idx < size; idx++)
			if(rand <= roulette[idx]) return idx;
		
		return 0;
	}

	/**
	 * @brief Initialized a vector of Bees with initial food sources within
	 *        the hypercube enclosed by the given bounds
	 *
	 * @param[out] bees
	 * @param[in]  bees_count
	 * @param[in]  function
	 * @param[in]  lower_bounds
	 * @param[in]  upper_bounds
	 *
	 * @note This function is separated from the main ABC function so that
	 *       the ABC function can be called repeatedly and be clearly
	 *       separated into steps or batches. This was done for benchmarking
	 *       purposes
	 * */
	template<uint32_t dimensions>
	void init_bees(
		std::vector<Bee>* bees,
		int               bees_count,
		opt_func          function,
		float             lower_bounds[],
		float             upper_bounds[]
	)
	{
		utils::random::seed_random();

		//? ne mi se dopagja toa shto ova e nested funkcija ama mislam
		//? deka ova e najdobriot nachin na koj da bide strukturirano?
		// Initialize initial food sources 
		for(int idx = 0; idx < bees_count; idx++)
		{
			generate_random_solution<dimensions>(
				&(*bees)[idx],
				function,
				lower_bounds,
				upper_bounds
			);
		}
	}


	//1 + 2 + 3 + 4 + 5
	//i + i + 1 + ...
	//
	//i + i + c + i + 2c + ..
	//n * i + (1c + 2c + ...)
	//n * i + c*(1 + 2 + ...)
	//n * i + c*n(n+1)/2
	//
	//n * i + (1c + 2c + ...)
	// sum(1/pow(a, i))

	template<uint32_t n>
	int rank_selection_arr(float* arr_ranks, float rand)
	{
		for(int idx = 0; idx < n; idx++)
			if(rand <= arr_ranks[idx]) return idx;

		return 0;
	}

	template<uint32_t n>
	int rank_selection_arr(std::array<float, n> arr_ranks, float rand)
	{
		for(int idx = 0; idx < n; idx++)
			if(rand <= arr_ranks[idx]) return idx;

		return 0;
	}

	//ova dodava i povekje krugovi na selekcija
	template<
		uint32_t num_of_contestants,
		uint32_t num_of_games
	>
	int tournament_selection_multiple(std::vector<Bee>* bees)
	{
		int    max_idx   = 0;
		float max_value = (*bees)[max_idx].value;
		for(int i = 0; i < num_of_games; i++)
		{
			int choices[num_of_contestants];
			for(int j = 0; j < num_of_contestants; j++)
			{
				choices[j] = utils::random::rand_bounded_int(
					0,
					(*bees).size() - 1
				);
			}

			for(int j = 1; j < num_of_contestants; j++)
			{
				if(max_value < (*bees)[choices[j]].value)
				{
					max_value = (*bees)[choices[j]].value;
					max_idx   = j;
				}
			}
		}

		return max_idx;
	}

	//! ova treba da bide staveno vo poseben file
	template<uint32_t tournament_size>
	int tournament_selection_single(std::vector<Bee>* bees)
	{
		int choices[tournament_size];

		#pragma unroll
		for(int j = 0; j < tournament_size; j++)
		{
			choices[j] = utils::random::rand_bounded_int(
				0,
				(*bees).size() - 1
			);
		}

		int max_idx = 0;
		float max_value = (*bees)[choices[0]].value;

		#pragma unroll
		for(int j = 1; j < tournament_size; j++)
		{
			if(max_value < (*bees)[choices[j]].value)
			{
				//pobrzo e da se chuva tuka deka kje
				//se osiguram da bide stack promenliva
				max_value = (*bees)[choices[j]].value;
				max_idx   = j;
			}
		}

		return max_idx;
	}

	/**
	 * @brief An optimized sequential CPU implementation of the ABC algorithm
	 * 
	 * @warning All memory used is to be managed by the user
	 * @warning Concurrent modfication of the bees vector can lead to
	 *          unforeseen problems	as much of the iteration presuposes the 
	 *          bees vector to be a continous immutable array
	 *
	 * @param bees[inout]        Bees - potential solutions to be optimized
	 * @param bees_count[in]    The total number of bees
	 * @param max_generation[in] The maximum number of iterations
	 * @param trials_limit[in]   The number of optimization attempts after
	 *                           which a candidate solution is discarded
	 * @param scouts_ratio[in]   Determines the number of candidate solutions
	 *                           that onlooker bees can choose from.
	 *                           If set to 1.0 all bees are considered as
	 *                           potential candidates, if set to 0.2 only
	 *                           the top 20% are considered.
	 * @param function[in]       pointer to a function to be optimized of
	 *                           the type: (array(float), int) -> float
	 * @param lower_bounds[in] The lower bounds for the search space
	 * @param upper_bounds[in] The upper bounds for the search space
	 *
	 * @return void
	 */
	template<
		uint32_t   dimensions,
		uint32_t   bees_count,
		uint32_t   scouts_count,
		uint32_t   trials_limit, 
		Selection  selection_type,
		Roulette   roulette_type,
		SortingCpu sorting,
		Rank       rank_type,
		Tourn      tournament_type,
		uint32_t   tournament_size,
		uint32_t   tournament_num
	>
	void abc(
		std::vector<Bee>* bees,
		int               max_generations, 
		opt_func          function,
		float             lower_bounds[],
		float             upper_bounds[]
	)
	{
		constexpr uint32_t roulette_size = scouts_count *
			(selection_type == ROULETTE_WHEEL);

		constexpr uint32_t rank_arr_size = scouts_count *
			(selection_type == RANK && rank_type < CONSTANT_LINEAR);

		std::array<float, roulette_size> roulette;
		std::array<float, rank_arr_size> rank_arr;

		if constexpr(selection_type == RANK)
		{
			if constexpr (rank_type == LINEAR_ARRAY)
				rank_arr = rank_arr::arr_lin<rank_arr_size>(1.9f); 
			else if constexpr (rank_type == EXPONENTIAL_ARRAY)
				rank_arr = rank_arr::arr_exp<rank_arr_size>(1.1f);
			else if constexpr (rank_type == LINEAR_SIMPLE_ARRAY)
				rank_arr = rank_arr::arr_simple<rank_arr_size>();
			else if constexpr (rank_type == EXPONENTIAL_SIMPLE_ARRAY)
				rank_arr = rank_arr::arr_simple_exp<rank_arr_size>(0.5f);
		}
		//-----------------------MAIN-LOOP---------------------------//
		for(int i = 0; i < max_generations; i++)
		{
			//-----------EMPLOYED-BEE-LOCAL-OPTIMIZATION---------//
			local_optimization<dimensions, bees_count>(	
				bees,
				function, 
				lower_bounds,
				upper_bounds
			);
			//---------------------------------------------------//

			//--------ONLOOKER-BEE-GLOBAL-OPTIMIZATION-----------//
			
			//-------------roulette wheel selection---------------//
			
			if constexpr (selection_type == ROULETTE_WHEEL)
			{
				// Execute roulette wheel selection on the best n candidates
				if constexpr (sorting == PARTIAL_SORT)
				{
					std::partial_sort(
						(*bees).begin(),
						(*bees).begin() + scouts_count,
						(*bees).end(),
						BeeCompare
					);
					create_roulette_wheel<
						scouts_count,
						roulette_type
					>(
						bees,
						roulette
					);
				}
				else if constexpr (sorting == FULL_SORT)
				{
					create_roulette_wheel<
						roulette_size,
						roulette_type
					>(
						bees,
						roulette
					);
				}
			}
			else if constexpr (selection_type == RANK)
			{
				// Execute roulette wheel selection on the best n candidates
				if constexpr (sorting == PARTIAL_SORT)
				{
					std::partial_sort(
						(*bees).begin(),
						(*bees).begin() + scouts_count,
						(*bees).end(),
						BeeCompare
					);
				}
				else if constexpr (sorting == FULL_SORT)
				{
					std::sort(
						(*bees).begin(),
						(*bees).end(),
						BeeCompare
					);
				}
			}


			for(int idx = 0; idx < bees_count; idx++)
			{
				int choice;

				if constexpr (selection_type == ROULETTE_WHEEL)
				{
					float rand;
					rand = utils::random::rand_bounded_float(0, 1);

					choice = spin_roulette<roulette_size>(roulette, rand);
				}
				else if constexpr (selection_type == RANK)
				{
					float rand;
					rand = utils::random::rand_bounded_float(0, 1);

					if constexpr (rank_type < CONSTANT_LINEAR)
						choice = rank_selection_arr<rank_arr_size>(rank_arr, rand);
					else if constexpr (rank_type == CONSTANT_LINEAR)
						choice = rank_const::lin<scouts_count>(rand);
					else if constexpr (rank_type == CONSTANT_EXPONENTIAL)
						choice = rank_const::exp<scouts_count, 1, 20>(rand);
					else if constexpr (rank_type == CONSTANT_EXPONENTIAL_2)
						choice = rank_const::exp2<scouts_count, 1, 20>(rand);
				}
				else if constexpr (selection_type == TOURNAMENT)
				{
					if constexpr (tournament_type == SINGLE)
						choice = tournament_selection_single<tournament_size>(
							bees
						);
					else if constexpr (tournament_type == MULTIPLE)
						choice = tournament_selection_multiple<tournament_size, tournament_num>(bees);
				}
				
				if(choice < 0) choice = 0;
				if((*bees)[idx].value > (*bees)[choice].value)
				{
					(*bees)[idx] = (*bees)[choice];
					(*bees)[idx].trials = 0;
				}
				else
				{
					(*bees)[idx] = (*bees)[choice];
					(*bees)[idx].trials++;
				}
			}

			//---------------------------------------------------//

			//----------------TRIAL-LIMIT-CHECK------------------//
			for(int idx = 0; idx < bees_count; idx++)
			{
				if((*bees)[idx].trials > trials_limit)
				{
					generate_random_solution<dimensions>(
							&(*bees)[idx],
							function,
							lower_bounds,
							upper_bounds
					);
				}
			}
			//---------------------------------------------------//
		}
		//-----------------------------------------------------------//
	}

}
