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
#include <math.h>

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
		{
			if(minimum.value > (*bees)[idx].value)
				minimum = (*bees)[idx];
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
	 * @param[in]    function
	 * @param[in]    lower_bounds
	 * @param[in]    upper_bounds
	 * @details Randomly select another bee and merge the solutions with
	 *          a stochastic step
	 * */
	template<unsigned int dimensions>
	void inline local_optimization(
			std::vector<Bee>* bees,
			int               bees_count,
			opt_func          function,
			double            lower_bounds[],
			double            upper_bounds[]
	)
	{
		Bee tmp_bee;
		for(int i = 0; i < bees_count; i++)
		{
			int  choice = utils::random::rand_bounded_int(0, bees_count);
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
	 * @param[in]  num_of_candidates The number of candidates considered
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
		for(int idx = 0; idx < num_of_candidates; idx++)
		{
			//? ne znam dali ova e najpametnata opcija deka vo
			//? gpu delot go pravam na razlichen nachin
			
			// the inverse is used in order to select for the 
			// global minimum
			(*roulette)[idx] = 1.00 / (*bees)[idx].value;
			sum += (*roulette)[idx];
		}

		//cumulative distribution
		(*roulette)[0] = (*roulette)[0] / sum;
		#pragma unroll
		for(int idx = 1; idx < num_of_candidates; idx++)
		{
			(*roulette)[idx] = (*roulette)[idx] / sum +
			                   (*roulette)[idx - 1];
		}
	}

	/**
	 * @brief Selects a random item from a roulette wheel according to
	 *        weighted roulette wheel selection
	 *
	 * @param[in] roulette A vector of doubles containing a cumulative
	 *            probabilty distribution
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
			if(choice <= roulette[idx]) return idx;
		
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
	template<unsigned int dimensions>
	void init_bees(
			std::vector<Bee>* bees,
			int               bees_count,
			opt_func          function,
			double            lower_bounds[],
			double            upper_bounds[]
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
	/**
	 * @brief Initializes an array of weights for rank selection
	 * @param[out] arr Array of at least size n
	 * @param[in]  n   Number of candidates
	 * @return The sum of all elements of the initialized array
	 *
	 * @note Even though this function initializes weights by rank,
	 *        it does not use the standard rank selection algorithm
	 * */
	template<unsigned int n>
	float init_rank_arr_custom(float* arr)
	{
		int sum = n * (n + 1) / 2;

		arr[0] = (float) n / (float) sum;

		#pragma unroll
		for(int i = 1; i < n; i++) 
		{
			arr[i] = arr[i - 1] + (float) (n - i) / (float) sum;
		}

		return (float) sum;
	}

	/**
	 * @brief Initializes an array of weights for rank selection
	 * @param[out] arr Array of at least size n
	 * @param[in]  n   Number of candidates
	 * @return The sum of all elements of the initialized array
	 *
	 * @note Even though this function initializes weights by rank,
	 *        it does not use the standard rank selection algorithm
	 * */
	template<unsigned int n>
	float init_rank_arr_custom_exponential(float* arr, float c)
	{
		//sumata e (c^(n+1) - 1) / (c - 1)
		float sum = (pow(c, n + 1) - 1) / (c - 1);

		arr[0] = 1 / sum;

		#pragma unroll
		for(int i = 1; i < n; i++) 
		{
			arr[i] = arr[i - 1] + pow(c, i) / sum;
		}

		return (float) sum;
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

	//bazirano na:
	//An Analysis of Linear Ranking and Binary Tournament Selection in
	//Genetic Algorithms shto e bazirano na ova:
	//Adaptive Selection Methods for Genetic Algorithms 
	//James Edward Baker 
	//Computer Science Department 
	//Vanderbilt University 
	//according to the paper the most ideal value of max is 1.1
	/**
	 * @brief Initializes an array of weights for linear rank selection
	 * @param[out] arr Array of at least size n
	 * @param[in]  a   Must be less than 2.0
	 * @param[in]  n   Number of candidates
	 *
	 * @return The sum of all elements of the initialized array
	 *
	 * @note According to cited reasearch the ideal value of a is 1.1
	 * */
	template<unsigned int n>
	float init_rank_arr_linear(
		float* arr_out,
		float  a 
	)
	{
		if(a > 2 || n < 1) return 0;

		float b = 2 - a;

		float c = 1.0 / n * (b + (a - b));
		arr_out[0] = c;
		#pragma unroll
		for(int i = 1; i < n; i++)
		{
			c = 1.0 / n * (b + (a - b) * (n - i - 1.0) / (n - 1.0));
			arr_out[i] = arr_out[i - 1] + c;
		}

		return arr_out[n - 1];
	}

	template<unsigned int n>
	float init_rank_arr_exponential(
		float* arr_out,
		float  c
	)
	{
		if(n < 1) return 0;

		float c_min_1     = 1 - c;
		float c_nth_min_i = pow(c, n - 1);
		float c_nth_min_1 = 1 - pow(c, n);

		arr_out[0] = c_nth_min_1  * c_min_1 / c_nth_min_1;

		#pragma unroll
		for(int i = 0; i < n; i++)
		{
			c_nth_min_i = c_nth_min_i / c;
			float tmp   = c_nth_min_i  * c_min_1 / c_nth_min_1; 
			arr_out[i]  = arr_out[i - 1] + tmp;
		}

		return arr_out[n - 1];
	}

	template<unsigned int n>
	int rank_selection(
		float* arr_ranks
	)
	{
		float choice = utils::random::rand_bounded_double(0, 1);
		float sum    = 0;

		for(int i = 0; i < n; i++)
		{
			sum += arr_ranks[i];
			if(sum >= choice) return i;
		}
		return n;
	}

	//? ne sum siguren za ova
	template<unsigned int n>
	int rank_selection_optimized_custom()
	{
		//ideata pozadi ova e vo O(1) vreme da se odredi izborot
		//pretpostavuvajki serija od 1, 2, 3, ..., n
		//(in + i^2) / (n^2 + n) >= f

		//ovde selekcijata e O(1), ama sepak problem e shto mora
		//da se sortira
		float choice = utils::random::rand_bounded_double(0, 1);
		static float nf = [](){
			return (float) n;
		}();
		static float nf_2_min_nf = [](){
			return nf * nf - nf;
		}();

		float i = (-1 + sqrt(1 + 4 * choice * nf_2_min_nf)) / 2;
		return (int) ceilf(i);
	}
	

	//? ne sum siguren za ova
	template<
		unsigned int n,
		unsigned int c_num,
		unsigned int c_div
	>
	int rank_selection_exponential_optimized_custom()
	{
		//sumata e (c^(n+1) - 1) / (c - 1)
		//da se reshi za n preku a (choice) pa se dobiva kraen rezultat:
		//x = (log((1 - c) (1/(1 - c) - a)) - log(c))/log(c)
		//x = log(a (c - 1) + 1)/log(c) - 1

		//ovde selekcijata e O(1), ama sepak problem e shto mora
		//da se sortira
		static float c = [](){
			return (float) c_num / (float) c_div;
		}();
		static float sum = [](){
			
			return (pow(c, (n + 1)) - 1) / (c - 1);
		}();
		static float ln_c = [](){
			return log(c);
		}();

		float choice = utils::random::rand_bounded_double(0, 1);
		choice *= (float) sum;

		float i = log(choice*(c - 1) + 1) / ln_c - 1;
		return (int) ceilf(i);
	}


	//? ako se presmeta sumata pred toa gornoto bi trebalo da e pobrzo
	//? deka tuka treba da se presmeta pow ciklichno
	template<
		unsigned int n,
		unsigned int c_num,
		unsigned int c_div
	>
	int rank_selection_exponential_optimized_custom_2()
	{
		//sumata e (c^(n+1) - 1) / (c - 1)
		//se reshava sumata do sega preku celata suma so formula
		//(c^(x + 1) - 1)/(c^(n + 1) - 1) = a
		//resheno za x ova e 
		//x = (log((1 - c^(n + 1)) (1/(1 - c^(n + 1)) - a)) - log(c))/log(c)
		//x = log(a (c^(n + 1) - 1) + 1)/log(c) - 1

		//ovde selekcijata e O(1), ama sepak problem e shto mora
		//da se sortira

		//precompute static vars to be faster
		static float c = []() {
			return (float) c_num / (float) c_div;
		}();
		static float ln_c = []() {
			return log(c);
		}();
		static float pow_c_min_one = []() {
			return pow(c, n + 1) - 1.0f;
		}();

		float choice = utils::random::rand_bounded_double(0, 1);
		float i = log(choice * pow_c_min_one + 1) / ln_c - 1;

		return (int) ceilf(i);
	}

	//ova dodava i povekje krugovi na selekcija
	template<unsigned int num_of_contestants, unsigned int num_of_games>
	int custom_tournament_selection(std::vector<Bee>* bees)
	{
		int max_idx   = 0;
		int max_value = (*bees)[max_idx].value;
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

	template<unsigned int tournament_size>
	int tournament_selection(std::vector<Bee>* bees)
	{
		int max_idx = 0;
		float max_value = (*bees)[max_idx].value;

		int choices[tournament_size];

		#pragma unroll
		for(int j = 0; j < tournament_size; j++)
		{
			choices[j] = utils::random::rand_bounded_int(
				0,
				( *bees).size() - 1
			);
		}

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
	 *                           the type: (array(double), int) -> double
	 * @param lower_bounds[in] The lower bounds for the search space
	 * @param upper_bounds[in] The upper bounds for the search space
	 *
	 * @return void
	 */
	template<
		unsigned int dimensions,
		unsigned int bees_count //TODO: ova kje mora posle da go usoglasam i da go izmeram na dve nachini
	>
	void abc(
		std::vector<Bee>*  bees,
		int                max_generations, 
		int                trials_limit, 
		double             scouts_ratio,
		opt_func           function,
		double             lower_bounds[],
		double             upper_bounds[]
	)
	{
		int scouts_count = (int) (((double) bees_count) * scouts_ratio);
		std::vector<double> roulette(scouts_count);

		//rank selection
		float* rank_arr = (float*) malloc(bees_count * sizeof(float));
		//init_rank_arr_linear<bees_count>(rank_arr, 1.1f);
		//init_rank_arr_exponential<bees_count>(rank_arr, 1.1f);
		//init_rank_arr_custom<bees_count>(rank_arr);
		//init_rank_arr_custom_exponential<bees_count>(rank_arr, 0.5);
		//-----------------------MAIN-LOOP---------------------------//
		for(int i = 0; i < max_generations; i++)
		{
			//-----------EMPLOYED-BEE-LOCAL-OPTIMIZATION---------//
			local_optimization<dimensions>(
				bees,
				bees_count,
				function, 
				lower_bounds,
				upper_bounds
			);
			//---------------------------------------------------//

			//--------ONLOOKER-BEE-GLOBAL-OPTIMIZATION-----------//
			//Vo ovaa implementacija na selekcija mora da se 
			//sortiraat rezultatite
			
			//-------------roulette wheel selection---------------//
			//? se koristi std::partial_sort poradi toa shto nadvor
			//? od opsegot beshe da napravam optimizirana 
			//? implementacija na quicksort ili merge_sort 
			//? specifichno za pcheli (voglavno poradi odlukata za
			//? abstrakcija da bide prikazhano kako struktura i
			//? problemi okolu chuvanje na pochetniot indeks)
			//
			// TODO: 
			// Tuka mora da se pretstavat i alternativi na 
			// ovaa selekcija
			//std::partial_sort( (*bees).begin(),
			//	(*bees).begin() + scouts_count,
			//	(*bees).end(),
			//	BeeCompare
			//);
			
			///TODO: Dodadi i drugi tipovi na selekcija
			////Roulette selection
			////initialize roulette
			//create_roulette_wheel(
			//	&(bees),
			//	&roulette,
			//	scouts_count,
			//	bees_count
			//);
			////do selection
			//for(int idx = scouts_count; idx < bees_count; idx++)
			//{
			//	int spin = spin_roulette(roulette);
			//	if((*bees)[idx].value > (*bees)[spin].value)
			//	{
			//		(*bees)[idx] = (*bees)[spin];
			//		(*bees)[idx].trials = 0;
			//	}
			//	else
			//	{
			//		(*bees)[idx] = (*bees)[spin];
			//		(*bees)[idx].trials++;
			//	}
			//}

			//------------------rank-selection-------------------//
			//std::sort(
			//	(*bees).begin(),
			//	(*bees).end(),
			//	BeeCompare
			//);

			for(int idx = 0; idx < bees_count; idx++)
			{
				//int choice = rank_selection<bees_count>(rank_arr);
				//int choice = rank_selection_optimized_custom<bees_count>();
				//int choice = rank_selection_exponential_optimized_custom<bees_count, 5, 10>();
				//int choice = rank_selection_exponential_optimized_custom_2<bees_count, 5, 10>();
				
				//! po grubi eksperimenti tournament selection
				//! ispagja deka ne e soodvetno poradi toa
				//! shto ne konvergira kako shto treba
				//int choice = tournament_selection<bees_count / 10>(bees);
				int choice = custom_tournament_selection<bees_count / 10, 3>(bees);

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
