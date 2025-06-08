#include <array>
#include <cstdint>
#include <cmath>
#include <stdio.h>
#include "utils/utils.hpp"
//za da go resham ova treba se da kompajliram so CUDA
#include <cuda_runtime.h>


namespace rank_arr
{
	/*
	 * @brief Returns an array of weights for linear rank selection
	 *
	 * @tparam[in]  n     Number of candidates
	 * @tparam[in]  a_num Numerator of a
	 * @tparam[in]  a_div Dividend of a
	 *
	 * @return A constexpr, compile time initialized array of weights
	 *
	 * @note a_num/a_div must be less than 2
	 * @note According to cited reasearch the ideal value of a is 1.1
	 */
	template<
		uint32_t n,
		int64_t  a_num,
		int64_t  a_div
	>
	__device__ __host__ constexpr std::array<float, n> arr_lin()
	{
		std::array<float, n> arr{};

		float a = (float) a_num / (float) a_div;

		if(a > 2 || n < 1) return arr;

		float b = 2 - a;

		float c = 1.0 / n * (b + (a - b));
		arr[0] = c;

		for(uint32_t i = 1; i < n; i++)
		{
			c = 1.0 / n * (b + (a - b) * (n - i - 1.0) / (n - 1.0));
			arr[i] = arr[i - 1] + c;
		}

		return arr;
	}

	//bazirano na:
	//An Analysis of Linear Ranking and Binary Tournament Selection in
	//Genetic Algorithms shto e bazirano na ova:
	//Adaptive Selection Methods for Genetic Algorithms 
	//James Edward Baker 
	//Computer Science Department 
	//Vanderbilt University 
	//according to the paper the most ideal value of max is 1.1
	/*
	 * @brief Initializes an array of weights for linear rank selection
	 *
	 * @param[out] arr Array of at least size n
	 * @param[in]  a   Must be less than 2.0
	 * @param[in]  n   Number of candidates
	 *
	 * @return The sum of all elements of the initialized array
	 *
	 * @note According to cited reasearch the ideal value of a is 1.1
	 */
	template<uint32_t n>
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

	/*
	 * @brief Returns an array of weights for exponential rank selection
	 *
	 * @tparam[in]  n     Number of candidates
	 * @tparam[in]  c_num Numerator of c
	 * @tparam[in]  c_div Dividend of c
	 *
	 * @return A constexpr, compile time initialized array of weights
	 */
	template<
		uint32_t n,
		int64_t c_num,
		int64_t c_div
	>
	__device__ __host__ constexpr std::array<float, n> arr_exp()
	{
		float c = (float) c_num / (float) c_div;

		std::array<float, n> arr = {};
		if(n < 1) return arr;

		float c_min_1     = 1 - c;
		float c_nth_min_i = pow(c, n - 1);
		float c_nth_min_1 = 1 - pow(c, n);

		arr[0] = c_nth_min_1  * c_min_1 / c_nth_min_1;

		for(int i = 0; i < n; i++)
		{
			c_nth_min_i = c_nth_min_i / c;
			float tmp   = c_nth_min_i  * c_min_1 / c_nth_min_1; 
			arr[i]      = arr[i - 1] + tmp;
		}

		return arr;
	}

	/*
	 * @brief Returns an array of weights for exponential rank selection
	 *
	 * @tparam[in] n       Number of candidates
	 * @param[out] arr_out Array to be initialized 
	 * @param[in]  c       Base of the exponent
	 *
	 * @return A constexpr, compile time initialized array of weights
	 */
	template<uint32_t n>
	float init_arr_exp(
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

	/*
	 * @brief Initializes an array of weights for rank selection
	 * @param[out] arr Array of at least size n
	 * @param[in]  n   Number of candidates
	 *
	 * @return A constexpr, compile time initialized array of weights
	 *
	 * @details
	 * Even though this function initializes weights by rank,
	 * it does not use the standard rank selection algorithm.
	 * It uses the sequence n, n - 1, ..., 1
	 */
	template<uint32_t n>
	__device__ __host__ constexpr std::array<float, n> arr_simple()
	{
		int sum = n * (n + 1) / 2;

		std::array<float, n> arr = {};

		arr[0] = (float) n / (float) sum;

		for(int i = 1; i < n; i++) 
			arr[i] = (float) (2*n - 2*i - 1) / (float) sum;

		return arr;
	}

	/*
	 * @brief Initializes an array of weights for rank selection
	 * @param[out] arr Array of at least size n
	 * @param[in]  n   Number of candidates
	 *
	 * @return The sum of all elements of the initialized array
	 *
	 * @details
	 * Even though this function initializes weights by rank,
	 * it does not use the standard rank selection algorithm.
	 * It uses the sequence n, n - 1, ..., 1
	 */
	template<uint32_t n>
	float init_arr_simple(float* arr)
	{
		int sum = n * (n + 1) / 2;

		arr[0] = (float) n / (float) sum;

		#pragma unroll
		for(int i = 1; i < n; i++) 
			arr[i] = (float) (2*n - 2*i - 1) / (float) sum;

		return (float) sum;
	}

	/*
	 * @brief Initializes an array of weights for rank selection
	 *
	 * @param[in]   n     Number of candidates
	 * @tparam[in]  c_num Numerator of c
	 * @tparam[in]  c_div Dividend of c
	 *
	 * @note c_num/d_div Base of the exponent, 0 < c < 1x
	 *
	 * @return A constexpr, compile time initialized array of weights
	 *
	 * @details
	 * Even though this function initializes weights by rank,
	 * it does not use a standard rank selection algorithm.
	 * Instead it uses the sequence c^0 + c^1 + ... c^n
	 */
	template<
		uint32_t n,
		int64_t  c_num,
		int64_t  c_div
	>
	__device__ __host__ constexpr std::array<float, n> arr_simple_exp()
	{
		float c = (float) c_num / (float) c_div;
		//the sum is (c^(n+1) - 1) / (c - 1)
		float sum = (pow(c, n + 1) - 1) / (c - 1);

		std::array<float, n> arr = {};
		arr[0] = 1 / sum;

		for(int i = 1; i < n; i++) 
			arr[i] = arr[i - 1] + pow(c, i) / sum;

		return arr;
	}

	/*
	 * @brief Initializes an array of weights for rank selection
	 * @param[out] arr Array of at least size n
	 * @param[in]  n   Number of candidates
	 * @param[in]  c   Base of the exponent, 0 < c < 1
	 *
	 * @return The sum of all elements of the initialized array
	 *
	 * @details
	 * Even though this function initializes weights by rank,
	 * it does not use a standard rank selection algorithm.
	 * Instead it uses the sequence c^0 + c^1 + ... c^n
	 */
	template<uint32_t n>
	float init_arr_simple_exp(float* arr, float c)
	{
		//the sum is (c^(n+1) - 1) / (c - 1)
		float sum = (pow(c, n + 1) - 1) / (c - 1);

		arr[0] = 1 / sum;

		#pragma unroll
		for(int i = 1; i < n; i++) 
			arr[i] = arr[i - 1] + pow(c, i) / sum;

		return (float) sum;
	}
}

///returns the index in constant time
namespace rank_const
{
	//? ne sum siguren za ova
	template<uint32_t n>
	__device__ __host__ inline int lin(float choice)
	{
		//ideata pozadi ova e vo O(1) vreme da se odredi izborot
		//pretpostavuvajki serija od 1, 2, 3, ..., n
		//(in + i^2) / (n^2 + n) >= f

		//ovde selekcijata e O(1), ama sepak problem e shto mora
		//da se sortira
		static float nf_2_min_nf = [](){
			return (float) n * (float ) n - (float) n;
		}();

		float i = (-1 + sqrt(1 + 4 * choice * nf_2_min_nf)) / 2;
		return (int) ceilf(i);
	}

	//! ne bi bilo losho da go gi uskladam dvete funkcii
	template<uint32_t n>
	__device__ inline int dev_lin(float choice)
	{
		//ideata pozadi ova e vo O(1) vreme da se odredi izborot
		//pretpostavuvajki serija od 1, 2, 3, ..., n
		//(in + i^2) / (n^2 + n) >= f

		float nf_2_min_nf = (float) n * (float ) n - (float) n;

		float i = (-1 + sqrt(1 + 4 * choice * nf_2_min_nf)) / 2;
		return (int) ceilf(i);
	}
	
	//? ne sum siguren za ova
	template<
		uint32_t n,
		uint32_t c_num,
		uint32_t c_div
	>
	__host__ int exp(float choice)
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

		choice *= (float) sum;

		float i = log(choice*(c - 1) + 1) / ln_c - 1;

		return (int) ceilf(i);
	}

	template<
		uint32_t n,
		uint32_t c_num,
		uint32_t c_div
	>
	__device__ int dev_exp(float choice)
	{
		//sumata e (c^(n+1) - 1) / (c - 1)
		//da se reshi za n preku a (choice) pa se dobiva kraen rezultat:
		//x = (log((1 - c) (1/(1 - c) - a)) - log(c))/log(c)
		//x = log(a (c - 1) + 1)/log(c) - 1

		float c    = (float) c_num / (float) c_div;
		float sum  = (pow(c, (n + 1)) - 1) / (c - 1);
		float ln_c = log(c);

		choice *= (float) sum;

		float i = log(choice*(c - 1) + 1) / ln_c - 1;

		return (int) ceilf(i);
	}

	//! ne mi e jasno shto sum mislel so ova
	//? ako se presmeta sumata pred toa gornoto bi trebalo da e pobrzo
	//? deka tuka treba da se presmeta pow ciklichno
	template<
		uint32_t n,
		uint32_t c_num,
		uint32_t c_div
	>
	__host__ int exp2(float choice)
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

		float i = log(choice * pow_c_min_one + 1) / ln_c - 1;

		return (int) ceilf(i);
	}
	
	//! ne mi e jasno shto sum mislel so ova
	//? ako se presmeta sumata pred toa gornoto bi trebalo da e pobrzo
	//? deka tuka treba da se presmeta pow ciklichno
	template<
		uint32_t n,
		uint32_t c_num,
		uint32_t c_div
	>
	__device__ int dev_exp2(float choice)
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
		float c = (float) c_num / (float) c_div;
		float ln_c = log(c);
		float pow_c_min_one = pow(c, n + 1) - 1.0f;

		float i = log(choice * pow_c_min_one + 1) / ln_c - 1;

		return (int) ceilf(i);
	}
}
