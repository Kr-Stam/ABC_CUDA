#include <matplot/matplot.h>
#include <stdio.h>
#include <math.h>
#include "abc_cpu.cuh"
#include "problems/problems.cuh"
#include "utils/array.hpp"

/*
   Rastrigin function
"$GPVAL_LAST_MULTIPLOT" line 51: Cannot read from '-' during multiplot playback
   f(x) = An + sum over an interval of n values
*/

//kje mora da gi izvadam kako vrednosti posle

void generate_linspace(std::vector<double> v, double start, double end, int n)
{
	double step = (end - start) / n;
	for(int i = 0; i < n; i++)
	{
		double tmp;
		if(i == 0) tmp  = start;
		else       tmp += step;

		v.push_back(tmp);
	}
}

int main()
{
	double upper_bound[] = {10, 10};
	double lower_bound[] = {-10, -10};
	OptimizationProblem problem  = {
		2, 
		lower_bound,
		upper_bound,
		problems::cross_in_tray 
	};

	//double start_x =  -10;
	//double end_x   = +10;
	//double start_y =  -10;
	//double end_y   = +10;
    //auto [X, Y] = matplot::meshgrid(
	//		matplot::linspace(start_x, end_x, 100), 
	//		matplot::linspace(start_y, end_y, 100)
	//);
    //auto Z = matplot::transform(X, Y, [&problem](double x, double y) {
	//	double point[] = {x, y};
    //    return problem.function(point, problem.n);
    //});
	//matplot::surf(X, Y, Z);

	//matplot::show();

	int num_of_bees = 1000;
	std::vector<cpu::Bee> bees(num_of_bees);

	cpu::init_bees(
		&bees,
		num_of_bees,
		problem,
		lower_bound,
		upper_bound
	);
	printf("Start\n");

	int    iterations  = 1000;
	int    max_trials  = 10;
	double scout_ratio = 0.2;

	int num_of_steps = 10;
	int step = iterations / num_of_steps;
	for(int i = 0; i < num_of_steps; i++)
	{
		cpu::abc(
			&bees,
			num_of_bees,
			step,
			max_trials,
			scout_ratio,
			problem,
			lower_bound,
			upper_bound
		);

		cpu::Bee min_bee = cpu::min_bee(&bees, num_of_bees);
		printf(
			"Iteration #%d: x: %.2f y: %.2f value: %.2f\n", 
			i*step,
			min_bee.coordinates[0],
			min_bee.coordinates[1],
			min_bee.value
		);
	}

	printf("Done\n");
	std::vector<double> coordinates_x(num_of_bees);
	std::vector<double> coordinates_y(num_of_bees);
	//std::vector<double> values(num_of_bees);
	double values[num_of_bees];
	for(int bee_idx = 0; bee_idx < num_of_bees; bee_idx++)
	{
		coordinates_x[bee_idx] = bees[bee_idx].coordinates[0];
		coordinates_y[bee_idx] = bees[bee_idx].coordinates[1];
		values[bee_idx] = bees[bee_idx].value;
	}
	utils::array::print_array_double(values, 10);
	for(int bee_idx = 0; bee_idx < 10; bee_idx++)
	{
		printf("x: %.2f y: %.2f value: %.2f\n", 
				bees[bee_idx].coordinates[0],
				bees[bee_idx].coordinates[1],
				bees[bee_idx].value);
	}

	//matplot::scatter3(coordinates_x, coordinates_y, values);

	//matplot::show();

	/*
    std::vector<double> t  = matplot::iota(0, M_PI / 50, 10 * M_PI);
    std::vector<double> st = matplot::transform(t, [](auto x) { return sin(x); });
    std::vector<double> ct = matplot::transform(t, [](auto x) { return cos(x); });
    auto l = matplot::plot3(st, ct, t);
	matplot::show();
	*/
    //std::vector<double> xt = matplot::iota(start, step, end);
    //std::vector<double> yt = matplot::iota(start, step, end);
    //std::vector<double> zt = matplot::transform(t, [](auto x) { return cos(x); });
    //auto l = matplot::plot3(st, ct, t);
	//matplot::show();

	return 0;
}
