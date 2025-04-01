/******************************************************************************
 * @file graphs.cpp                                                           *
 * @brief Functions for visualizing and visually exploring the problem space  *
 *                                                                            *
 * @author Kristijan Stameski                                                 *
 ******************************************************************************/

#include "matplot.h"

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

void matplot_graph()
{
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
}
