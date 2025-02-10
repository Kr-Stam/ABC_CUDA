#include "problems_many_local_minima.hpp"
#include <math.h>

double problems::ackley(double* args, int n, double a, double b, double c)
{
	double result;
	double sum_x2 = 0;
	double sum_cx = 0;
	for(int i = 0; i < n; i++)
	{
		sum_x2 += args[i]*args[i];
		sum_cx += std::cos(c*args[i]);
	}
	result = -a * std::pow(M_E, -b*std::sqrt(1.0/n*sum_x2))
			 -std::pow(M_E, 1.0/n*sum_cx) + a + M_E;

	return result;
}

double problems::ackley2(double* args, int n)
{
	return ackley(args, n, 20, 0.2, 2*M_PI);
}

///\note this function is only a 2d function
double problems::bukin6(double* args, int n)
{
	if(n < 2) return 0;

	return 100  * std::sqrt(std::abs(args[1]-0.01*args[0]*args[0])) +
		   0.01 * std::abs(args[0] + 10);
}

///\note this function in only a 2d function
double problems::cross_in_tray(double* args, int n)
{
	if(n < 2) return 0;

	return -0.0001 * std::pow(std::abs(std::sin(args[0]) *
		   std::sin(args[1])*std::pow(M_E, std::abs(100 -
		   std::sqrt(args[0]*args[0]+args[1]*args[1])/M_PI))) + 1,
		   0.1);
}

///\note this function in only a 2d function
double problems::drop_wave(double* args, int n)
{
	if(n < 2) return 0;

	double tmp = args[0]*args[0] + args[1]*args[1];

	return (-1 - std::cos(12*std::sqrt(tmp))) /
		   (0.5*tmp + 2);
}

///\note this function in only a 2d function
double problems::eggholder(double* args, int n)
{
	if(n < 2) return 0;

	return -(args[1]+47)*std::sin(std::sqrt(std::abs(args[1]+args[0]/2+47)))
		   -args[0]*std::sin(std::sqrt(std::abs(args[0]-args[1]-47)));
}

///\note this function is only a 1d function
double problems::gramacy_and_lee(double* args, int n)
{
	if(n < 1) return 0;

	double tmp = args[0] - 1;
	return std::sin(10*M_PI*args[0])/2*args[0]+
		   tmp * tmp * tmp * tmp;
}

double problems::griewank(double* args, int n)
{
	double result  = 0;
	double product = 1;
	for(int i = 0; i < n; i++)
	{
		result += args[i]*args[i];
		product *= std::cos(args[i]/std::sqrt(i+1));
	}
	result /= 4000.0;
	result += product + 1.0;
	return result;
}

///\note This only a 2d function
double problems::holder_table(double* args, int n)
{
	if(n < 2) return 0;

	return -std::abs(std::sin(args[0])*std::cos(args[1])*
			std::pow(M_E, std::abs(1.0-std::sqrt(args[0]*args[0]+args[1]*args[1])/M_PI)));
}

///\note A has dimensions nxm
double problems::langerman(double* args, int n, double* c, int m, double* A)
{
	double result = 0;
	for(int i = 0; i < m; i++)
	{
		double sum1 = 0;
		for(int j = 0; j < n; j++)
		{
			double tmp = args[j] - A[i*n+j];
			sum1 += tmp * tmp;
		}
		result += c[i] * std::pow(M_E, -1.0/M_PI*sum1) *
				  std::cos(M_PI*sum1);
	}

	return result;
}

double problems::langerman2(double* args, int n)
{
	double A[] = {
		3, 5,
		5, 2,
		2, 1,
		1, 4,
		7, 9
	};
	double c[] = {1, 2, 5, 2, 3};

	return langerman(args, n, c, 5, A);
}

double problems::levy(double* args, int n)
{
	double tmp, tmp1;
	double result = 0;

	double w = 1.0 + (args[0]-1.0)/4.0;
	tmp = std::sin(M_PI*w);
	result += tmp*tmp;

	for(int i = 0; i < n - 1; i++)
	{
		w = 1.0 + (args[i]-1.0)/4.0;
		tmp = w - 1;
		tmp1 = std::sin(M_PI*w+1);
		result += tmp*tmp*(1.0+10.0*tmp1*tmp1);
	}

	w = 1.0 + (args[n-1]-1.0)/4.0;
	tmp  = w - 1;
	tmp1 = std::sin(2*M_PI*w);
	result += tmp*tmp*(1+tmp1*tmp1);

	return result;
}

///\note This is only a 2d function
double problems::levy13(double* args, int n)
{
	if(n < 2) return 0;

	double tmp1 = std::sin(3*M_PI*args[0]);
	double tmp2 = args[0] - 1.0;
	double tmp3 = std::sin(3*M_PI*args[1]);
	double tmp4 = args[1] - 1.0;
	double tmp5 = std::sin(2*M_PI*args[1]);

	return tmp1*tmp1 + tmp2*tmp2*(1 + tmp3*tmp3) +
		   tmp4*tmp4*(1 + tmp5*tmp5);
}

///\brief This is a fucntion with mutiple global
///       maxima which is commonly used to test
///       optimization functions
double problems::rastrigin(int a, double* x, int n)
{
	float result = a * n;
	for(int i = 0; i < n; i++)
	{
		result += x[i]*x[i] - a*std::cos(2*M_PI*x[i]);
	}
	return result;
}

///brief This function is a wrapper around rastrigin
///      in order to make it compatible to the problem
///      definition
double problems::rastrigin2(double* args, int n)
{
	return rastrigin(10, args, n);
}

///\note This is only a 2d function
double problems::schaffer2(double* args, int n)
{
	if(n < 2) return 0;

	double x2 = args[0]*args[0];
	double y2 = args[1]*args[1];
	double tmp1 = std::sin(x2 - y2);
	double tmp2 = 1.0 + 0.001*(x2+y2);
	return 0.5 + (tmp1*tmp1 - 0.5)/tmp2/tmp2;
}

///\note This is only a 2d function
double problems::schaffer4(double* args, int n)
{
	if(n < 2) return 0;

	double x2 = args[0]*args[0];
	double y2 = args[1]*args[1];
	double tmp1 = std::cos(std::sin(std::abs(x2 - y2)));
	double tmp2 = 1.0 + 0.001*(x2+y2);
	return 0.5 + (tmp1*tmp1 - 0.5)/tmp2/tmp2;
}

double problems::schwefel(double* args, int n)
{
	double result = 418.9829*n;
	for(int i = 0; i < n; i++)
	{
		result -= args[i]*std::sin(std::sqrt(std::abs(args[i])));
	}

	return result;
}

///\note This is only a 2d function
double problems::shubert(double* args, int n)
{
	if(n < 2) return 0;
	double sum1 = 0;
	double sum2 = 0;
	for(int i = 0; i < 5; i++)
	{
		sum1 += i*std::cos(args[0]*(i+1)+i);
		sum2 += i*std::cos(args[1]*(i+1)+i);
	}
	return sum1 * sum2;
}
