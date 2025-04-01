#include "problems_other.h"
#include <math.h>

double problems::cpu::beale(double* args, int n)
{
	if(n < 2) return 0;

	double tmp1 = args[0]*args[1];
	double tmp2 = 1.5   - args[0] + tmp1;
	double tmp3 = 2.25  - args[0] + tmp1*args[1];
	double tmp4 = 2.625 - args[0] + tmp1*args[1]*args[1];

	return tmp2*tmp2 + tmp3*tmp3 + tmp4*tmp4;
}

double problems::cpu::branin(double* args, int n, double a, double b, double c, double r, double s, double t)
{
	if(n < 2) return 0;

	double tmp = args[1] - b*args[0]*args[0] + c*args[0] - r;

	return a*tmp*tmp + s*(1-t)*std::cos(args[0]) + s;
}

double problems::cpu::branin2(double* args, int n)
{
	return problems::cpu::branin(args, n, 1, 5.1/4.0/M_PI/M_PI, 5/M_PI, 6, 10, 1.0/8.0/M_PI);
}

double problems::cpu::colville(double* args, int n)
{
	if(n < 4) return 0;
	double tmp1 = args[0]*args[0] - args[1];
	double tmp2 = args[0] - 1;
	double tmp3 = args[2] - 1;
	double tmp4 = args[2]*args[2] - args[3];
	double tmp5 = args[1] - 1;
	double tmp6 = args[3] - 1;
	return 100*tmp1*tmp1 + tmp2*tmp2 + tmp3*tmp3 + 90*tmp4*tmp4 + 
	       10.1*(tmp5*tmp5 + tmp6*tmp6) +
		   19.8*tmp5*tmp6;
}

double problems::cpu::forrester(double* args, int n)
{
	if(n < 1) return 0;
	double tmp = 6*args[0] - 2;
	return tmp*tmp*std::sin(12*args[0] - 4);
}

double problems::cpu::goldstein_price(double* args, int n)
{
	double tmp1 = args[0] + args[1] + 1;
	double x2 = args[0]*args[0];
	double y2 = args[1]*args[1];
	double tmp2 = 2*args[0] - 3*args[1];

	return tmp1*tmp1*(19 - 14*args[0] + 3*x2 - 14*args[1] +
	       6*args[0]*args[1] + 3*y2)*(30 + tmp2*tmp2*(18 -
	       32*args[0] + 12*x2 + 48*args[1] - 36*args[0]*args[1] +
		   27*y2));
}

const double h3_alpha[] = {1.0, 1.2, 3.0, 3.2};
const double h3_a[] = {
	3.0, 10, 30,
	0.1, 10, 35,
	3.0, 10, 30,
	0.1, 10, 35
};
const double h3_p[] = {
	0.3689, 0.1170, 0.2673,
	0.4699, 0.4387, 0.7470,
	0.1091, 0.8732, 0.5547,
	0.0381, 0.5743, 0.8828
};

double problems::cpu::hartmann3d (double* args, int n)
{
	if(n < 3) return 0;

	double outer = 0;
	for(int i = 0; i < 4; i++)
	{
		double inner = 0;
		for(int j = 0; j < 3; j++)
		{
			double tmp = args[j] - h3_p[i*3+j];
			inner -= h3_a[i*3+j]*tmp*tmp;
		}
		outer -= h3_alpha[i]*std::pow(M_E, inner);
	}
	return outer;
}

const double h4_alpha[] = {1.0, 1.2, 3.0, 3.2};
const double h4_a[] = {
	  10,  3,   17,  3.5, 1.7,  8,
	0.05, 10,   17,  0.1,   8, 14,
	   3, 3.5,  1.7,  10,  17,  8,
	  17,   8, 0.05,  10, 0.1, 14 
};
const double h4_p[] = {
	0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886,
	0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991,
	0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650,
	0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381
};

double problems::cpu::hartmann4d (double* args, int n)
{
	if(n < 4) return 0;

	double outer = 1.1;
	for(int i = 0; i < 4; i++)
	{
		double inner = 0;
		for(int j = 0; j < 4; j++)
		{
			double tmp = args[j] - h4_p[i*6+j];
			inner -= h4_a[i*6+j]*tmp*tmp;
		}
		outer -= h4_alpha[i]*std::pow(M_E, inner);
	}
	return 1.0/0.839*outer;
}

const double h6_alpha[] = {1.0, 1.2, 3.0, 3.2};
const double h6_a[] = {
	  10,  3,   17,  3.5, 1.7,  8,
	0.05, 10,   17,  0.1,   8, 14,
	   3, 3.5,  1.7,  10,  17,  8,
	  17,   8, 0.05,  10, 0.1, 14 
};
const double h6_p[] = {
	0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886,
	0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991,
	0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650,
	0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381
};

double problems::cpu::hartmann6d (double* args, int n)
{
	if(n < 6) return 0;

	double outer = 0;
	for(int i = 0; i < 4; i++)
	{
		double inner = 0;
		for(int j = 0; j < 6; j++)
		{
			double tmp = args[j] - h6_p[i*6+j];
			inner -= h6_a[i*6+j]*tmp*tmp;
		}
		outer -= h6_alpha[i]*std::pow(M_E, inner);
	}
	return outer;
}

double problems::cpu::permdb(double* args, int n, double b)
{
	double outer = 0;
	for(int i = 0; i < n; i++)
	{
		double inner = 0;
		for(int j = 0; j < n; j++)
		{
			//? ne znam dali porgreshno beshe opishano?
			//inner += ((std::pow(j+1, i+1)+b)*(std::pow(args[i]/(j+1), i+1) - 1));
			inner += ((std::pow(j+1, i+1)+b)*(std::pow(args[j]/(j+1), i+1) - 1));
		}
		outer += inner*inner;
	}
	return outer;
}

double problems::cpu::permdb2(double* args, int n)
{
	return problems::cpu::permdb(args, n, 0.5);
}

double problems::cpu::powell(double* args, int n)
{
	if(n < 4) return 0;

	double result = 0;
	for(int i = 0; i < n / 4; i++)
	{
		double tmp1 = args[4*i - 3] + 10*args[4*i - 2];
		double tmp2 = args[4*i - 1] - args[4*i];
		double tmp3 = args[4*i - 2] - 2*args[4*i - 1];
		double tmp4 = args[4*i - 3] - args[4*i];

		result += tmp1*tmp1 + 5*tmp2*tmp2 + tmp3*tmp3*tmp3*tmp3 +
		          10*tmp4*tmp4*tmp4*tmp4;
	}
	return result;
}

double problems::cpu::shekel(double* args, int n, int m, const double* beta, const double* C)
{
	double outer = 0;
	for(int i = 0; i < m; i++)
	{
		double inner = 0;
		for(int j = 0; j < 4; j++)
		{
			double tmp = args[j] - C[j*m+i];
			inner += tmp*tmp + beta[i];
		}
		outer -= 1.0/inner;
	}
	return outer;
}

const double sh_beta[] = {
	0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5
};

const double sh_c[] = {
	4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0,
	4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6,
	4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0,
	4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6
};

double problems::cpu::shekel2(double* args, int n)
{
	return problems::cpu::shekel(args, n, 10, sh_beta, sh_c);
}

double problems::cpu::styblinsky_tang(double* args, int n)
{
	double result = 0;
	for(int i = 0; i < n; i++)
	{
		double tmp = args[i]*args[i];
		result += tmp*tmp - 16*tmp + 5*args[i];
	}
	return 0.5*result;
}
