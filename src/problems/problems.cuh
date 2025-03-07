#pragma once

#include "problems_many_local_minima.cuh"
#include "problems_bowl_shaped.cuh"
#include "problems_plate_shaped.cuh"
#include "problems_valley_shaped.cuh"
#include "problems_steep_ridges.cuh"
#include "problems_other.cuh"

typedef struct OptimizationProblem
{
	int     n;
	double* upper_bound;
	double* lower_bound;
	double(*function)(double*, int);
} OptimizationProblem;
//? ako imam vreme za podolgo refaktoriranje ova kje go napravam klasa

//site osven 4d+ funkciite se testirani da se tochni
//OptimizationProblem p1  = {2, problems::rastrigin2             };
//OptimizationProblem p2  = {2, problems::sphere                 };
//OptimizationProblem p3  = {2, problems::rosenbrock             };
//OptimizationProblem p4  = {2, problems::ackley2                };
//OptimizationProblem p5  = {2, problems::bukin6                 };
//OptimizationProblem p6  = {2, problems::cross_in_tray          };
//OptimizationProblem p7  = {2, problems::drop_wave              };
//OptimizationProblem p8  = {2, problems::eggholder              };
//OptimizationProblem p9  = {1, problems::gramacy_and_lee        };
//OptimizationProblem p10 = {2, problems::griewank               };
//OptimizationProblem p11 = {2, problems::holder_table           };
//OptimizationProblem p12 = {2, problems::langerman2             };
//OptimizationProblem p13 = {2, problems::levy                   };
//OptimizationProblem p14 = {2, problems::levy13                 };
//OptimizationProblem p15 = {2, problems::schaffer2              };
//OptimizationProblem p16 = {2, problems::schaffer4              };
//OptimizationProblem p17 = {2, problems::schwefel               };
//OptimizationProblem p18 = {2, problems::shubert                };
//OptimizationProblem p19 = {2, problems::bohachevsky1           };
//OptimizationProblem p20 = {2, problems::bohachevsky2           };
//OptimizationProblem p21 = {2, problems::bohachevsky3           };
//OptimizationProblem p22 = {2, problems::perm2                  };
//OptimizationProblem p23 = {2, problems::rotated_hyper_elipsoid };
//OptimizationProblem p24 = {2, problems::sum_of_different_powers};
//OptimizationProblem p25 = {2, problems::sum_squares            };
//OptimizationProblem p26 = {2, problems::trid                   };
//OptimizationProblem p27 = {2, problems::booth                  };
//OptimizationProblem p28 = {2, problems::matyas                 };
//OptimizationProblem p29 = {2, problems::mccormick              };
//OptimizationProblem p30 = {2, problems::power_sum2             };
//OptimizationProblem p31 = {2, problems::zakharov               };
//OptimizationProblem p32 = {2, problems::three_hump_camel       };
//OptimizationProblem p33 = {2, problems::six_hump_camel         };
//OptimizationProblem p34 = {2, problems::dixon_price            };
//OptimizationProblem p35 = {2, problems::dejong5                };
//OptimizationProblem p36 = {2, problems::easom                  };
//OptimizationProblem p37 = {2, problems::michalewicz2           };
//OptimizationProblem p38 = {2, problems::michalewicz2           };
//OptimizationProblem p39 = {2, problems::beale                  };
//OptimizationProblem p40 = {4, problems::colville               }; //untested
//OptimizationProblem p41 = {1, problems::forrester              };
//OptimizationProblem p42 = {2, problems::goldstein_price        };
//OptimizationProblem p43 = {3, problems::hartmann3d             }; //untested
//OptimizationProblem p44 = {4, problems::hartmann4d             }; //untested
//OptimizationProblem p45 = {6, problems::hartmann6d             }; //untested
//OptimizationProblem p46 = {2, problems::permdb2                };
//OptimizationProblem p47 = {4, problems::powell                 }; //untested
//OptimizationProblem p48 = {4, problems::shekel2                }; //untested
//OptimizationProblem p49 = {4, problems::styblinsky_tang        };
