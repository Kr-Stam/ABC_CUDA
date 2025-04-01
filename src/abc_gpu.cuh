#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "problems/problems.h"

namespace gpu{

	//template<unsigned int DIMENSIONS>
	void launch_abc(
		double*      coordinates,
		double*      values,
		int          num_of_bees, 
		int          max_generations, 
		int          trials_limit, 
		opt_func     optimization_function,
		double       lower_bounds[],
		double       upper_bounds[],
		int          steps
	);

}

