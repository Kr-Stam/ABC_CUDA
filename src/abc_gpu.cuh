#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "problems/problems.h"
#include "abc_main.cuh"

namespace gpu{

	//TODO: ova treba da go stavam vo headerot
	//template<unsigned int DIMENSIONS>
	void launch_abc(
		float*       coordinates,
		float*       values,
		int          num_of_bees, 
		int          max_generations, 
		int          trials_limit, 
		opt_func     optimization_function,
		float        lower_bounds[],
		float        upper_bounds[],
		int          steps
	);

}
