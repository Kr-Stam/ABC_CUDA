#include <cuda_runtime.h>
#include <curand_kernel.h>

typedef double(*opt_func)(double*, int);

namespace gpu{

	template<unsigned int DIMENSIONS>
	void launch_abc(
		double*      coordinates,
		double*      values,
		int*         trials,
		int          num_of_bees, 
		int          num_of_scouts, 
		int          max_generations, 
		int          trials_limit, 
		double       ratio_of_scouts,
		opt_func     optimization_function,
		double       lower_bounds[],
		double       upper_bounds[]
	);

}

