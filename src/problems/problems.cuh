#pragma once

#include "cpu/many_local_minima.h"
#include "cpu/bowl_shaped.h"
#include "cpu/plate_shaped.h"
#include "cpu/valley_shaped.h"
#include "cpu/steep_ridges.h"
#include "cpu/other.h"
//#include "gpu/many_local_minima.cuh"
//#include "gpu/bowl_shaped.cuh"
//#include "gpu/plate_shaped.cuh"
//#include "gpu/valley_shaped.cuh"
//#include "gpu/steep_ridges.cuh"
//#include "gpu/other.cuh"

typedef float(*opt_func)(float*, int);

namespace problems::gpu {}
namespace problems::cpu {}
