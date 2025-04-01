/******************************************************************************
 * @file abc_cpu.cpp                                                          *
 * @brief Optimized sequential cpu implementation of the ABC algorihtm        *
 * @details Further optimization is possible although deemed unnecessary as   *
 *          the projects's main goal is the parallelization of this algorihtm *
 * @author Kristijan Stameski                                                 *
 ******************************************************************************/

#include "abc_cpu.h"
#include "utils/utils.hpp"
#include <math.h>
#include <vector>
#include <stdlib.h>

using namespace cpu;
