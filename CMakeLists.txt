cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

if (MSVC)
  set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin/nvcc.exe")
  set(CMAKE_CXX_COMPILER "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.34.31933/bin/Hostx64/x64/cl.exe")
endif()

if (UNIX)
	message(STATUS "Project is building for Linux")
	set(CMAKE_CUDA_COMPILER /opt/cuda/bin/nvcc)
	set(CMAKE_CXX_STANDARD 17)
	set(CMAKE_CUDA_STANDARD 17)
	set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

	#turn on linking to matplotplusplus
	#message(STATUS "Linking to matplotplusplus opencv")
	#add_subdirectory(matplotplusplus)
	#find_package(Matplot++ REQUIRED PATHS ~/aur/matplotplusplus/src/build/)

endif()

if(WIN32)
    message(STATUS "Project is building for Windows")
    set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3")
    set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/bin/nvcc.exe")
	set(CMAKE_CXX_STANDARD 17)
	set(CMAKE_CUDA_STANDARD 17)
	set(DCMAKE_CUDA_ARCHITECTURES native)

	#message(STATUS "Linking to matplotplusplus opencv")
	#add_subdirectory(matplotplusplus)
	#find_package(Matplot++ REQUIRED)
endif()


if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
	message(STATUS "Building in Debug mode")
endif()

if(MNIST)
	project(MNIST LANGUAGES CXX CUDA)
elseif(TEST_ALL)
	project(Test_All LANGUAGES CXX CUDA)
else()
	project(CUDA_Artificial_Bee_Colony LANGUAGES CXX CUDA)
endif()

message("Project name: " ${CMAKE_PROJECT_NAME})


if (MSVC)
	message(STATUS "Building for Windows with MSVC")
	set(_CRT_SECURE_NO_WARNINGS 1)
	set(CURL_STATICLIB 1)
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")
endif()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

file(GLOB SOURCE_FILES_CPP "src/*.cpp")
file(GLOB SOURCE_FILES_CU  "src/*.cu")
file(GLOB HEADER_FILES_HPP "src/*.hpp")
file(GLOB HEADER_FILES_CUH "src/*.cuh")

set(CMAKE_BUILD_TYPE Debug)

add_executable(
	${CMAKE_PROJECT_NAME} 
	#${SOURCE_FILES_CPP} ${HEADER_FILES_HPP} 
	#${SOURCE_FILES_CU}  ${HEADER_FILES_CUH}
	"${CMAKE_CURRENT_SOURCE_DIR}/src/utils/array.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/utils/sort.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/utils/random.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/utils/random.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/utils/utils.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/problems/problems.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/problems/gpu/many_local_minima.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/problems/gpu/bowl_shaped.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/problems/gpu/plate_shaped.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/problems/gpu/valley_shaped.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/problems/gpu/steep_ridges.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/problems/gpu/other.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/problems/cpu/many_local_minima.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/problems/cpu/bowl_shaped.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/problems/cpu/plate_shaped.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/problems/cpu/valley_shaped.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/problems/cpu/steep_ridges.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/problems/cpu/other.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/abc_main.cuh"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/abc_cpu.cuh"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/abc_gpu.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cuh"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/main.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/rank_array.cuh"
	)

# turn off matplotplusplus
#target_link_libraries( 
#	"${CMAKE_PROJECT_NAME}"
#	#PUBLIC matplot
#	Matplot++::matplot
#)

target_compile_options(${CMAKE_PROJECT_NAME}
	PRIVATE $<$<COMPILE_LANGUAGE:CXX>: >
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --ptxas-options=-v -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored --expt-relaxed-constexpr>
	# -g -G --ptxas-options=–allow-expensive-optimizations=true>
)

set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 61 75 87 80) # Running on GTX 1080 Ti, GTX 1660 Ti, and Jetson AGX Orin
set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 80)

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE _DEBUG)
    target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
