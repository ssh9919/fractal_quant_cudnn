#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

#m4_include([m4/ax_check_cuda.m4])

AC_PREREQ([2.69])
AC_CONFIG_MACRO_DIR([m4])
AC_INIT([libfractal], [0.2], [kyuyeon.hwang@gmail.com])
AM_INIT_AUTOMAKE([foreign])
AC_CONFIG_HEADERS([config.h])
LT_INIT

: ${CXXFLAGS="-Ofast -march=native  -funroll-loops"} #-flto
: ${NVCCFLAGS="-O3"}
SO_VERSION_INFO="0:2:0"
GPU_ARCH="-gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35"

AC_DEFINE([FRACTAL_CUDA_MULTISTREAM], [], [Enable CUDA stream for concurrent kernel launch])
AC_DEFINE([FRACTAL_PIPELINE], [], [Enable pipelining using CUDA multistream for better GPU utilization])
AC_DEFINE([FRACTAL_USE_OMP], [], [Enable OpenMP to accelerate CPU computation])

# Checks for programs.
AC_PROG_CXX
AC_PROG_LIBTOOL

# Checks for CUDA
AX_CHECK_CUDA

AM_CXXFLAGS="$AM_CXXFLAGS -std=c++11 -Wall -fopenmp"
AM_CXXFLAGS="$AM_CXXFLAGS $CUDA_CFLAGS"
AM_LDFLAGS="$AM_LDFLAGS $CUDA_LDFLAGS"
AM_LDFLAGS="$AM_LDFLAGS -version-info $SO_VERSION_INFO"
NVCCFLAGS="$NVCCFLAGS -m64 -Xcompiler -Wall $GPU_ARCH"

AC_SUBST([AM_CXXFLAGS])
AC_SUBST([AM_LDFLAGS])
AC_SUBST([NVCCFLAGS])

# Checks for libraries.

# Checks for header files.

# Checks for typedefs, structures, and compiler characteristics.
AC_CHECK_HEADER_STDBOOL
AC_C_INLINE
AC_TYPE_SIZE_T

# Checks for library functions.
AC_FUNC_MALLOC
AC_CHECK_FUNCS([mkdir])

AC_CONFIG_FILES([Makefile
                 src/Makefile
                 src/core/Makefile
                 src/util/Makefile])
AC_OUTPUT
