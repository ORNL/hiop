/**
 * @file evloser_gpu_defs.hpp
 *
 * Selects CUDA, HIP or CPU backend definitions for EVLOSER.
 *
 */

#ifndef EVLOSER_GPU_DEFS_H
#define EVLOSER_GPU_DEFS_H

#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA)

#include "evloser_cusolver_defs.hpp"

#elif defined(HIOP_USE_HIP) || defined(HAVE_HIP)

#include "evloser_hipsolver_defs.hpp"

#else

#include "evloser_cpu_defs.hpp"

#endif

#endif  // EVLOSER_GPU_DEFS_H
