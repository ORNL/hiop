/**
 * @file evloser_gpu_defs.hpp
 *
 * Selects CUDA or HIP GPU backend definitions for EVLOSER.
 *
 */

#ifndef EVLOSER_GPU_DEFS_H
#define EVLOSER_GPU_DEFS_H

#if defined(HIOP_USE_CUDA)

#include "evloser_cusolver_defs.hpp"

#elif defined(HIOP_USE_HIP) || defined(HAVE_HIP)

#include "evloser_hipsolver_defs.hpp"

#else

#error "EVLOSER GPU backend requires either HIOP_USE_CUDA or HIOP_USE_HIP."

#endif

#endif  // EVLOSER_GPU_DEFS_H
