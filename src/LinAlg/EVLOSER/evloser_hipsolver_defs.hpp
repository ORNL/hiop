/**
 * @file evloser_hipsolver_defs.hpp
 *
 * Defines HIP GPU backend wrappers used by EVLOSER.
 *
 */

#ifndef EVLOSER_HIPSOLVER_DEFS_H
#define EVLOSER_HIPSOLVER_DEFS_H

#include <cstddef>

#include <hip/hip_runtime.h>

using evloserGpuError_t = hipError_t;
using evloserGpuMemcpyKind_t = hipMemcpyKind;

static const evloserGpuError_t evloserGpuSuccess = hipSuccess;
static const evloserGpuMemcpyKind_t evloserMemcpyHostToDevice = hipMemcpyHostToDevice;
static const evloserGpuMemcpyKind_t evloserMemcpyDeviceToHost = hipMemcpyDeviceToHost;
static const evloserGpuMemcpyKind_t evloserMemcpyDeviceToDevice = hipMemcpyDeviceToDevice;

inline evloserGpuError_t evloserGpuMalloc(void** ptr, size_t size)
{
  return hipMalloc(ptr, size);
}

inline evloserGpuError_t evloserGpuFree(void* ptr)
{
  return hipFree(ptr);
}

inline evloserGpuError_t evloserGpuMemcpy(void* dst, const void* src, size_t count, evloserGpuMemcpyKind_t kind)
{
  return hipMemcpy(dst, src, count, kind);
}

inline evloserGpuError_t evloserGpuDeviceSynchronize()
{
  return hipDeviceSynchronize();
}

inline const char* evloserGpuGetErrorString(evloserGpuError_t status)
{
  return hipGetErrorString(status);
}

#endif  // EVLOSER_HIPSOLVER_DEFS_H
