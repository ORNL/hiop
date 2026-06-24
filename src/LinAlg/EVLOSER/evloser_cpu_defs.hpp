#ifndef EVLOSER_CPU_DEFS_HPP
#define EVLOSER_CPU_DEFS_HPP

#if !defined(HIOP_USE_CUDA) && !defined(HAVE_CUDA) && \
    !defined(HIOP_USE_HIP) && !defined(HAVE_HIP)

using evloserGpuError_t = int;
using evloserGpuMemcpyKind_t = int;

using cusolverStatus_t = int;
using cusparseHandle_t = void*;
using cusolverSpHandle_t = void*;
using cublasHandle_t = void*;
using cusparseMatDescr_t = void*;
using csrluInfoHost_t = void*;
using csrgluInfo_t = void*;

using evloserRfStatus_t = int;
using evloserRfHandle_t = void*;

static constexpr evloserGpuError_t evloserGpuSuccess = 0;
static constexpr evloserRfStatus_t evloserRfSuccess = 0;

#endif

#endif  // EVLOSER_CPU_DEFS_HPP