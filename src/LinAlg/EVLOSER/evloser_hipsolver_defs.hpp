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
#include <hipblas/hipblas.h>
#include <hipsparse/hipsparse.h>
#include <hipsolver/hipsolver.h>

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

template <typename T>
inline evloserGpuError_t evloserGpuMalloc(T** ptr, size_t size)
{
  return hipMalloc(reinterpret_cast<void**>(ptr), size);
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

/*
 * Compatibility aliases for EVLOSER code that still uses CUDA-style sparse
 * solver handle names.  The EVLOSER source keeps those names so the HIP path
 * stays close to the original ReSolve implementation.
 */
using cusolverStatus_t = hipsolverStatus_t;
using cusparseHandle_t = hipsparseHandle_t;
using cusolverSpHandle_t = hipsolverSpHandle_t;
using cublasHandle_t = hipblasHandle_t;
using cusparseMatDescr_t = hipsparseMatDescr_t;

#define CUSOLVER_STATUS_SUCCESS HIPSOLVER_STATUS_SUCCESS

#define cusparseCreate hipsparseCreate
#define cusparseDestroy hipsparseDestroy
#define cusparseCreateMatDescr hipsparseCreateMatDescr
#define cusparseDestroyMatDescr hipsparseDestroyMatDescr
#define cusparseSetMatType hipsparseSetMatType
#define cusparseSetMatIndexBase hipsparseSetMatIndexBase

#define CUSPARSE_MATRIX_TYPE_GENERAL HIPSPARSE_MATRIX_TYPE_GENERAL
#define CUSPARSE_INDEX_BASE_ZERO HIPSPARSE_INDEX_BASE_ZERO

#define cusolverSpCreate hipsolverSpCreate
#define cusolverSpDestroy hipsolverSpDestroy

#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy

using evloserRfStatus_t = hipsolverStatus_t;
using evloserRfHandle_t = hipsolverRfHandle_t;
using evloserRfFactorization_t = hipsolverRfFactorization_t;
using evloserRfTriangularSolve_t = hipsolverRfTriangularSolve_t;
using evloserRfMatrixFormat_t = hipsolverRfMatrixFormat_t;
using evloserRfUnitDiagonal_t = hipsolverRfUnitDiagonal_t;
using evloserRfResetValuesFastMode_t = hipsolverRfResetValuesFastMode_t;

static const evloserRfStatus_t evloserRfSuccess = HIPSOLVER_STATUS_SUCCESS;
static const evloserRfFactorization_t evloserRfFactorizationAlg2 = HIPSOLVERRF_FACTORIZATION_ALG2;
static const evloserRfTriangularSolve_t evloserRfTriangularSolveAlg2 = HIPSOLVERRF_TRIANGULAR_SOLVE_ALG2;
static const evloserRfMatrixFormat_t evloserRfMatrixFormatCsr = HIPSOLVERRF_MATRIX_FORMAT_CSR;
static const evloserRfUnitDiagonal_t evloserRfUnitDiagonalStoredL = HIPSOLVERRF_UNIT_DIAGONAL_STORED_L;
static const evloserRfResetValuesFastMode_t evloserRfResetValuesFastModeOn = HIPSOLVERRF_RESET_VALUES_FAST_MODE_ON;

inline evloserRfStatus_t evloserRfCreate(evloserRfHandle_t* handle)
{
  return hipsolverRfCreate(handle);
}

inline evloserRfStatus_t evloserRfDestroy(evloserRfHandle_t handle)
{
  return hipsolverRfDestroy(handle);
}

inline evloserRfStatus_t evloserRfSetAlgs(evloserRfHandle_t handle,
                                          evloserRfFactorization_t fact_alg,
                                          evloserRfTriangularSolve_t solve_alg)
{
  return hipsolverRfSetAlgs(handle, fact_alg, solve_alg);
}

inline evloserRfStatus_t evloserRfSetMatrixFormat(evloserRfHandle_t handle,
                                                  evloserRfMatrixFormat_t format,
                                                  evloserRfUnitDiagonal_t diag)
{
  return hipsolverRfSetMatrixFormat(handle, format, diag);
}

inline evloserRfStatus_t evloserRfSetResetValuesFastMode(evloserRfHandle_t handle,
                                                         evloserRfResetValuesFastMode_t fast_mode)
{
  return hipsolverRfSetResetValuesFastMode(handle, fast_mode);
}

inline evloserRfStatus_t evloserRfSetNumericProperties(evloserRfHandle_t handle, double zero, double boost)
{
  return hipsolverRfSetNumericProperties(handle, zero, boost);
}

inline evloserRfStatus_t evloserRfSetupHost(int n,
                                            int nnzA,
                                            int* csrRowPtrA,
                                            int* csrColIndA,
                                            double* csrValA,
                                            int nnzL,
                                            int* csrRowPtrL,
                                            int* csrColIndL,
                                            double* csrValL,
                                            int nnzU,
                                            int* csrRowPtrU,
                                            int* csrColIndU,
                                            double* csrValU,
                                            int* P,
                                            int* Q,
                                            evloserRfHandle_t handle)
{
  return hipsolverRfSetupHost(n,
                              nnzA,
                              csrRowPtrA,
                              csrColIndA,
                              csrValA,
                              nnzL,
                              csrRowPtrL,
                              csrColIndL,
                              csrValL,
                              nnzU,
                              csrRowPtrU,
                              csrColIndU,
                              csrValU,
                              P,
                              Q,
                              handle);
}

inline evloserRfStatus_t evloserRfResetValues(int n,
                                              int nnzA,
                                              int* csrRowPtrA,
                                              int* csrColIndA,
                                              double* csrValA,
                                              int* P,
                                              int* Q,
                                              evloserRfHandle_t handle)
{
  return hipsolverRfResetValues(n, nnzA, csrRowPtrA, csrColIndA, csrValA, P, Q, handle);
}

inline evloserRfStatus_t evloserRfAnalyze(evloserRfHandle_t handle)
{
  return hipsolverRfAnalyze(handle);
}

inline evloserRfStatus_t evloserRfRefactor(evloserRfHandle_t handle)
{
  return hipsolverRfRefactor(handle);
}

inline evloserRfStatus_t evloserRfSolve(evloserRfHandle_t handle,
                                        int* P,
                                        int* Q,
                                        int nrhs,
                                        double* Temp,
                                        int ldt,
                                        double* XF,
                                        int ldxf)
{
  return hipsolverRfSolve(handle, P, Q, nrhs, Temp, ldt, XF, ldxf);
}

/*
 * cuSOLVER GLU is CUDA-only.  These stubs allow HIP EVLOSER builds to compile
 * code paths that are present in the shared implementation but not used by the
 * HIP RF validation path.
 */
using csrluInfoHost_t = void*;
using csrgluInfo_t = void*;

template<typename... Args>
inline cusolverStatus_t cusolverSpCreateGluInfo(Args...)
{
  return static_cast<cusolverStatus_t>(1);
}

template<typename... Args>
inline cusolverStatus_t cusolverSpDestroyGluInfo(Args...)
{
  return CUSOLVER_STATUS_SUCCESS;
}

template<typename... Args>
inline cusolverStatus_t cusolverSpDgluSetup(Args...)
{
  return static_cast<cusolverStatus_t>(1);
}

template<typename... Args>
inline cusolverStatus_t cusolverSpDgluBufferSize(Args...)
{
  return static_cast<cusolverStatus_t>(1);
}

template<typename... Args>
inline cusolverStatus_t cusolverSpDgluAnalysis(Args...)
{
  return static_cast<cusolverStatus_t>(1);
}

template<typename... Args>
inline cusolverStatus_t cusolverSpDgluReset(Args...)
{
  return static_cast<cusolverStatus_t>(1);
}

template<typename... Args>
inline cusolverStatus_t cusolverSpDgluFactor(Args...)
{
  return static_cast<cusolverStatus_t>(1);
}

template<typename... Args>
inline cusolverStatus_t cusolverSpDgluSolve(Args...)
{
  return static_cast<cusolverStatus_t>(1);
}

template<typename... Args>
inline cusolverStatus_t cusolverSpDnrminf(Args...)
{
  return static_cast<cusolverStatus_t>(1);
}

#endif  // EVLOSER_HIPSOLVER_DEFS_H
