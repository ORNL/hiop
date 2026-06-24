// This file is part of HiOp. For details, see https://github.com/LLNL/hiop.
// HiOp is released under the BSD 3-clause license
// (https://opensource.org/licenses/BSD-3-Clause). Please also read “Additional
// BSD Notice” below.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// i. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the disclaimer below. ii. Redistributions in
// binary form must reproduce the above copyright notice, this list of
// conditions and the disclaimer (as noted below) in the documentation and/or
// other materials provided with the distribution.
// iii. Neither the name of the LLNS/LLNL nor the names of its contributors may
// be used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC,
// THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Additional BSD Notice
// 1. This notice is required to be provided under our contract with the U.S.
// Department of Energy (DOE). This work was produced at Lawrence Livermore
// National Laboratory under Contract No. DE-AC52-07NA27344 with the DOE.
// 2. Neither the United States Government nor Lawrence Livermore National
// Security, LLC nor any of their employees, makes any warranty, express or
// implied, or assumes any liability or responsibility for the accuracy,
// completeness, or usefulness of any information, apparatus, product, or
// process disclosed, or represents that its use would not infringe
// privately-owned rights.
// 3. Also, reference herein to any specific commercial products, process, or
// services by trade name, trademark, manufacturer or otherwise does not
// necessarily constitute or imply its endorsement, recommendation, or favoring
// by the United States Government or Lawrence Livermore National Security,
// LLC. The views and opinions of authors expressed herein do not necessarily
// state or reflect those of the United States Government or Lawrence Livermore
// National Security, LLC, and shall not be used for advertising or product
// endorsement purposes.

/**
 * @file MatrixCsr.cpp
 *
 * @author Kasia Swirydowicz <kasia.Swirydowicz@pnnl.gov>, PNNL
 * @author Slaven Peles <peless@ornl.gov>, ORNL
 *
 */

#include "hiop_blasdefs.hpp"
#include "MatrixCsr.hpp"

#include "evloser_gpu_defs.hpp"
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <cassert>

#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA) || \
    defined(HIOP_USE_HIP) || defined(HAVE_HIP)
#define checkGpuErrors(val) evloserCheckGpuError((val), __FILE__, __LINE__)
#endif

namespace EVLOSER
{

MatrixCsr::MatrixCsr() {}

MatrixCsr::~MatrixCsr()
{
  clear_data();
}

#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA) || \
    defined(HIOP_USE_HIP) || defined(HAVE_HIP)
bool MatrixCsr::has_device_storage() const
{
  const bool size_allocated = (n_ == 0) || (irows_ != nullptr);
  const bool nnz_allocated = (nnz_ == 0) || (jcols_ != nullptr && vals_ != nullptr);
  return size_allocated && nnz_allocated;
}
#endif

bool MatrixCsr::has_host_mirror() const
{
  const bool size_allocated = (n_ == 0) || (irows_host_ != nullptr);
  const bool nnz_allocated = (nnz_ == 0) || (jcols_host_ != nullptr && vals_host_ != nullptr);
  return size_allocated && nnz_allocated;
}

void MatrixCsr::allocate_size(int n)
{
  bool storage_allocated = irows_host_ != nullptr;

#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA) || \
    defined(HIOP_USE_HIP) || defined(HAVE_HIP)

  storage_allocated = storage_allocated || irows_ != nullptr;

#endif

  if(storage_allocated) {
    clear_data();
  }

  n_ = n;

#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA) || \
    defined(HIOP_USE_HIP) || defined(HAVE_HIP)

  checkGpuErrors(evloserGpuMalloc(reinterpret_cast<void**>(&irows_), (n_ + 1) * sizeof(int)));

#endif

  irows_host_ = new int[n_ + 1]{0};
}

void MatrixCsr::allocate_nnz(int nnz)
{
  bool storage_allocated = jcols_host_ != nullptr || vals_host_ != nullptr;

#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA) || \
    defined(HIOP_USE_HIP) || defined(HAVE_HIP)

  storage_allocated = storage_allocated || jcols_ != nullptr || vals_ != nullptr;

#endif

  if(storage_allocated) {
#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA) || \
    defined(HIOP_USE_HIP) || defined(HAVE_HIP)

    checkGpuErrors(evloserGpuFree(jcols_));
    checkGpuErrors(evloserGpuFree(vals_));

    jcols_ = nullptr;
    vals_ = nullptr;

#endif

    delete[] jcols_host_;
    delete[] vals_host_;

    jcols_host_ = nullptr;
    vals_host_ = nullptr;
    nnz_ = 0;
  }

  nnz_ = nnz;

  if(nnz_ == 0) {
    return;
  }

#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA) || \
    defined(HIOP_USE_HIP) || defined(HAVE_HIP)

  checkGpuErrors(evloserGpuMalloc(reinterpret_cast<void**>(&jcols_), nnz_ * sizeof(int)));
  checkGpuErrors(evloserGpuMalloc(reinterpret_cast<void**>(&vals_), nnz_ * sizeof(double)));

#endif

  jcols_host_ = new int[nnz_]{0};
  vals_host_ = new double[nnz_]{0};
}

void MatrixCsr::clear_data()
{
#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA) || \
    defined(HIOP_USE_HIP) || defined(HAVE_HIP)

  checkGpuErrors(evloserGpuFree(irows_));
  checkGpuErrors(evloserGpuFree(jcols_));
  checkGpuErrors(evloserGpuFree(vals_));

  irows_ = nullptr;
  jcols_ = nullptr;
  vals_ = nullptr;

#endif

  delete[] irows_host_;
  delete[] jcols_host_;
  delete[] vals_host_;

  irows_host_ = nullptr;
  jcols_host_ = nullptr;
  vals_host_ = nullptr;

  n_ = 0;
  nnz_ = 0;
}

#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA) || \
    defined(HIOP_USE_HIP) || defined(HAVE_HIP)
void MatrixCsr::update_from_host_mirror()
{
  assert(has_device_storage());
  assert(has_host_mirror());

  checkGpuErrors(evloserGpuMemcpy(irows_, irows_host_, sizeof(int) * (n_ + 1), evloserMemcpyHostToDevice));

  if(nnz_ > 0) {
    checkGpuErrors(evloserGpuMemcpy(jcols_, jcols_host_, sizeof(int) * nnz_, evloserMemcpyHostToDevice));
    checkGpuErrors(evloserGpuMemcpy(vals_, vals_host_, sizeof(double) * nnz_, evloserMemcpyHostToDevice));
  }
}

void MatrixCsr::copy_to_host_mirror()
{
  assert(has_device_storage());
  assert(has_host_mirror());

  checkGpuErrors(evloserGpuMemcpy(irows_host_, irows_, sizeof(int) * (n_ + 1), evloserMemcpyDeviceToHost));

  if(nnz_ > 0) {
    checkGpuErrors(evloserGpuMemcpy(jcols_host_, jcols_, sizeof(int) * nnz_, evloserMemcpyDeviceToHost));
    checkGpuErrors(evloserGpuMemcpy(vals_host_, vals_, sizeof(double) * nnz_, evloserMemcpyDeviceToHost));
  }
}
#endif

bool MatrixCsr::validate_host_structure(const char* caller, bool silent_output) const
{
  const char* caller_name = caller == nullptr ? "unknown caller" : caller;

  auto report = [&](const std::string& message) {
    if(!silent_output) {
      std::cout << "[EVLOSER] Invalid CSR matrix in " << caller_name << ": " << message << "\n";
    }
    return false;
  };

  if(n_ <= 0) {
    return report("matrix dimension must be positive");
  }

  if(nnz_ < 0) {
    return report("number of nonzeros is negative");
  }

  if(irows_host_ == nullptr) {
    return report("host row pointer is null");
  }

  if(irows_host_[0] != 0) {
    return report("row pointer must start at zero");
  }

  for(int row = 0; row < n_; ++row) {
    if(irows_host_[row] > irows_host_[row + 1]) {
      return report("row pointer is not monotone");
    }
  }

  if(irows_host_[n_] != nnz_) {
    return report("final row pointer does not match nnz");
  }

  if(nnz_ == 0) {
    return true;
  }

  if(jcols_host_ == nullptr) {
    return report("host column index array is null");
  }

  if(vals_host_ == nullptr) {
    return report("host value array is null");
  }

  for(int k = 0; k < nnz_; ++k) {
    if(jcols_host_[k] < 0 || jcols_host_[k] >= n_) {
      return report("column index out of range");
    }
  }

  return true;
}

#if defined(HIOP_USE_CUDA) || defined(HAVE_CUDA) || \
    defined(HIOP_USE_HIP) || defined(HAVE_HIP)
// Error checking utility for GPU backend
// KS: might later become part of src/Utils, putting it here for now
template<typename T>
void MatrixCsr::evloserCheckGpuError(T result, const char* const file, int const line)
{
  if(result) {
    std::cout << "GPU error at " << file << ":" << line << " error# " << result << "\n";
    assert(false);
  }
}
#endif
}  // namespace EVLOSER
