#pragma once

namespace EVLOSER
{

class MatrixCsr
{
public:
  MatrixCsr();
  ~MatrixCsr();
  void allocate_size(int n);
  void allocate_nnz(int nnz);
  void clear_data();

  int n() const { return n_; }
  int nnz() const { return nnz_; }
  bool has_device_storage() const;
  bool has_host_mirror() const;
  bool validate_host_structure(const char* caller, bool silent_output) const;

  int* device_irows() { return irows_; }

  const int* device_irows() const { return irows_; }

  int* device_jcols() { return jcols_; }

  const int* device_jcols() const { return jcols_; }

  double* device_vals() { return vals_; }

  const double* device_vals() const { return vals_; }

  int* host_irows() { return irows_host_; }

  const int* host_irows() const { return irows_host_; }

  int* host_jcols() { return jcols_host_; }

  const int* host_jcols() const { return jcols_host_; }

  double* host_vals() { return vals_host_; }

  const double* host_vals() const { return vals_host_; }

  void update_from_host_mirror();
  void copy_to_host_mirror();

private:
  int n_{0};
  int nnz_{0};

  int* irows_{nullptr};
  int* jcols_{nullptr};
  double* vals_{nullptr};

  int* irows_host_{nullptr};
  int* jcols_host_{nullptr};
  double* vals_host_{nullptr};

  /**
   * @brief Check for CUDA errors.
   *
   * @tparam T - type of the result
   * @param result - result value
   * @param file   - file name where the error occured
   * @param line   - line at which the error occured
   */
  template<typename T>
  void resolveCheckCudaError(T result, const char* const file, int const line);
};

}  // namespace EVLOSER
