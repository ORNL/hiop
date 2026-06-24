#include "MatrixCsr.hpp"
#include "RefactorizationSolver.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace
{

constexpr double absolute_tolerance = 1e-10;
constexpr double relative_tolerance = 1e-8;

bool nearly_equal(double actual, double expected)
{
  return std::abs(actual - expected) <=
         absolute_tolerance +
             relative_tolerance * std::max(std::abs(actual), std::abs(expected));
}

bool vectors_equal(const std::vector<double>& actual,
                   const std::vector<double>& expected)
{
  if(actual.size() != expected.size()) {
    return false;
  }

  for(std::size_t i = 0; i < actual.size(); ++i) {
    if(!nearly_equal(actual[i], expected[i])) {
      return false;
    }
  }

  return true;
}

bool load_matrix(EVLOSER::RefactorizationSolver& solver,
                 int n,
                 const std::vector<int>& rowptr,
                 const std::vector<int>& colind,
                 const std::vector<double>& values)
{
  if(static_cast<int>(rowptr.size()) != n + 1 ||
     colind.size() != values.size()) {
    return false;
  }

  EVLOSER::MatrixCsr* matrix = solver.mat_A_csr();
  matrix->allocate_size(n);
  matrix->allocate_nnz(static_cast<int>(values.size()));

  std::copy(rowptr.begin(), rowptr.end(), matrix->host_irows());
  std::copy(colind.begin(), colind.end(), matrix->host_jcols());
  std::copy(values.begin(), values.end(), matrix->host_vals());

  solver.set_nnz(static_cast<int>(values.size()));
  solver.ordering() = 1;
  solver.fact() = "klu";
  solver.refact() = "klu";
  solver.use_ir() = "no";
  solver.set_silent_output(false);

  return true;
}

bool factorize(EVLOSER::RefactorizationSolver& solver)
{
  return solver.setup_factorization() == 0 &&
         solver.factorize() == 0;
}

bool test_analyze_factor_solve()
{
  EVLOSER::RefactorizationSolver solver(3, EVLOSER::ExecutionMode::CPU);

  const std::vector<int> rowptr{0, 2, 5, 7};
  const std::vector<int> colind{0, 1, 0, 1, 2, 1, 2};
  const std::vector<double> values{4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0};

  if(!load_matrix(solver, 3, rowptr, colind, values) ||
     !factorize(solver)) {
    return false;
  }

  std::vector<double> rhs{6.0, 10.0, 8.0};

  return solver.triangular_solve(rhs.data(), 0.0) &&
         vectors_equal(rhs, {1.0, 2.0, 3.0});
}

bool test_repeated_solve()
{
  EVLOSER::RefactorizationSolver solver(3, EVLOSER::ExecutionMode::CPU);

  const std::vector<int> rowptr{0, 2, 5, 7};
  const std::vector<int> colind{0, 1, 0, 1, 2, 1, 2};
  const std::vector<double> values{4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0};

  if(!load_matrix(solver, 3, rowptr, colind, values) ||
     !factorize(solver)) {
    return false;
  }

  std::vector<double> first_rhs{6.0, 10.0, 8.0};
  if(!solver.triangular_solve(first_rhs.data(), 0.0) ||
     !vectors_equal(first_rhs, {1.0, 2.0, 3.0})) {
    return false;
  }

  std::vector<double> second_rhs{-3.5, 2.5, 4.5};

  return solver.triangular_solve(second_rhs.data(), 0.0) &&
         vectors_equal(second_rhs, {-1.0, 0.5, 2.0});
}

bool test_value_only_refactor()
{
  EVLOSER::RefactorizationSolver solver(3, EVLOSER::ExecutionMode::CPU);

  const std::vector<int> rowptr{0, 2, 5, 7};
  const std::vector<int> colind{0, 1, 0, 1, 2, 1, 2};
  const std::vector<double> initial_values{4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0};
  const std::vector<double> updated_values{5.0, 1.0, 1.0, 4.0, 1.0, 1.0, 3.0};

  if(!load_matrix(solver, 3, rowptr, colind, initial_values) ||
     !factorize(solver)) {
    return false;
  }

  std::copy(updated_values.begin(),
            updated_values.end(),
            solver.mat_A_csr()->host_vals());

  solver.setup_refactorization();

  if(solver.refactorize() != 0) {
    return false;
  }

  std::vector<double> rhs{7.0, 12.0, 11.0};

  return solver.triangular_solve(rhs.data(), 0.0) &&
         vectors_equal(rhs, {1.0, 2.0, 3.0});
}

bool test_singular_factorization()
{
  EVLOSER::RefactorizationSolver solver(3, EVLOSER::ExecutionMode::CPU);

  const std::vector<int> rowptr{0, 1, 2, 3};
  const std::vector<int> colind{0, 1, 2};
  const std::vector<double> values{1.0, 0.0, 1.0};

  return load_matrix(solver, 3, rowptr, colind, values) &&
         solver.setup_factorization() == 0 &&
         solver.factorize() != 0;
}

bool test_invalid_csr_structure()
{
  EVLOSER::RefactorizationSolver solver(3, EVLOSER::ExecutionMode::CPU);

  const std::vector<int> rowptr{0, 2, 1, 3};
  const std::vector<int> colind{0, 1, 2};
  const std::vector<double> values{1.0, 1.0, 1.0};

  return load_matrix(solver, 3, rowptr, colind, values) &&
         solver.setup_factorization() != 0;
}

bool test_null_rhs()
{
  EVLOSER::RefactorizationSolver solver(3, EVLOSER::ExecutionMode::CPU);

  const std::vector<int> rowptr{0, 1, 2, 3};
  const std::vector<int> colind{0, 1, 2};
  const std::vector<double> values{2.0, 3.0, 4.0};

  return load_matrix(solver, 3, rowptr, colind, values) &&
         factorize(solver) &&
         !solver.triangular_solve(nullptr, 0.0);
}

bool test_nonfinite_matrix_values()
{
  EVLOSER::RefactorizationSolver solver(3, EVLOSER::ExecutionMode::CPU);

  const std::vector<int> rowptr{0, 1, 2, 3};
  const std::vector<int> colind{0, 1, 2};
  const std::vector<double> values{
      1.0,
      std::numeric_limits<double>::quiet_NaN(),
      1.0};

  return load_matrix(solver, 3, rowptr, colind, values) &&
         solver.setup_factorization() != 0;
}

bool test_nonfinite_solution()
{
  EVLOSER::RefactorizationSolver solver(1, EVLOSER::ExecutionMode::CPU);

  const std::vector<int> rowptr{0, 1};
  const std::vector<int> colind{0};
  const std::vector<double> values{1e-200};

  if(!load_matrix(solver, 1, rowptr, colind, values) ||
     !factorize(solver)) {
    return false;
  }

  std::vector<double> rhs{1e200};

  return !solver.triangular_solve(rhs.data(), 0.0);
}


bool test_refactor_accepted()
{
  EVLOSER::RefactorizationSolver solver(3, EVLOSER::ExecutionMode::CPU);

  const std::vector<int> rowptr{0, 2, 5, 7};
  const std::vector<int> colind{0, 1, 0, 1, 2, 1, 2};
  const std::vector<double> initial_values{
      4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0};
  const std::vector<double> updated_values{
      5.0, 1.0, 1.0, 4.0, 1.0, 1.0, 3.0};

  if(!load_matrix(solver, 3, rowptr, colind, initial_values) ||
     !factorize(solver)) {
    return false;
  }

  std::copy(updated_values.begin(),
            updated_values.end(),
            solver.mat_A_csr()->host_vals());

  solver.setup_refactorization();

  if(solver.refactorize() != 0) {
    return false;
  }

  std::vector<double> rhs{7.0, 12.0, 11.0};

  return solver.triangular_solve(rhs.data(), 0.0) &&
         vectors_equal(rhs, {1.0, 2.0, 3.0}) &&
         solver.last_klu_recovery_action() ==
             EVLOSER::RefactorizationSolver::
                 KluRecoveryAction::RefactorAccepted;
}

bool test_refactor_retained_after_comparison()
{
  EVLOSER::RefactorizationSolver solver(3, EVLOSER::ExecutionMode::CPU);

  const std::vector<int> rowptr{0, 2, 5, 7};
  const std::vector<int> colind{0, 1, 0, 1, 2, 1, 2};
  const std::vector<double> initial_values{
      4.0, 1.0, 1.0, 3.0, 1.0, 1.0, 2.0};
  const std::vector<double> updated_values{
      5.0, 1.0, 1.0, 4.0, 1.0, 1.0, 3.0};

  if(!load_matrix(solver, 3, rowptr, colind, initial_values) ||
     !factorize(solver)) {
    return false;
  }

  std::copy(updated_values.begin(),
            updated_values.end(),
            solver.mat_A_csr()->host_vals());

  solver.setup_refactorization();

  /*
   * Force comparison with a fresh numeric factorization. Since both
   * candidates should have comparable residuals, retain the refactored
   * factors.
   */
  solver.klu_suspicious_residual_threshold() = -1.0;

  if(solver.refactorize() != 0) {
    return false;
  }

  std::vector<double> rhs{7.0, 12.0, 11.0};

  return solver.triangular_solve(rhs.data(), 0.0) &&
         vectors_equal(rhs, {1.0, 2.0, 3.0}) &&
         solver.last_klu_recovery_action() ==
             EVLOSER::RefactorizationSolver::
                 KluRecoveryAction::RefactorRetained;
}

bool test_failed_refactor_recovered_by_full_factorization()
{
  EVLOSER::RefactorizationSolver solver(2, EVLOSER::ExecutionMode::CPU);

  const std::vector<int> rowptr{0, 2, 4};
  const std::vector<int> colind{0, 1, 0, 1};

  /*
   * The updated matrix retains the sparsity pattern but invalidates the
   * previous numerical pivot. A fresh factorization can choose a new pivot.
   */
  const std::vector<double> initial_values{
      10.0, 1.0,
      1.0, 1.0};

  const std::vector<double> updated_values{
      0.0, 1.0,
      1.0, 10.0};

  if(!load_matrix(solver, 2, rowptr, colind, initial_values) ||
     !factorize(solver)) {
    return false;
  }

  std::copy(updated_values.begin(),
            updated_values.end(),
            solver.mat_A_csr()->host_vals());

  solver.setup_refactorization();

  /*
   * Refactorization failure is recoverable, so refactorize() permits the
   * subsequent solve to attempt a fresh numeric factorization.
   */
  if(solver.refactorize() != 0) {
    return false;
  }

  std::vector<double> rhs{-1.0, -8.0};

  return solver.triangular_solve(rhs.data(), 0.0) &&
         vectors_equal(rhs, {2.0, -1.0}) &&
         solver.last_klu_recovery_action() ==
             EVLOSER::RefactorizationSolver::
                 KluRecoveryAction::FullFactorAccepted;
}

bool test_unrecoverable_refactor_failure()
{
  EVLOSER::RefactorizationSolver solver(2, EVLOSER::ExecutionMode::CPU);

  const std::vector<int> rowptr{0, 2, 4};
  const std::vector<int> colind{0, 1, 0, 1};
  const std::vector<double> initial_values{
      10.0, 1.0,
      1.0, 1.0};
  const std::vector<double> singular_values{
      1.0, 1.0,
      1.0, 1.0};

  if(!load_matrix(solver, 2, rowptr, colind, initial_values) ||
     !factorize(solver)) {
    return false;
  }

  std::copy(singular_values.begin(),
            singular_values.end(),
            solver.mat_A_csr()->host_vals());

  solver.setup_refactorization();

  if(solver.refactorize() != 0) {
    return false;
  }

  std::vector<double> rhs{2.0, 2.0};

  return !solver.triangular_solve(rhs.data(), 0.0) &&
         solver.last_klu_recovery_action() ==
             EVLOSER::RefactorizationSolver::
                 KluRecoveryAction::Failed;
}

}  // namespace

int main()
{
  struct TestCase
  {
    const char* name;
    bool (*run)();
  };

  const TestCase tests[] = {
      {"analyze/factor/solve", test_analyze_factor_solve},
      {"repeated solve", test_repeated_solve},
      {"value-only refactor", test_value_only_refactor},
      {"singular factorization", test_singular_factorization},
      {"invalid CSR structure", test_invalid_csr_structure},
      {"null RHS", test_null_rhs},
      {"non-finite matrix values", test_nonfinite_matrix_values},
      {"non-finite solution", test_nonfinite_solution},
      {"refactor accepted", test_refactor_accepted},
      {"refactor retained after fresh comparison",
       test_refactor_retained_after_comparison},
      {"failed refactor recovered by full factorization",
       test_failed_refactor_recovered_by_full_factorization},
      {"unrecoverable refactor failure",
       test_unrecoverable_refactor_failure},
  };

  int failures = 0;

  for(const TestCase& test : tests) {
    const bool passed = test.run();
    std::cout << (passed ? "[PASS] " : "[FAIL] ") << test.name << "\n";

    if(!passed) {
      ++failures;
    }
  }

  if(failures != 0) {
    std::cout << failures << " EVLOSER KLU CPU test(s) failed.\n";
    return 1;
  }

  std::cout << "All EVLOSER KLU CPU tests passed.\n";
  return 0;
}
