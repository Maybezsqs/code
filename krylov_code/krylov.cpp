#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>

/**
  For convenience, we introduce some new types
*/
using basis = std::unordered_map<std::string, int>;
using inversebasis = std::unordered_map<int, std::string>;
using observable = Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>;
using state = Eigen::VectorXcd;
using complex = std::complex<double>;
typedef Eigen::Triplet<complex> T;


/**
 * Creates an orthonormal subspace the size of Q's columns.
 */
void build_orthogonal_basis_Arnoldi(observable H, state Psi,
                                    Eigen::MatrixXcd &Q, Eigen::MatrixXcd &h) {
  Q.col(0) = Psi.normalized();
  for (int i = 1; i < Q.cols(); ++i) {
    // Obtain the next q vector
    state q = H * Q.col(i - 1);

    // Orthonormalize it w.r.t. all the others
    for (int k = 0; k < i; ++k) {
      h(k, i - 1) = Q.col(k).dot(q);
      q = q - h(k, i - 1) * Q.col(k);
    }

    // Normalize it and store that in our upper Hessenberg matrix h
    h(i, i - 1) = q.norm();

    // Check if we found a subspace
    if (h(i, i - 1).real() <= 1e-14) {
      std::cerr << "Invariant subspace found before finishing Arnoldi"
                << std::endl;
    }

    // And set it as the next column of Q
    Q.col(i) = q / h(i, i - 1);
  }

  // Complete h by setting the last column manually
  int i = Q.cols() - 1;
  h(i, i) = Q.col(i).dot(H * Q.col(i));
  h(i - 1, i) = h(i, i - 1);
}


/**
 * Creates an orthonormal subspace the size of Q's columns.
 */
void build_orthogonal_basis_Lanczos(observable H, state Psi,
                                    Eigen::MatrixXcd &Q, Eigen::MatrixXcd &h) {
  Eigen::VectorXcd alpha(Q.cols());
  Eigen::VectorXcd beta(Q.cols());

  beta(0) = 0;

  Q.col(0) = Psi.normalized();  // u1; U1 = [u1]
  for (int j = 1; j < Q.cols(); ++j) {
    // Obtain the next q vector
    state q = H * Q.col(j - 1);

    alpha(j - 1) = Q.col(j - 1).dot(q);

    q = q - alpha(j - 1) * Q.col(j - 1);
    if (j > 1) {
      q = q - beta(j - 2) * Q.col(j - 2);
    }

    // Re-orthogonalize
    auto delta = Q.col(j - 1).dot(q);
    q = q - Q.col(j - 1) * delta;
    alpha(j - 1) = alpha(j - 1) + delta;

    // Compute the norm
    beta(j - 1) = q.norm();

    // Check if we found a subspace
    if (beta(j - 1).real() <= 1e-14) {
      std::cerr << "Invariant subspace found before finishing Arnoldi"
                << std::endl;
    }

    // And set it as the next column of Q
    Q.col(j) = q / beta(j - 1);
  }

  // Set the projected Hamiltonian
  h.diagonal() = alpha.tail(Q.cols());
  h.diagonal(-1) = beta.head(Q.cols() - 1);
  h.diagonal(+1) = beta.head(Q.cols() - 1);

  // The last diagonal has to be set separately still
  int i = Q.cols() - 1;
  h(i, i) = Q.col(i).dot(H * Q.col(i));
}
