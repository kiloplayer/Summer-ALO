#include <RcppArmadillo.h>
#include <math.h>

using namespace Rcpp;
using namespace arma;

// [[Rcpp::plugins(cpp11)]]

// [[Rcpp::export]]
vec ElasticNetALO(const vec &beta, const double &intercept, 
                  const mat &X, const vec &y, const mat &XtX, 
                  const double &lambda, const double &alpha) {
  // define full beta (intercept & slope)
  vec beta_full(X.n_cols); // X is full design matrix
  beta_full(0) = intercept;
  beta_full(span(1, X.n_cols - 1)) = beta;
  // compute prediction
  vec y_hat = X * beta_full;
  // find active set
  uvec A = find(beta_full != 0);
  cout << "Active Set Size: " << A.n_elem << endl;
  // compute matrix H
  vec diag_H(A.n_elem, fill::zeros);
  if(!A.is_empty()) {
    // mat X_active = X_full.cols(A).t();
    vec R_diff2(A.n_elem, fill::ones);
    // mat R_diff2(A.n_elem, A.n_elem, fill::eye);
    R_diff2 = R_diff2 * X.n_rows * lambda * (1 - alpha);
    if(intercept != 0) {
      R_diff2(0) = 0;
      // R_diff2(0,0) = 0;
    }
    // mat middle = X_full.cols(A).t() * X_full.cols(A);
    mat middle = XtX(A, A);
    middle.diag() = middle.diag() + R_diff2;
    middle = inv_sympd(middle);
    diag_H = sum((X.cols(A) * middle) % X.cols(A), 1);
    // mat L = chol(X_active.t() * X_active + R_diff2).t(); // lower triangle
    // mat AE = solve(trimatl(L), X_active.t());
    // H = AE.t() * AE;
    // H = X_active * inv_sympd(X_active.t() * X_active + R_diff2) * X_active.t();
  }
  // compute the ALO prediction
  // vec y_alo = y_hat + H.diag() % (y_hat - y) / (1-H.diag());
  vec y_alo = y_hat + diag_H % (y_hat - y) / (1 - diag_H);
  return y_alo;
}


// [[Rcpp::export]]
vec ElasticNetALO_Weight(const vec &beta, const bool &intercept, 
                         const mat &X, const vec &y, const vec &weights, 
                         const mat &XWX, const double &lambda, const double &alpha) {
  // compute prediction
  vec y_hat = X * beta;
  // find active set
  uvec A = find(beta != 0);
  if(intercept) {
    A.insert_rows(0, 0);
    A = unique(A);
  }
  // compute matrix H
  vec diag_HW(X.n_rows, fill::zeros);
  if(!A.is_empty()) {
    vec R_diff2(A.n_elem, fill::ones);
    R_diff2 = R_diff2 * sum(weights) * lambda * (1 - alpha);
    if(intercept) {
      R_diff2(0) = 0;
    }
    mat middle = XWX(A, A);
    middle.diag() = middle.diag() + R_diff2;
    middle = inv_sympd(middle);
    diag_HW = sum((X.cols(A) * middle) % X.cols(A), 1) % weights;
  }
  // compute the ALO prediction
  vec y_alo = y_hat + diag_HW % (y_hat - y) / (1 - diag_HW);
  return y_alo;
}


// [[Rcpp::export]]
vec LogisticALO(vec beta, double intercept, 
                mat X, vec y, 
                double lambda, double alpha) {
  // find out the dimension of X
  double n = X.n_rows;
  double p = X.n_cols;
  // define full data set
  vec ones(n,fill::ones);
  mat X_full = X;
  X_full.insert_cols(0, ones);
  // define full beta (intercept & slope)
  vec beta_full(p + 1);
  beta_full(0) = intercept;
  beta_full(span(1, p)) = beta;
  // compute linear prediction
  vec y_linear = X_full * beta_full;
  vec y_exp = exp(y_linear);
  // find active set
  uvec A = find(beta_full != 0);
  // compute matrix D
  mat D(n, n, fill::zeros);
  D.diag() = y_exp / ((1 + y_exp) % (1 + y_exp));
  // compute matrix H
  mat H(n, n, fill::zeros);
  if(!A.is_empty()) {
    mat X_active = X_full.cols(A);
    mat R_diff2(A.n_elem, A.n_elem, fill::eye);
    R_diff2 = R_diff2 * n * lambda * (1 - alpha);
    if(intercept != 0) {
      R_diff2(0,0) = 0;
    }
    mat L = chol(X_active.t() * D * X_active + R_diff2).t();
    mat L_inv = inv(L);
    mat AE = L_inv * X_active.t();
    H = AE.t() * AE;
  }
  // compute the ALO prediction
  vec y_alo_linear = y_linear + H.diag() % (y_exp / (1 + y_exp) - y) / (1-H.diag() % D.diag());
  vec y_alo = exp(y_alo_linear) / (1 + exp(y_alo_linear));
  return y_alo;
}

// [[Rcpp::export]]
vec LogisticALO_Weight(const vec &beta, const bool &intercept, 
                       const mat &X, const vec &y, const vec &weights,
                       const double &lambda, const double &alpha) {
  // compute linear prediction
  vec y_linear = X * beta;
  vec y_exp = exp(y_linear);
  // find active set
  uvec A = find(beta != 0);
  if(intercept) {
    A.insert_rows(0, 0);
    A = unique(A);
  }
  // compute matrix D
  vec D = y_exp / ((1 + y_exp) % (1 + y_exp));
  // compute matrix H
  vec diag_HW(X.n_rows, fill::zeros);
  if(!A.is_empty()) {
    mat X_active = X.cols(A);
    vec R_diff2(A.n_elem, fill::ones);
    R_diff2 = R_diff2 * sum(weights) * lambda * (1 - alpha);
    if(intercept) {
      R_diff2(0) = 0;
    }
    mat middle = X_active.t() * diagmat(D % weights) * X_active;
    middle.diag() = middle.diag() + R_diff2;
    middle = inv_sympd(middle);
    diag_HW = sum((X_active * middle) % X_active, 1) % weights;
  }
  // compute the ALO prediction
  vec y_alo_linear = y_linear + diag_HW % (y_exp / (1 + y_exp) - y) / (1 - diag_HW % D);
  vec y_alo = exp(y_alo_linear) / (1 + exp(y_alo_linear));
  return y_alo;
}


// [[Rcpp::export]]
vec PoissonALO(vec beta, double intercept, 
               mat X, vec y, 
               double lambda, double alpha) {
  // find out the dimension of X
  double n = X.n_rows;
  double p = X.n_cols;
  // define full data set
  vec ones(n,fill::ones);
  mat X_full = X;
  X_full.insert_cols(0, ones);
  // define full beta (intercept & slope)
  vec beta_full(p + 1);
  beta_full(0) = intercept;
  beta_full(span(1, p)) = beta;
  // compute linear prediction
  vec y_linear = X_full * beta_full;
  // find active set
  uvec A = find(beta_full != 0);
  // compute matrix D
  mat D(n, n, fill::zeros);
  D.diag() = exp(y_linear);
  // compute matrix H
  mat H(n, n, fill::zeros);
  if(!A.is_empty()) {
    mat X_active = X_full.cols(A);
    mat R_diff2(A.n_elem, A.n_elem, fill::eye);
    R_diff2 = R_diff2 * n * lambda * (1 - alpha);
    if(intercept != 0) {
      R_diff2(0,0) = 0;
    }
    mat L = chol(X_active.t() * D * X_active + R_diff2).t();
    mat L_inv = inv(L);
    mat AE = L_inv * X_active.t();
    H = AE.t() * AE;
  }
  // compute the ALO prediction
  vec y_alo_linear = y_linear + H.diag() % (exp(y_linear) - y) / (1-H.diag() % D.diag());
  vec y_alo = exp(y_alo_linear);
  return y_alo;
}

// [[Rcpp::export]]
mat MultinomialALO(vec beta, bool intercept, 
                   mat X, mat X_Sp, mat y_mat, 
                   double lambda, double alpha) {
  // find out the dimension of X
  double n = X.n_rows; // N
  double p; // P
  if (intercept) {
    p = X.n_cols - 1;
  } else {
    p = X.n_cols;
  }
  double num_class = y_mat.n_cols; // K
  // find the active set
  uvec E = find(beta != 0);
  // compute vector A(beta) and matrix D(beta)
  vec A(n * num_class, fill::none);
  mat D(n * num_class, n * num_class, fill::zeros);
  for (uword i = 0; i < n; ++i) {
    uvec idx = regspace<uvec>(i * num_class, 1, (i + 1) * num_class - 1);
    A(idx) = exp(X_Sp.rows(idx) * beta);
    A(idx) = A(idx) / sum(A(idx));
    D(idx, idx) = diagmat(A(idx)) - A(idx) * A(idx).t();
  }
  // compute R_diff2
  mat R_diff2;
  if (intercept) {
    R_diff2 = eye<mat>((p + 1) * num_class, (p + 1) * num_class);
    R_diff2.diag() = R_diff2.diag() * n * lambda * (1 - alpha);
    uvec idx = regspace<uvec>(0, num_class - 1) * (p + 1);
    R_diff2(idx, idx) = R_diff2(idx, idx) * 0;
  } else {
    R_diff2 = eye<mat>(p * num_class, p * num_class);
    R_diff2.diag() = R_diff2.diag() * n * lambda * (1 - alpha);
  }
  // compute matrix K(beta) and its inverse
  mat K_inv = pinv(X_Sp.cols(E).t() * D * X_Sp.cols(E) + R_diff2(E, E), 0);
  // do leave-i-out prediction
  mat y_alo(n, num_class, fill::none);
  for (uword i = 0; i < n; ++i) {
    // find the X_i and y_i
    uvec idx = regspace<uvec>(i * num_class, 1, (i + 1) * num_class - 1);
    mat X_i = X_Sp.rows(idx);
    vec y_i = conv_to<vec>::from(y_mat.row(i));
    // find A_i
    vec A_i = A(idx);
    // compute XKX
    mat XKX = X_i.cols(E) * K_inv * X_i.cols(E).t();
    // compute the inversion of diag(A)-A*A^T
    mat middle_inv = pinv(diagmat(A_i) - A_i * A_i.t(), 0);
    // compute the leave-i-out prediction
    vec y_linear = X_i * beta + XKX * (A_i - y_i) - 
      XKX * pinv(-middle_inv + XKX, 0) * XKX * (A_i - y_i);
    vec y_exp = exp(y_linear);
    y_alo.row(i) = conv_to<rowvec>::from(y_exp) / sum(y_exp);
  }
  // return alo
  return(y_alo);
}

// [[Rcpp::export]]
mat MultinomialALO_Weight(const vec &beta, const bool &intercept, 
                          const mat &X, const mat &X_Sp, const mat &y_mat,
                          const vec &weights, 
                          const double &lambda, const double &alpha) {
  // find out the dimension of X
  double n = X.n_rows; // N
  double p; // P
  if (intercept) {
    p = X.n_cols - 1;
  } else {
    p = X.n_cols;
  }
  double num_class = y_mat.n_cols; // K
  // find the active set
  uvec E = find(beta != 0);
  if (intercept) {
    uvec idx = regspace<uvec>(0, num_class - 1) * (p + 1);
    E.insert_rows(0, idx);
    E = unique(E);
  }
  // compute vector A(beta) and matrix D(beta)
  vec A(n * num_class, fill::none);
  mat D(n * num_class, n * num_class, fill::zeros);
  for (uword i = 0; i < n; ++i) {
    uvec idx = regspace<uvec>(i * num_class, 1, (i + 1) * num_class - 1);
    A(idx) = exp(X_Sp.rows(idx) * beta);
    A(idx) = A(idx) / sum(A(idx));
    D(idx, idx) = weights(i) * (diagmat(A(idx)) - A(idx) * A(idx).t());
  }
  // compute R_diff2
  vec R_diff2((p + 1) * num_class, fill::ones);
  R_diff2 = R_diff2 * sum(weights) * lambda * (1 - alpha);
  if(intercept) {
    uvec idx = regspace<uvec>(0, num_class - 1) * (p + 1);
    R_diff2(idx) = R_diff2(idx) * 0;
  }
  // compute matrix K(beta) and its inverse
  mat K_inv = X_Sp.cols(E).t() * D * X_Sp.cols(E);
  K_inv.diag() = K_inv.diag() + R_diff2(E);
  K_inv = pinv(K_inv, 0);
  // do leave-i-out prediction
  mat y_alo(n, num_class, fill::none);
  for (uword i = 0; i < n; ++i) {
    // find the X_i and y_i
    uvec idx = regspace<uvec>(i * num_class, 1, (i + 1) * num_class - 1);
    mat X_i = X_Sp.rows(idx);
    vec y_i = conv_to<vec>::from(y_mat.row(i));
    // find A_i
    vec A_i = A(idx);
    // compute XKX
    mat XKX = X_i.cols(E) * K_inv * X_i.cols(E).t();
    // compute the inversion of diag(A)-A*A^T
    mat middle_inv = pinv(weights(i) * (diagmat(A_i) - A_i * A_i.t()), 0);
    // compute the leave-i-out prediction
    vec y_linear = X_i * beta + XKX * 
      (eye<mat>(num_class, num_class) - pinv(-middle_inv + XKX, 0) * XKX) * 
      (weights(i) * (A_i - y_i));
    vec y_exp = exp(y_linear);
    y_alo.row(i) = conv_to<rowvec>::from(y_exp) / sum(y_exp);
  }
  // return alo
  return(y_alo);
}


// [[Rcpp::export]]
vec GenLASSOALO(vec beta, vec u, mat X, vec y, mat D, 
                double lambda, double tol) {
  // compute prediction
  vec y_hat = X * beta;
  // find complement of the active set
  uvec mE = find(abs(abs(u) - lambda) >= tol);
  // find the null space of D_{-E,*}
  mat VE = null(D.rows(mE));
  // find matrix H
  mat H = X * VE * pinv(X * VE);
  // compute the ALO prediction
  vec y_alo = y + (y_hat - y) / (1-H.diag());
  return y_alo;
}

// [[Rcpp::export]]
field<mat> CholeskyAdd(mat X, mat L, 
                       uvec idx_old, uvec idx_new) {
  // X - full symmetric matrix
  // L - upper triangle matrix of Cholesky decomposition for X[idx_old, idx_old]
  // idx_old - old index of L for the X[idx_old, idx_old]
  // idx_new - find the Cholesky decomposition for X[idx_new, idx_new], 
  //           where idx_old \in idx_new
  
  // if idx_old is empty, just find the Cholesky of the idx_new
  if (idx_old.is_empty() && !idx_new.is_empty()) {
    mat L_out = chol(X(idx_new,idx_new));
    field<mat> out(2);
    out(0) = L_out;
    out(1) = conv_to<mat>::from(idx_new);
    return out;
  }
  // find the rows that is needed to be added
  uvec idx_add;
  for (size_t i = 0; i < idx_new.n_elem; ++i) {
    uvec temp = find(idx_old == idx_new[i]);
    if (temp.is_empty()) {
      idx_add.resize(idx_add.n_elem + 1);
      idx_add(idx_add.n_elem - 1) = idx_new[i];
    }
  }
  // if idx_add is empty - no adding rows, then return matrix
  if (idx_add.is_empty()) {
    field<mat> out(2);
    out(0) = L; // upper triangle matrix
    out(1) = conv_to<mat>::from(idx_old);
    return out;
  }
  // update Cholesky decomposition
  mat S12 = solve(trimatl(L.t()), X(idx_old, idx_add));
  mat L_out(idx_new.n_elem, idx_new.n_elem, fill::zeros);
  L_out.submat(span(0, L.n_rows - 1), span(0, L.n_cols - 1)) = L;
  L_out.submat(span(0, L.n_rows - 1), span(L.n_cols, L_out.n_cols - 1)) = S12;
  L_out.submat(span(L.n_rows, L_out.n_rows - 1), span(L.n_cols, L_out.n_cols - 1)) = chol(X(idx_add, idx_add) - S12.t() * S12);
  // find the output index for Cholesky decomposition
  uvec idx_out = idx_old;
  idx_out.resize(idx_new.n_elem);
  idx_out(span(idx_old.n_elem, idx_out.n_elem - 1)) = idx_add;
  // output
  field<mat> out(2);
  out(0) = L_out;
  out(1) = conv_to<mat>::from(idx_out);
  return out;
}

// [[Rcpp::export]]
field<mat> CholeskyAdd_Weight(const mat &XWX, const mat &L, 
                              const uvec &idx_old, const uvec &idx_new) {
  // XWX - t(X) %*% W %X% X
  // L - upper triangle matrix of Cholesky decomposition for XWX[idx_old, idx_old]
  // idx_old - old index of L for the XWX[idx_old, idx_old]
  // idx_new - find the Cholesky decomposition for XWX[idx_new, idx_new], 
  //           where idx_old \in idx_new
  
  // if idx_old is empty, just find the Cholesky of the idx_new
  if (idx_old.is_empty() && !idx_new.is_empty()) {
    mat L_out = chol(XWX(idx_new,idx_new));
    field<mat> out(2);
    out(0) = L_out;
    out(1) = conv_to<mat>::from(idx_new);
    return out;
  }
  // find the rows that is needed to be added
  uvec idx_add;
  for (size_t i = 0; i < idx_new.n_elem; ++i) {
    uvec temp = find(idx_old == idx_new[i]);
    if (temp.is_empty()) {
      idx_add.resize(idx_add.n_elem + 1);
      idx_add(idx_add.n_elem - 1) = idx_new[i];
    }
  }
  // if idx_add is empty - no adding rows, then return matrix
  if (idx_add.is_empty()) {
    field<mat> out(2);
    out(0) = L; // upper triangle matrix
    out(1) = conv_to<mat>::from(idx_old);
    return out;
  }
  // update Cholesky decomposition
  mat S12 = solve(trimatl(L.t()), XWX(idx_old, idx_add));
  mat L_out(idx_new.n_elem, idx_new.n_elem, fill::zeros);
  L_out.submat(span(0, L.n_rows - 1), span(0, L.n_cols - 1)) = L;
  L_out.submat(span(0, L.n_rows - 1), span(L.n_cols, L_out.n_cols - 1)) = S12;
  L_out.submat(span(L.n_rows, L_out.n_rows - 1), span(L.n_cols, L_out.n_cols - 1)) = chol(XWX(idx_add, idx_add) - S12.t() * S12);
  // find the output index for Cholesky decomposition
  uvec idx_out = idx_old;
  idx_out.resize(idx_new.n_elem);
  idx_out(span(idx_old.n_elem, idx_out.n_elem - 1)) = idx_add;
  // output
  field<mat> out(2);
  out(0) = L_out;
  out(1) = conv_to<mat>::from(idx_out);
  return out;
}

// [[Rcpp::export]]
field<mat> CholeskyDrop(mat X, mat L, 
                        uvec idx_old, uvec idx_new) {
  // X - full symmetric matrix
  // L - upper triangle matrix of Cholesky decomposition for X[idx_old, idx_old]
  // idx_old - old index of L for the X[idx_old, idx_old]
  // idx_new - find the Cholesky decomposition for X[idx_new, idx_new], 
  //           where idx_new \in idx_old
  
  // find one row that is needed to be droped
  uvec idx_drop;
  uword pos = 0;
  for (uword i = 0; i < idx_old.n_elem; ++i) {
    uvec temp = find(idx_new == idx_old(i));
    if (temp.is_empty()) {
      idx_drop = idx_old(i); // drop this row
      pos = i;
      break;
    }
  }
  // if idx_drop is empty - no dropping row, then return matrix
  if (idx_drop.is_empty()) {
    field<mat> out(2);
    out(0) = L; // upper triangle matrix
    out(1) = conv_to<mat>::from(idx_old);
    return out;
  } else if (pos == 0) { // drop the first row
    if (L.n_cols == 1) { // only have one row before dropping
      mat L_out(0,0);
      uvec idx_out;
      field<mat> out(2);
      out(0) = L_out;
      out(1) = conv_to<mat>::from(idx_out);
      return out;
    } else {
      mat S23 = L(pos, span(pos + 1, L.n_cols - 1));
      mat S33 = L(span(pos + 1, L.n_rows - 1), span(pos + 1, L.n_cols - 1));
      field<mat> out(2);
      out(0) = chol(S33.t() * S33 + S23.t() * S23);
      out(1) = conv_to<mat>::from(idx_old(span(pos + 1, idx_old.n_elem - 1)));
      return out;
    }
  } else if (pos == idx_old.n_elem - 1) { // drop the last row
    field<mat> out(2);
    out(0) = L(span(0, pos - 1), span(0, pos - 1)); // upper triangle matrix
    out(1) = conv_to<mat>::from(idx_old(span(0, pos - 1)));
    return out;
  }
  // update Cholesky decomposition
  mat S11 = L(span(0, pos - 1), span(0, pos - 1));
  mat S13 = L(span(0, pos - 1), span(pos + 1, L.n_cols - 1));
  mat S33 = L(span(pos + 1, L.n_rows - 1), span(pos + 1, L.n_cols - 1));
  mat S23 = L(pos, span(pos + 1, L.n_cols - 1));
  mat L_out(idx_old.n_elem - 1, idx_old.n_elem - 1, fill::zeros);
  L_out.submat(span(0, S11.n_rows - 1), span(0, S11.n_cols - 1)) = S11;
  L_out.submat(span(0, S11.n_rows - 1), span(S11.n_cols, L_out.n_cols - 1)) = S13;
  L_out.submat(span(S11.n_rows, L_out.n_rows - 1), 
               span(S11.n_cols, L_out.n_cols - 1)) = 
                 chol(S33.t() * S33 + S23.t() * S23);
  // find the output index for Cholesky decomposition
  uvec idx_out(idx_old.n_elem - 1);
  idx_out(span(0, pos - 1)) = idx_old(span(0, pos - 1));
  idx_out(span(pos, idx_out.n_elem - 1)) = idx_old(span(pos + 1, idx_old.n_elem - 1));
  // output
  field<mat> out(2);
  out(0) = L_out;
  out(1) = conv_to<mat>::from(idx_out);
  return out;
}

// [[Rcpp::export]]
field<mat> CholeskyUpdate(mat X, mat L, 
                          uvec idx_old, uvec idx_new) {
  // X - full symmetric matrix
  // L - upper triangle matrix of Cholesky decomposition for X[idx_old, idx_old]
  // idx_old - old index of L for the X[idx_old, idx_old]
  // idx_new - find the Cholesky decomposition for X[idx_new, idx_new]
  
  // first, we need to drop the rows
  uword drop_done = 0;
  while (drop_done == 0) {
    if (idx_old.is_empty()) {
      drop_done = 1;
    }
    for (uword i = 0; i < idx_old.n_elem; ++i) {
      uvec temp = find(idx_new == idx_old(i));
      if (temp.is_empty()) {
        field<mat> drop_temp = CholeskyDrop(X, L, idx_old, idx_new);
        L = drop_temp(0);
        idx_old = conv_to<uvec>::from(drop_temp(1));
        break;
      }
      if (i == idx_old.n_elem - 1) {
        drop_done = 1;
      }
    }
  }
  // second, add new rows
  field<mat> out = CholeskyAdd(X, L, idx_old, idx_new);
  return out;
}

// [[Rcpp::export]]
field<mat> ElasticNetALO_CholUpdate(vec beta, double intercept, 
                                    mat X, vec y, 
                                    double lambda, double alpha, 
                                    mat L_old, uvec idx_old, mat XtX) {
  // find out the dimension of X
  double n = X.n_rows;
  double p = X.n_cols;
  // define full data set
  vec ones(n,fill::ones);
  mat X_full = X;
  X_full.insert_cols(0, ones);
  // define full beta (intercept & slope)
  vec beta_full(p + 1);
  beta_full(0) = intercept;
  beta_full(span(1, p)) = beta;
  // compute prediction
  vec y_hat = X_full * beta_full;
  // find active set
  uvec A = find(beta_full != 0);
  // compute matrix H
  mat H(n, n, fill::zeros);
  uvec idx;
  mat L;
  cout<<A.n_elem<<endl;
  if(!A.is_empty()) {
    mat X_active = X_full.cols(A);
    if (L_old.n_rows == 0) {
      L = chol(X_active.t() * X_active); // upper triangle matrix
      idx = A;
    } else {
      field<mat> update = CholeskyUpdate(XtX, L_old, idx_old, A);
      L = update(0);
      idx = conv_to<uvec>::from(update(1));
    }
    // cout<<idx<<endl;
    mat AE = solve(trimatl(L.t()), X_full.cols(idx).t());
    H = AE.t() * AE;
  }
  // compute the ALO prediction
  vec y_alo = y_hat + H.diag() % (y_hat - y) / (1 - H.diag());
  // generate the output variable
  field<mat> out(3);
  out(0) = conv_to<mat>::from(y_alo);
  out(1) = L;
  out(2) = conv_to<mat>::from(idx);
  return out;
}


// [[Rcpp::export]]
vec ElasticNetALO_CholUpdate_Weight(const vec &beta, const bool &intercept, 
                                    const mat &X, const vec &y, const vec &weights,
                                    const double &lambda, const double &alpha, 
                                    const mat &L, const uvec &idx) {
  // compute prediction
  vec y_hat = X * beta;
  // find active set
  uvec A = find(beta != 0);
  if(intercept) {
    A.insert_rows(0, 0);
    A = unique(A);
  }
  // compute diag_HW
  vec diag_HW(X.n_rows, fill::zeros);
  if(!A.is_empty()) {
    mat AE = solve(trimatl(L.t()), X.cols(idx).t()); // |E| * N
    diag_HW = sum(AE % AE, 0).t() % weights;
  }
  // compute the ALO prediction
  vec y_alo = y_hat + diag_HW % (y_hat - y) / (1 - diag_HW);
  return y_alo;
}

// [[Rcpp::export]]
mat BlockInverse_Add(const mat &X, const uvec &idx_A, 
                     const uvec &idx_add, const mat &A_inv) {
  // using block matrix inversion lemma to update the inverse of a matrix
  // X - symmetric matrix
  // idx_A - old index in X for A_inv
  // idx_add - index in X that should be added to the updated inverse
  // A_inv - the inverse of X[idx_old, idx_old]
  
  // if A_inv is empty, then return the inverse
  if (idx_A.is_empty()) {
    mat out = inv(X(idx_add, idx_add));
    return(out);
  }
  
  // compute matrix E
  mat E = inv(X(idx_add, idx_add) - 
    X(idx_add, idx_A) * A_inv * X(idx_A, idx_add));
  
  // compute matrix B * E
  mat BE = X(idx_A, idx_add) * E;
  
  // compute the updated inverse matrix
  uword size = idx_A.n_elem + idx_add.n_elem;
  mat out(size, size);
  out.submat(0, idx_A.n_elem, idx_A.n_elem - 1, size - 1) = - A_inv * BE;
  out.submat(0, 0, idx_A.n_elem - 1, idx_A.n_elem - 1) = 
    A_inv - out.submat(0, idx_A.n_elem, idx_A.n_elem - 1, size - 1) * 
    X(idx_add, idx_A) * A_inv;
  // out.submat(idx_A.n_elem, 0, size - 1, idx_A.n_elem - 1) = - E * X(idx_add, idx_A) * A_inv;
  out.submat(idx_A.n_elem, 0, size - 1, idx_A.n_elem - 1) = out.submat(0, idx_A.n_elem, idx_A.n_elem - 1, size - 1).t();
  out.submat(idx_A.n_elem, idx_A.n_elem, size - 1, size - 1) = E;
  
  // return value
  return(out);
}


// [[Rcpp::export]]
mat BlockInverse_Drop(const mat &F_inv, const uword &n_keep) {
  // using block matrix inversion lemma to update the inverse of a matrix
  // F_inv - the inverse of the full matrix, we only need the inverse of the first a few rows
  // n_keep - number of rows & cols to keep
  
  // if the number of keeped is equal to the F_inv, then return the F_inv
  if (n_keep == F_inv.n_rows) {
    return(F_inv);
  }
  
  // if keep 0
  if (n_keep == 0) {
    mat out;
    return(out);
  }
  
  // compute the updated matrix
  mat out = F_inv.submat(0, 0, n_keep - 1, n_keep - 1) - 
    F_inv.submat(0, n_keep, n_keep - 1, F_inv.n_cols - 1) * 
    inv(F_inv.submat(n_keep, n_keep, F_inv.n_rows - 1, F_inv.n_cols - 1)) * 
    F_inv.submat(n_keep, 0, F_inv.n_rows - 1, n_keep - 1);
  
  // return value
  return out;
}


// [[Rcpp::export]]
vec BlockInverse_ALO(const mat &X, const mat &A_inv, const vec &y, 
                     const vec &beta, const uvec &E) {
  // compute y_hat
  vec y_hat = X * beta;
  // compute diag(H)
  vec diag_H = sum((X.cols(E) * A_inv) % X.cols(E), 1);
  // compute ALO
  vec y_alo = y_hat + diag_H % (y_hat - y) / (1 - diag_H);
  return y_alo;
}


// [[Rcpp::export]]
mat Schulz_Iterate(const mat &A, const mat &V_old) {
  // one-step Schulz iteration
  // want to find out the inverse of A
  // V_old - kth step approximate inverse of A
  
  // compute the updated V_new
  mat identity(A.n_rows, A.n_cols, fill::eye);
  mat V_new = V_old * (2 * identity - A * V_old);
  // mat V_new = 0.5 * V_old * (3 * identity - V_old * V_old);
  return V_new;
}

// [[Rcpp::export]]
double Schulz_Error(const mat &A, const mat &V_old) {
  // compute the updated V_new
  mat identity(A.n_rows, A.n_cols, fill::eye);
  double err = abs(A * V_old - identity).max();
  return err;
}


// [[Rcpp::export]]
mat ElasticNet_Taylor(const mat &A_inv, const mat &mid) {
  // compute the first order Taylor expansion of F inverse
  // A_inv - inverse of matrix A = X[, idx_new].t() * D(lambda_old) * X[, idx_new] + lambda_old * alpha * identity_matrix
  // X - design matrix, n * p
  // idx_new - index of columns of X for active set
  // delta - D(lambda_new) - D(lambda_old)
  
  // define identity matrix
  mat identity(A_inv.n_rows, A_inv.n_cols, fill::eye);
  // compute approximate F inverse
  mat F_inv = A_inv * (identity - mid * A_inv);
  return F_inv;
}
  
// You can include R code blocks in C++ files processed with sourceCpp
// (useful for testing and development). The R code will be automatically 
// run after the compilation.
//

/*** R
L = matrix(c(2, 0, 0, 6, 1, 0, -8, 5, 3), ncol = 3)
temp_add = CholeskyAdd(t(L)%*%L, L[c(1,2),c(1,2)], c(0,1), c(1,2,0))
temp_drop = CholeskyDrop(t(L)%*%L, L, c(0,1,2), c(0,1))
temp_update = CholeskyUpdate(t(L)%*%L, L[c(1,2),c(1,2)], c(0,1), c(1))

set.seed(1)
mat = matrix(rnorm(16), ncol = 4)
mat = t(mat) * mat;
mat.inv = solve(mat)
mat.12 = mat[1:2, 1:2]
# mat.added = BlockInverse_Add(mat, (1:2)-1, (3:4)-1, solve(mat.12))
mat.added = BlockInverse_Add(mat, numeric(0), 3, matrix(ncol=0,nrow=0))
mat.dropped = BlockInverse_Drop(mat.added, 1)

A = matrix(c(1,2,3,1), ncol=2)
V_old = solve(A) + matrix(rep(0.01,4),ncol=2)
V_new = Schulz_Iterate(A, V_old)
*/
