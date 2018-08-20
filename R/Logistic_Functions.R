Logistic_LOO = function(X, y, i, alpha, lambda, intercept = TRUE) {
  # find out the dimension of X
  n = dim(X)[1]
  p = dim(X)[2]
  # build the model
  model = glmnet(
    x = X[-i,],
    y = y[-i],
    family = "binomial",
    alpha = alpha,
    lambda = lambda / (n - 1) * n,
    thresh = 1E-14,
    intercept = intercept,
    standardize = FALSE,
    maxit = 1000000
  )
  # prediction
  beta = as.matrix(model$beta)
  intercept = model$a0
  y.loo = vector(mode = "double", length = length(lambda))
  for (k in 1:length(lambda)) {
    linear.pred = X[i, ] %*% beta[, k] + intercept[k]
    y.loo[k] = exp(linear.pred) / (1 + exp(linear.pred))
  }
  return(y.loo)
}

Multinomial_LOO = function(X, y, i, alpha, lambda, intercept = TRUE) {
  # find out the dimension of X
  n = dim(X)[1]
  p = dim(X)[2]
  num_class = length(levels(y))
  # build the model
  model = glmnet(
    x = X[-i,],
    y = y[-i],
    family = "multinomial",
    alpha = alpha,
    lambda = lambda / (n - 1) * n,
    thresh = 1E-14,
    intercept = intercept,
    standardize = FALSE,
    maxit = 1000000
  )
  # prediction
  y.exp = matrix(nrow = num_class, ncol = length(lambda))
  for (k in 1:num_class) {
    beta = as.matrix(model$beta[[k]])
    intercept = as.vector(model$a0[k, ])
    y.linear = as.vector(X[i, ] %*% beta) + intercept
    y.exp[k,] = exp(y.linear)
  }
  y.loo = y.exp / matrix(rep(colSums(y.exp), num_class), nrow = num_class, byrow =
                           TRUE)
  return(y.loo)
}