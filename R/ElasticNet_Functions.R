Elastic_Net_LOO = function(X, y, i, alpha, lambda, intercept = TRUE) {
  # find out the dimension of X
  n = dim(X)[1]
  p = dim(X)[2]
  # compute the scale parameter for y
  sd.y = sqrt(var(y[-i]) * (n - 1) / (n - 2))
  y.scaled = y / sd.y
  X.scaled = X / sd.y
  # build the model
  model = glmnet(
    x = X.scaled[-i, ],
    y = y.scaled[-i],
    family = "gaussian",
    alpha = alpha,
    lambda = lambda / sd.y ^ 2 / (n - 1) * n,
    thresh = 1E-14,
    intercept = intercept,
    standardize = FALSE,
    maxit = 1000000
  )
  # prediction
  beta.hat = as.matrix(model$beta)
  intercept.hat = model$a0 * sd.y
  y.loo = vector(mode = "double", length = length(lambda))
  for (k in 1:length(lambda))
    y.loo[k] = X[i, ] %*% beta.hat[, k] + intercept.hat[k]
  return(y.loo)
}

ElasticNet_LOO_Weight = function(X, y, weights, i,
                                 alpha, lambda, intercept = TRUE) {
  # find out the dimension of X
  n = dim(X)[1]
  p = dim(X)[2]
  # compute the scale parameter for y
  sd.y = sqrt(var(y[-i]) * (n - 2) / (n - 1))
  y.scaled = y / sd.y
  X.scaled = X / sd.y
  # build the model
  model = glmnet(
    x = X.scaled[-i,],
    y = y.scaled[-i],
    family = "gaussian",
    weights = weights[-i],
    alpha = alpha,
    lambda = lambda / sd.y ^ 2 / sum(weights[-i]) * sum(weights),
    thresh = 1E-14,
    intercept = intercept,
    standardize = FALSE,
    maxit = 1000000
  )
  # prediction
  beta.hat = as.matrix(model$beta)
  intercept.hat = model$a0 * sd.y
  y.loo = vector(mode = "double", length = length(lambda))
  for (k in 1:length(lambda))
    y.loo[k] = X[i, ] %*% beta.hat[, k] + intercept.hat[k]
  return(y.loo)
}

Poisson_LOO = function(X, y, i, alpha, lambda, intercept = TRUE) {
  # find out the dimension of X
  n = dim(X)[1]
  p = dim(X)[2]
  # build the model
  model = glmnet(
    x = X[-i,],
    y = y[-i],
    family = "poisson",
    alpha = alpha,
    lambda = lambda / (n - 1) * n,
    thresh = 1E-14,
    intercept = intercept,
    # standardize = FALSE,
    standardize = TRUE,
    maxit = 1000000
  )
  # prediction
  beta = as.matrix(model$beta)
  intercept = model$a0
  y.loo = vector(mode = "double", length = length(lambda))
  for (k in 1:length(lambda)) {
    y.linear = X[i, ] %*% beta[, k] + intercept[k]
    y.loo[k] = exp(y.linear)
  }
  return(y.loo)
}

Logistic_LOO_Weight = function(X, y, weights,
                               i, alpha, lambda, intercept = TRUE) {
  # find out the dimension of X
  n = dim(X)[1]
  p = dim(X)[2]
  # build the model
  model = glmnet(
    x = X[-i,],
    y = y[-i],
    family = "binomial",
    weights = weights[-i],
    alpha = alpha,
    lambda = lambda / sum(weights[-i]) * sum(weights),
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


Multinomial_LOO_Weight = function(X, y, weights, i,
                                  alpha, lambda, intercept = TRUE) {
  # find out the number of classes
  num_class = length(levels(y))
  # build the model
  model = glmnet(
    x = X[-i, ],
    y = y[-i],
    family = "multinomial",
    alpha = alpha,
    lambda = lambda / sum(weights[-i]) * sum(weights),
    thresh = 1E-14,
    intercept = intercept,
    standardize = FALSE,
    maxit = 1000000
  )
  # prediction
  y.exp = matrix(nrow = num_class, ncol = length(lambda))
  for (k in 1:num_class) {
    beta = as.matrix(model$beta[[k]])
    intercept = as.vector(model$a0[k,])
    y.linear = as.vector(X[i,] %*% beta) + intercept
    y.exp[k, ] = exp(y.linear)
  }
  y.loo = apply(y.exp, 2, function(x)
    x / sum(x))
  return(y.loo)
}

MGaussian_LOO = function(X, y, i, alpha, lambda, intercept) {
  # find out the dimension of X
  n = dim(X)[1]
  p = dim(X)[2]
  num_class = dim(y)[2]
  # build the model
  model = glmnet(
    x = X[-i,],
    y = y[-i,],
    family = "mgaussian",
    alpha = alpha,
    lambda = lambda / (n - 1) * n,
    thresh = 1E-14,
    intercept = intercept,
    standardize = FALSE,
    maxit = 1000000
  )
  # prediction
  y.loo = matrix(nrow = num_class, ncol = length(lambda))
  for (k in 1:num_class) {
    beta = as.matrix(model$beta[[k]])
    intercept = as.vector(model$a0[k,])
    y.loo[k, ] = as.vector(X[i,] %*% beta) + intercept
  }
  return(y.loo)
}