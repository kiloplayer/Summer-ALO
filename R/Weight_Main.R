# Analysis for the Effect of Weights --------------------------------------


# 1 Elastic Net Case ------------------------------------------------------

rm(list = ls())
setwd("E:\\Columbia_University\\Internship\\R_File\\LASSO\\")
library(glmnet)
library(ggplot2)
library(Rcpp)
sourceCpp("src/ALO_Primal.cpp")
source("R/ElasticNet_Functions.R")

# 1.1 Simulation ----------------------------------------------------------

n = 300
p = 500
k = 100
set.seed(1234)
beta = rnorm(p, mean = 0, sd = 1)
beta[(k + 1):p] = 0
intercept = 1
X = matrix(rnorm(n * p, mean = 0, sd = sqrt(1 / k)), ncol = p)
sigma = rnorm(n, mean = 0, sd = 0.5)
y = intercept + X %*% beta + sigma
weights = seq(1, n)
index = which(y >= 0)
y[index] = sqrt(y[index])
y[-index] = -sqrt(-y[-index])


# 1.2 Set Parameters ------------------------------------------------------

sd.y = as.numeric(sqrt(var(y) / length(y) * (length(y) - 1)))
y.scaled = y / sd.y
X.scaled = X / sd.y
model = glmnet(
  x = X.scaled,
  y = y.scaled,
  family = "gaussian",
  weights = weights,
  alpha = 1,
  thresh = 1E-14,
  intercept = TRUE,
  standardize = FALSE,
  nlambda = 60,
  maxit = 1000000
)
lambda = sort(model$lambda[1:50] * sd.y ^ 2, decreasing = TRUE)
# log10.lambda = seq(log10(1E-3), log10(5E-2), length.out = 50)
# lambda = 10 ^ log10.lambda
# lambda = sort(lambda, decreasing = TRUE)
alpha = seq(0, 1, 0.2)
param = data.frame(alpha = numeric(0),
                   lambda = numeric(0))
for (i in 1:length(alpha)) {
  for (j in 1:length(lambda)) {
    param[j + (i - 1) * length(lambda), c('alpha', 'lambda')] = c(alpha[i], lambda[j])
  }
}


# 1.3 Leave-One-Out -------------------------------------------------------

y.loo = matrix(ncol = dim(param)[1], nrow = n)
library(foreach)
library(doParallel)
no_cores = detectCores() - 1
cl = makeCluster(no_cores)
registerDoParallel(cl)
starttime = proc.time()
for (i in 1:n) {
  y.loo[i, ] <-
    foreach(
      k = 1:length(alpha),
      .combine = cbind,
      .packages = 'glmnet'
    ) %dopar%
    ElasticNet_LOO_Weight(X, y, weights, i,
                          alpha[k], lambda, intercept = TRUE)
  if (i %% 10 == 0)
    print(
      paste(
        i,
        " samples have beed calculated. ",
        "On average, every sample needs ",
        round((proc.time() - starttime)[3] / i, 2),
        " seconds."
      )
    )
}
stopCluster(cl)
risk.loo = 1 / n * apply(y.loo, 2, function(x)
  sum((x - y) ^ 2))
result = cbind(param, risk.loo)
save(result, y.loo,
     file = "RData/ElasticNet_LOO_Weight.RData")

# 1.4 ALO -----------------------------------------------------------------

load("RData/ElasticNet_LOO_Weight.RData")
sd.y = as.numeric(sqrt(var(y) / length(y) * (length(y) - 1)))
y.scaled = y / sd.y
X.scaled = X / sd.y
y.alo = matrix(ncol = dim(param)[1], nrow = n)
XWX = t(cbind(1, X)) %*% diag(weights) %*% cbind(1, X)
starttime = proc.time()
for (k in 1:length(alpha)) {
  model = glmnet(
    x = X.scaled,
    y = y.scaled,
    family = "gaussian",
    weights = weights,
    alpha = alpha[k],
    lambda = lambda / sd.y ^ 2,
    thresh = 1E-14,
    intercept = TRUE,
    standardize = FALSE,
    maxit = 1000000
  )
  beta.temp = rbind(model$a0 * sd.y, as.matrix(model$beta))
  # find the prediction for each alpha value
  y.temp <- foreach(j = 1:length(lambda), .combine = cbind) %do% {
    ElasticNetALO_Weight(beta.temp[, j],
                         TRUE,
                         cbind(1, X),
                         y,
                         weights,
                         XWX,
                         lambda[j],
                         alpha[k])
  }
  y.alo[, ((k - 1) * length(lambda) + 1):(k * length(lambda))] = y.temp
  # print middle result
  print(
    paste(
      k,
      " alphas have beed calculated. ",
      "On average, every alpha needs ",
      round((proc.time() - starttime)[3] / k, 2),
      " seconds."
    )
  )
}
risk.alo = 1 / n * apply(y.alo, 2, function(x)
  sum((x - y) ^ 2))
result = cbind(result, risk.alo)
save(result, y.loo,
     file = "RData/ElasticNet_ALO_Weight.RData")


# 1.5 Plot ----------------------------------------------------------------

load("RData/ElasticNet_ALO_Weight.RData")
result$alpha = factor(result$alpha)
plt = ggplot(result) +
  geom_line(aes(x = log10(lambda), y = risk.loo), lty = 2) +
  geom_line(aes(x = log10(lambda), y = risk.alo), col = "red", lty = 2) +
  facet_wrap( ~ alpha, nrow = 2)
bmp("figure/ElasticNet_ALO_Weight.bmp",
    width = 1280,
    height = 720)
plt
dev.off()


# 2 Logistic Regression Case (Binomial) -----------------------------------

rm(list = ls())
setwd("E:\\Columbia_University\\Internship\\R_File\\LASSO\\")
library(glmnet)
library(ggplot2)
library(Rcpp)
sourceCpp("src/ALO_Primal.cpp")
source("R/ElasticNet_Functions.R")

# 2.1 Simulation ----------------------------------------------------------

n = 300
p = 600
k = 100
set.seed(1234)
beta = rnorm(p, mean = 0, sd = 1)
beta[(k + 1):p] = 0
intercept = 1
X = matrix(rnorm(n * p, mean = 0, sd = sqrt(1 / k)), ncol = p)
y.linear = intercept + X %*% beta
prob = exp(y.linear) / (1 + exp(y.linear))
y = rbinom(n, 1, prob = prob)
y.factor = factor(y)
weights = seq(1, n)

# 2.2 Set Parameters ------------------------------------------------------

model = glmnet(
  x = X,
  y = y.factor,
  family = "binomial",
  weights = weights,
  alpha = 1,
  thresh = 1E-14,
  intercept = TRUE,
  standardize = FALSE,
  nlambda = 50,
  maxit = 1000000
)
lambda = sort(model$lambda, decreasing = TRUE)
lambda = 10 ^ seq(-1, -4, length.out = 50)
alpha = seq(0, 1, 0.2)
param = data.frame(alpha = numeric(0),
                   lambda = numeric(0))
for (i in 1:length(alpha)) {
  for (j in 1:length(lambda)) {
    param[j + (i - 1) * length(lambda), c('alpha', 'lambda')] = c(alpha[i], lambda[j])
  }
}


# 2.3 Leave-One-Out -------------------------------------------------------

y.loo = matrix(ncol = dim(param)[1], nrow = n)
library(foreach)
library(doParallel)
no_cores = detectCores() - 1
cl = makeCluster(no_cores)
registerDoParallel(cl)
starttime = proc.time()
for (i in 1:n) {
  y.loo[i, ] <-
    foreach(
      k = 1:length(alpha),
      .combine = cbind,
      .packages = 'glmnet'
    ) %dopar%
    Logistic_LOO_Weight(X, y, weights, i,
                        alpha[k], lambda, intercept = TRUE)
  if (i %% 10 == 0)
    print(
      paste(
        i,
        " samples have beed calculated. ",
        "On average, every sample needs ",
        round((proc.time() - starttime)[3] / i, 2),
        " seconds."
      )
    )
}
stopCluster(cl)
risk.loo = 1 / n * apply(y.loo, 2, function(x)
  sum((x - y) ^ 2))
result = cbind(param, risk.loo)
save(result, y.loo,
     file = "RData/Logistic_LOO_Weight.RData")

# 2.4 ALO -----------------------------------------------------------------

load("RData/Logistic_LOO_Weight.RData")
y.alo = matrix(ncol = dim(param)[1], nrow = n)
starttime = proc.time()
for (k in 1:length(alpha)) {
  model = glmnet(
    x = X,
    y = y.factor,
    family = "binomial",
    weights = weights,
    alpha = alpha[k],
    lambda = lambda,
    thresh = 1E-14,
    intercept = TRUE,
    standardize = FALSE,
    maxit = 1000000
  )
  beta.temp = rbind(model$a0, as.matrix(model$beta))
  # find the prediction for each alpha value
  y.temp <- foreach(j = 1:length(lambda), .combine = cbind) %do% {
    LogisticALO_Weight(beta.temp[, j],
                       TRUE,
                       cbind(1, X),
                       y,
                       weights,
                       lambda[j],
                       alpha[k])
  }
  y.alo[, ((k - 1) * length(lambda) + 1):(k * length(lambda))] = y.temp
  # print middle result
  print(
    paste(
      k,
      " alphas have beed calculated. ",
      "On average, every alpha needs ",
      round((proc.time() - starttime)[3] / k, 2),
      " seconds."
    )
  )
}
risk.alo = 1 / n * apply(y.alo, 2, function(x)
  sum((x - y) ^ 2))
result = cbind(result, risk.alo)
save(result, y.loo,
     file = "RData/Logistic_ALO_Weight.RData")


# 2.5 Plot ----------------------------------------------------------------

load("RData/Logistic_ALO_Weight.RData")
result$alpha = factor(result$alpha)
plt = ggplot(result) +
  geom_line(aes(x = log10(lambda), y = risk.loo), lty = 2) +
  geom_line(aes(x = log10(lambda), y = risk.alo), col = "red", lty = 2) +
  facet_wrap( ~ alpha, nrow = 2)
bmp("figure/Logistic_ALO_Weight.bmp",
    width = 1280,
    height = 720)
plt
dev.off()


# 3 Multinomial Regression Case -------------------------------------------

rm(list = ls())
setwd("E:\\Columbia_University\\Internship\\R_File\\LASSO\\")
library(glmnet)
library(ggplot2)
library(Rcpp)
sourceCpp("src/ALO_Primal.cpp")
source("R/ElasticNet_Functions.R")

# 3.1 Simulation ----------------------------------------------------------

n = 300
p = 200
k = 100
num_class = 5
set.seed(1234)
beta = matrix(rnorm(num_class * p, mean = 0, sd = 1), ncol = num_class)
beta[(k + 1):p, ] = 0
intercept = rnorm(num_class, mean = 0, sd = 1)
X = matrix(rnorm(n * p, mean = 0, sd = sqrt(1 / k)), ncol = p)
y.linear = t(apply(X %*% beta, 1, function(x)
  x + intercept))
prob = t(apply(y.linear, 1, function(x)
  exp(x) / sum(exp(x))))
y.mat = t(apply(prob, 1, function(x)
  rmultinom(1, 1, prob = x)))
y.num = apply(y.mat == 1, 1, which) # vector
y.num.factor = factor(y.num, levels = seq(1:num_class))
weights = seq(1, n)

# 3.2 Set Parameters ------------------------------------------------------

model = glmnet(
  x = X,
  y = y.num.factor,
  family = "multinomial",
  weights = weights,
  alpha = 1,
  thresh = 1E-14,
  intercept = TRUE,
  standardize = FALSE,
  nlambda = 50,
  maxit = 1000000
)
lambda = sort(model$lambda, decreasing = TRUE)
lambda = 10 ^ seq(-1, -3.5, length.out = 50)
alpha = seq(0, 1, 0.2)
param = data.frame(alpha = numeric(0),
                   lambda = numeric(0))
for (i in 1:length(alpha)) {
  for (j in 1:length(lambda)) {
    param[j + (i - 1) * length(lambda), c('alpha', 'lambda')] = c(alpha[i], lambda[j])
  }
}


# 3.3 Leave-One-Out -------------------------------------------------------

y.loo = array(numeric(0),
              dim = c(n, num_class, dim(param)[1])) # N * K * #param
library(foreach)
library(doParallel)
no_cores = detectCores() - 1
cl = makeCluster(no_cores)
registerDoParallel(cl)
starttime = proc.time()
for (i in 1:n) {
  y.loo[i, , ] <-
    foreach(
      k = 1:length(alpha),
      .combine = cbind,
      .packages = 'glmnet'
    ) %dopar%
    Multinomial_LOO_Weight(X, y.num.factor, weights, i,
                           alpha[k], lambda, intercept = TRUE)
  if (i %% 1 == 0)
    print(
      paste(
        i,
        " samples have beed calculated. ",
        "On average, every sample needs ",
        round((proc.time() - starttime)[3] / i, 2),
        " seconds."
      )
    )
}
stopCluster(cl)
risk.loo = vector(mode = 'double', length = dim(param)[1])
for (k in 1:dim(param)[1]) {
  risk.loo[k] = 1 / n * sum(colSums((y.loo[, , k] - y.mat) ^ 2))
}
result = cbind(param, risk.loo)
save(result, y.loo,
     file = "RData/Multinomial_LOO_Weight.RData")

# 3.4 ALO -----------------------------------------------------------------

load("RData/Multinomial_LOO_Weight.RData")
y.alo = array(numeric(0),
              dim = c(n, num_class, dim(param)[1])) # N * K * #param
X.expand = matrix(0, nrow = n * num_class, ncol = (p + 1) * num_class)
for (k in 1:num_class) {
  X.expand[seq(0, n - 1) * num_class + k,
           ((p + 1) * (k - 1) + 1):((p + 1) * k)] = cbind(1, X)
}
starttime = proc.time()
for (k in 1:length(alpha)) {
  model = glmnet(
    x = X,
    y = y.num.factor,
    family = "multinomial",
    weights = weights,
    alpha = alpha[k],
    lambda = lambda,
    thresh = 1E-14,
    intercept = TRUE,
    standardize = FALSE,
    maxit = 1000000
  )
  for (j in 1:length(lambda)) {
    beta.temp = matrix(nrow = p + 1, ncol = num_class)
    beta.temp[1, ] = as.vector(model$a0[, j])
    for (i in 1:num_class) {
      beta.temp[2:(p + 1), i] = as.matrix(model$beta[[i]])[, j]
    }
    beta.temp = as.vector(beta.temp)
    
    # leave-i-out prediction
    y.alo[, , (k - 1) * length(lambda) + j] =
      MultinomialALO_Weight(beta.temp,
                            TRUE,
                            cbind(1, X),
                            X.expand,
                            y.mat,
                            weights,
                            lambda[j],
                            alpha[k])
    
    # print result
    print(paste('#alpha =', k, ', #lambda =', j))
  }
  # print middle result
  print(
    paste(
      k,
      " alphas have beed calculated. ",
      "On average, every alpha needs ",
      round((proc.time() - starttime)[3] / k, 2),
      " seconds."
    )
  )
}
risk.alo = vector(mode = 'double', length = dim(param)[1])
for (k in 1:dim(param)[1]) {
  risk.alo[k] = 1 / n * sum(colSums((y.alo[, , k] - y.mat) ^ 2))
}
result = cbind(result, risk.alo)
save(result, y.loo,
     file = "RData/Multinomial_ALO_Weight.RData")


# 3.5 Plot ----------------------------------------------------------------

load("RData/Multinomial_ALO_Weight.RData")
result$alpha = factor(result$alpha)
plt = ggplot(result) +
  geom_line(aes(x = log10(lambda), y = risk.loo), lty = 2) +
  geom_line(aes(x = log10(lambda), y = risk.alo), col = "red", lty = 2) +
  facet_wrap( ~ alpha, nrow = 2)
bmp("figure/Multinomial_ALO_Weight.bmp",
    width = 1280,
    height = 720)
plt
dev.off()
