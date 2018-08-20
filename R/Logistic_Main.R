# Logistic Regression -----------------------------------------------------
setwd("E:\\Columbia_University\\Internship\\R_File\\LASSO\\")
library(glmnet)
library(ggplot2)
library(Rcpp)
sourceCpp("src/ALO_Primal.cpp")
source("R/Logistic_Functions.R")

# 1 Logistic with Intercept -----------------------------------------------


# 1.1 Set Parameters ------------------------------------------------------

n = 300
p = 600
k = 60
lambda = 10 ^ seq(log10(5E-4), log10(1E-1), length.out = 50)
lambda = sort(lambda, decreasing = TRUE)
alpha = seq(0, 1, 0.1)
param = data.frame(alpha = numeric(0),
                   lambda = numeric(0))
for (i in 1:length(alpha)) {
  for (j in 1:length(lambda)) {
    param[j + (i - 1) * length(lambda), c('alpha', 'lambda')] = c(alpha[i], lambda[j])
  }
}


# 1.2 Simulation ----------------------------------------------------------

set.seed(1234)
beta = rnorm(p, mean = 0, sd = 1)
beta[(k + 1):p] = 0
intercept = 1
X = matrix(rnorm(n * p, mean = 0, sd = sqrt(1 / k)), ncol = p)
y.linear = intercept + X %*% beta
prob = exp(y.linear) / (1 + exp(y.linear))
y = rbinom(n, 1, prob = prob)
y.factor = factor(y)

# 1.3 LOO -----------------------------------------------------------------

y.loo = matrix(ncol = dim(param)[1], nrow = n)
starttime = proc.time() # count time
library(foreach)
library(doParallel)
no_cores = detectCores() - 1
cl = makeCluster(no_cores)
registerDoParallel(cl)
for (i in 1:n) {
  # do leave one out prediction
  y.temp <-
    foreach(
      k = 1:length(alpha),
      .combine = cbind,
      .packages = 'glmnet'
    ) %dopar%
    Logistic_LOO(X, y.factor, i, alpha[k], lambda, intercept = TRUE)
  # save the prediction value
  y.loo[i,] = y.temp
  # print middle result
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
# true leave-one-out risk estimate
risk.loo = 1 / n * colSums((y.loo -
                              matrix(rep(y, dim(
                                param
                              )[1]), ncol = dim(param)[1])) ^ 2)
# record the result
result = cbind(param, risk.loo)
# save the data
save(result, y.loo,
     file = "RData/Logistic_LOO.RData")


# 1.4 ALO -----------------------------------------------------------------

load('RData/Logistic_LOO.RData')
# find the ALO prediction
y.alo = matrix(ncol = dim(param)[1], nrow = n)
starttime = proc.time() # count time
for (k in 1:length(alpha)) {
  # build the full data model
  model = glmnet(
    x = X,
    y = y,
    family = "binomial",
    alpha = alpha[k],
    lambda = lambda,
    thresh = 1E-14,
    intercept = TRUE,
    standardize = FALSE,
    maxit = 1000000
  )
  # find the prediction for each alpha value
  y.temp <- foreach(j = 1:length(lambda), .combine = cbind) %do% {
    LogisticALO(as.vector(model$beta[, j]),
                model$a0[j],
                X,
                y,
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
# true leave-one-out risk estimate
risk.alo = 1 / n * colSums((y.alo -
                              matrix(rep(y, dim(
                                param
                              )[1]), ncol = dim(param)[1])) ^ 2)
# record the result
result = cbind(result, risk.alo)

# save the data
save(result, y.loo, y.alo,
     file = "RData/Logistic_ALO")


# 1.5 Plot ----------------------------------------------------------------

load("RData/Logistic_ALO")
result$alpha = factor(result$alpha)
p = ggplot(result) +
  geom_line(aes(x = log10(lambda), y = risk.loo), col = "black", lty = 2) +
  geom_line(aes(x = log10(lambda), y = risk.alo), col = "red", lty = 2) +
  ggtitle('Logistic Regression with Intercept & Elastic Net penalty') +
  xlab("Logarithm of Lambda") +
  ylab("Risk Estimate") +
  facet_wrap( ~ alpha, nrow = 2)
bmp("figure/Logistic_with_Intercept.bmp",
    width = 1280,
    height = 720)
p
dev.off()


# 2 Logistic without Intercept --------------------------------------------


# 2.1 Set Parameters ------------------------------------------------------

n = 300
p = 600
k = 60
lambda = 10 ^ seq(log10(5E-4), log10(1E-1), length.out = 50)
lambda = sort(lambda, decreasing = TRUE)
alpha = seq(0, 1, 0.1)
param = data.frame(alpha = numeric(0),
                   lambda = numeric(0))
for (i in 1:length(alpha)) {
  for (j in 1:length(lambda)) {
    param[j + (i - 1) * length(lambda), c('alpha', 'lambda')] = c(alpha[i], lambda[j])
  }
}


# 2.2 Simulation ----------------------------------------------------------

set.seed(1234)
beta = rnorm(p, mean = 0, sd = 1)
beta[(k + 1):p] = 0
intercept = 0
X = matrix(rnorm(n * p, mean = 0, sd = sqrt(1 / k)), ncol = p)
y.linear = intercept + X %*% beta
prob = exp(y.linear) / (1 + exp(y.linear))
y = rbinom(n, 1, prob = prob)
y.factor = factor(y)


# 2.3 LOO -----------------------------------------------------------------

y.loo = matrix(ncol = dim(param)[1], nrow = n)
starttime = proc.time() # count time
library(foreach)
library(doParallel)
no_cores = detectCores() - 1
cl = makeCluster(no_cores)
registerDoParallel(cl)
for (i in 1:n) {
  # do leave one out prediction
  y.temp <-
    foreach(
      k = 1:length(alpha),
      .combine = cbind,
      .packages = 'glmnet'
    ) %dopar%
    Logistic_LOO(X, y.factor, i, alpha[k], lambda, intercept = FALSE)
  # save the prediction value
  y.loo[i,] = y.temp
  # print middle result
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
# true leave-one-out risk estimate
risk.loo = 1 / n * colSums((y.loo -
                              matrix(rep(y, dim(
                                param
                              )[1]), ncol = dim(param)[1])) ^ 2)
# record the result
result = cbind(param, risk.loo)
# save the data
save(result, y.loo,
     file = "RData/Logistic_without_InterceptLOO.RData")


# 2.4 ALO -----------------------------------------------------------------

load('RData/Logistic_without_InterceptLOO.RData')
# find the ALO prediction
y.alo = matrix(ncol = dim(param)[1], nrow = n)
starttime = proc.time() # count time
for (k in 1:length(alpha)) {
  # build the full data model
  model = glmnet(
    x = X,
    y = y,
    family = "binomial",
    alpha = alpha[k],
    lambda = lambda,
    thresh = 1E-14,
    intercept = FALSE,
    standardize = FALSE,
    maxit = 1000000
  )
  # find the prediction for each alpha value
  y.temp <- foreach(j = 1:length(lambda), .combine = cbind) %do% {
    LogisticALO(as.vector(model$beta[, j]),
                model$a0[j],
                X,
                y,
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
# true leave-one-out risk estimate
risk.alo = 1 / n * colSums((y.alo -
                              matrix(rep(y, dim(
                                param
                              )[1]), ncol = dim(param)[1])) ^ 2)
# record the result
result = cbind(result, risk.alo)

# save the data
save(result, y.loo, y.alo,
     file = "RData/Logistic_without_Intercept_ALO")


# 2.5 Plot ----------------------------------------------------------------

load("RData/Logistic_without_Intercept_ALO")
result$alpha = factor(result$alpha)
p = ggplot(result) +
  geom_line(aes(x = log10(lambda), y = risk.loo), col = "black", lty = 2) +
  geom_line(aes(x = log10(lambda), y = risk.alo), col = "red", lty = 2) +
  ggtitle('Logistic Regression with Elastic Net penalty & without Intercept') +
  xlab("Logarithm of Lambda") +
  ylab("Risk Estimate") +
  facet_wrap( ~ alpha, nrow = 2)
bmp("figure/Logistic_without_Intercept.bmp",
    width = 1280,
    height = 720)
p
dev.off()


# 3 Multinomial with Intercept --------------------------------------------


# 3.1 Set Parameters ------------------------------------------------------

n = 300
p = 200
k = 60
num_class = 5


# 3.2 Simulation ----------------------------------------------------------

set.seed(1234)
beta = matrix(rnorm(num_class * p, mean = 0, sd = 1), ncol = num_class)
beta[(k + 1):p,] = 0
intercept = rnorm(num_class, mean = 0, sd = 1)
X = matrix(rnorm(n * p, mean = 0, sd = sqrt(1 / k)), ncol = p)
y.linear = matrix(rep(intercept, n), ncol = num_class, byrow = TRUE) +
  X %*% beta
prob = exp(y.linear) / matrix(rep(rowSums(exp(y.linear)), num_class), ncol =
                                num_class)
y.mat = t(apply(prob, 1, function(x)
  rmultinom(1, 1, prob = x))) # N * K matrix (N - #obs, K - #class)
y.num = apply(y.mat == 1, 1, which) # vector
y.num.factor = factor(y.num, levels = seq(1:num_class))


# 3.3 Define Lambda and Alpha ---------------------------------------------

lambda = 10 ^ seq(-3.5, -0.5, length.out = 30)
lambda = sort(lambda, decreasing = TRUE)
alpha = seq(0, 1, 0.2)
param = data.frame(alpha = numeric(0),
                   lambda = numeric(0))
for (i in 1:length(alpha)) {
  for (j in 1:length(lambda)) {
    param[j + (i - 1) * length(lambda), c('alpha', 'lambda')] = c(alpha[i], lambda[j])
  }
}


# 3.4 LOO -----------------------------------------------------------------

y.loo = array(numeric(0), dim = c(dim(y.mat)[1],
                                  dim(y.mat)[2],
                                  dim(param)[1])) # N * K * #param
starttime = proc.time() # count time
library(foreach)
library(doParallel)
no_cores = detectCores() - 1
cl = makeCluster(no_cores)
registerDoParallel(cl)
for (i in 1:n) {
  # do leave one out prediction
  y.temp <-
    foreach(
      k = 1:length(alpha),
      .combine = cbind,
      .packages = 'glmnet'
    ) %dopar%
    Multinomial_LOO(X, y.num.factor, i, alpha[k], lambda, intercept = TRUE)
  # save the prediction value
  y.loo[i, , ] = y.temp
  # print middle result
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
# true leave-one-out risk estimate
risk.loo = vector(mode = 'double', length = dim(param)[1])
for (k in 1:dim(param)[1]) {
  risk.loo[k] = 1 / n * sum(colSums((y.loo[, , k] - y.mat) ^ 2))
}
# record the result
result = cbind(param, risk.loo)
# record time
time.loo = proc.time() - starttime
# save the data
save(result, y.loo, risk.loo,
     file = "RData/Multinomial_LOO.RData")


# 3.5 ALO in R ------------------------------------------------------------

load('RData/Multinomial_LOO.RData')
library(MASS)
# find the ALO prediction
y.alo = array(numeric(0), dim = c(dim(y.mat)[1],
                                  dim(y.mat)[2],
                                  dim(param)[1])) # N * K * #param
starttime = proc.time() # count time

# compute X.expand
X.expand = matrix(0, nrow = n * num_class, ncol = (p + 1) * num_class)
for (k in 1:num_class) {
  X.expand[seq(0, n - 1) * num_class + k, ((p + 1) * (k - 1) + 1):((p + 1) *
                                                                     k)] = cbind(1, X)
}

# compute for the leave-i-out prediction
for (k in 1:length(alpha)) {
  # build the full data model
  model = glmnet(
    x = X,
    y = y.num.factor,
    family = "multinomial",
    alpha = alpha[k],
    lambda = lambda,
    thresh = 1E-14,
    intercept = TRUE,
    standardize = FALSE,
    maxit = 1000000
  )
  # find the prediction for each alpha value
  for (j in 1:length(lambda)) {
    # extract beta under all of the class
    beta.temp = matrix(nrow = p + 1, ncol = num_class)
    beta.temp[1,] = as.vector(model$a0[, j])
    for (i in 1:num_class) {
      beta.temp[2:(p + 1), i] = as.matrix(model$beta[[i]])[, j]
    }
    # for(i in 1:num_class) {
    #   beta.temp[,i]=beta.temp[,i]-beta.temp[,num_class]
    # }
    beta.temp = as.vector(beta.temp)
    
    # find the active set
    E = which(beta.temp != 0)
    
    # compute matrix A(beta) and D(beta)
    A = as.vector(matrix(nrow = n * num_class, ncol = 1))
    D = matrix(0, nrow = n * num_class, ncol = n * num_class)
    for (i in 1:n) {
      idx = ((i - 1) * num_class + 1):(i * num_class)
      A[idx] = exp(X.expand[idx,] %*% beta.temp)
      A[idx] = A[idx] / sum(A[idx])
      D[idx, idx] = diag(A[idx]) - A[idx] %*% t(A[idx])
    }
    # idx=seq(1,n)*num_class
    # A=A[-idx]
    # D=D[-idx,-idx]
    
    # compute R_diff2
    R_diff2 = matrix(0,
                     nrow = (p + 1) * num_class,
                     ncol = (p + 1) * num_class)
    diag(R_diff2) = n * lambda[j] * (1 - alpha[k])
    R_diff2[1 + seq(0, num_class - 1) * (p + 1), 1 + seq(0, num_class -
                                                           1) * (p + 1)] = 0
    
    # compute matrix K(beta) and its inverse
    K = t(X.expand[, E]) %*% D %*% X.expand[, E] + R_diff2[E, E]
    # K = t(X.expand[-idx, E]) %*% D %*% X.expand[-idx, E] + R_diff2[E, E]
    K.inv = ginv(K)
    
    # do leave-i-out prediction
    for (i in 1:n) {
      # find the X_i and y_i
      X.i = X.expand[((i - 1) * num_class + 1):(i * num_class),]
      # X.i = X.expand[((i - 1) * num_class + 1):(i * num_class-1), ]
      y.i = y.mat[i, ]
      # y.i=y.mat[i,1:(num_class-1)]
      
      # find the A_i
      A.i = A[((i - 1) * num_class + 1):(i * num_class)]
      # A.i = A[((i - 1) * (num_class-1) + 1):(i * (num_class-1))]
      
      # compute X_i * K.inv * X_i^T
      XKX = X.i[, E] %*% K.inv %*% t(X.i[, E])
      
      # compute the inversion of diag(A)-A*A^T
      middle.inv = ginv(diag(A.i) - A.i %*% t(A.i))
      # middle.inv = solve(diag(A.i)) -
      #   solve(diag(A.i)) %*% A.i %*% solve(-1 +
      #                                        t(A.i) %*% solve(diag(A.i)) %*% A.i) %*% t(A.i) %*% solve(diag(A.i))
      
      # compute the leave-i-out prediction
      y.alo.linear = X.i %*% beta.temp + XKX %*% (A.i - y.i) -
        XKX %*% ginv(-middle.inv + XKX) %*% XKX %*% (A.i - y.i)
      y.alo.exp = exp(y.alo.linear)
      # y.alo.exp = c(exp(y.alo.linear),1)
      y.alo[i, , (k - 1) * length(lambda) + j] = y.alo.exp / sum(y.alo.exp)
    }
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
# approximate leave-one-out risk estimate
risk.alo = vector(mode = 'double', length = dim(param)[1])
for (k in 1:dim(param)[1]) {
  risk.alo[k] = 1 / n * sum(colSums((y.alo[, , k] - y.mat) ^ 2))
}
# record the result
result = cbind(result, risk.alo)
# record time
time.alo = proc.time() - starttime
# save the data
save(result, y.loo, y.alo, risk.loo, risk.alo,
     file = "RData/Multinomial_ALO")


# 3.6 Plot in R -----------------------------------------------------------

load("RData/Multinomial_ALO")
result$alpha = factor(result$alpha)
plt = ggplot(result) +
  geom_line(aes(x = log10(lambda), y = risk.loo), col = "black", lty = 2) +
  geom_line(aes(x = log10(lambda), y = risk.alo), col = "red", lty = 2) +
  ggtitle('Logistic Regression with Intercept & Elastic Net penalty') +
  xlab("Logarithm of Lambda") +
  ylab("Risk Estimate") +
  facet_wrap( ~ alpha, nrow = 2)
bmp("figure/Multinomial_in_R.bmp",
    width = 1280,
    height = 720)
plt
dev.off()



# 3.7 ALO in Cpp ----------------------------------------------------------

load('RData/Multinomial_LOO.RData')
# find the ALO prediction
y.alo = array(numeric(0), dim = c(dim(y.mat)[1],
                                  dim(y.mat)[2],
                                  dim(param)[1])) # N * K * #param
starttime = proc.time() # count time
# compute X.expand
X.expand = matrix(0, nrow = n * num_class, ncol = (p + 1) * num_class)
for (k in 1:num_class) {
  X.expand[seq(0, n - 1) * num_class + k, ((p + 1) * (k - 1) + 1):((p + 1) *
                                                                     k)] = cbind(1, X)
}

# compute for the leave-i-out prediction
for (k in 1:length(alpha)) {
  # build the full data model
  model = glmnet(
    x = X,
    y = y.num.factor,
    family = "multinomial",
    alpha = alpha[k],
    lambda = lambda,
    thresh = 1E-14,
    intercept = TRUE,
    standardize = FALSE,
    maxit = 1000000
  )
  # find the prediction for each alpha value
  for (j in 1:length(lambda)) {
    # extract beta under all of the class
    beta.temp = matrix(nrow = p + 1, ncol = num_class)
    beta.temp[1,] = as.vector(model$a0[, j])
    for (i in 1:num_class) {
      beta.temp[2:(p + 1), i] = as.matrix(model$beta[[i]])[, j]
    }
    beta.temp = as.vector(beta.temp)
    
    # leave-i-out prediction
    y.alo[, , (k - 1) * length(lambda) + j] =
      MultinomialALO(beta.temp,
                     TRUE,
                     cbind(1, X),
                     X.expand,
                     y.mat,
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
# approximate leave-one-out risk estimate
risk.alo = vector(mode = 'double', length = dim(param)[1])
for (k in 1:dim(param)[1]) {
  risk.alo[k] = 1 / n * sum(colSums((y.alo[, , k] - y.mat) ^ 2))
}
# record the result
result = cbind(result, risk.alo)
# record time
time.alo = proc.time() - starttime
# save the data
save(result, y.loo, y.alo, time.loo, time.alo,
     file = "RData/Multinomial_ALO_Cpp")

# 3.8 Plot in Cpp ---------------------------------------------------------

load("RData/Multinomial_ALO_Cpp")
result$alpha = factor(result$alpha)
plt = ggplot(result) +
  geom_line(aes(x = log10(lambda), y = risk.loo), col = "black", lty = 2) +
  geom_line(aes(x = log10(lambda), y = risk.alo), col = "red", lty = 2) +
  ggtitle('Logistic Regression with Intercept & Elastic Net penalty') +
  xlab("Logarithm of Lambda") +
  ylab("Risk Estimate") +
  facet_wrap( ~ alpha, nrow = 2)
bmp("figure/Multinomial_in_Cpp.bmp",
    width = 1280,
    height = 720)
plt
dev.off()