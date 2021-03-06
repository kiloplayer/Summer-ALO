#### 07/13/2018 ####
library(glmnet)
library(MASS)
library(Rcpp)
sourceCpp("lassoALO.cpp")

# setup
n = 300
p = 400
k = 60
a = 0.5
true_beta = rnorm(p, 0, 1)
true_beta[-(1:k)] = 0

# misspecification example
X_mis = matrix(rnorm(n * p, 0, sqrt(1 / k)), n, p)
y_mis = X_mis %*% true_beta + rnorm(n, 0, 0.5)
y_mis[y_mis >= 0] = sqrt(y_mis[y_mis >= 0])
y_mis[y_mis < 0] = -sqrt(-y_mis[y_mis < 0])
sd_mis = c(sd(y_mis) * sqrt(n - 1) / sqrt(n))
y_mis = y_mis / sd_mis

# heavy-tailed noise example
X_hvy = matrix(rnorm(n * p, 0, sqrt(1 / k)), n, p)
t_noise = 0.5 * scale(rt(n, 3), center = F)
y_hvy = X_hvy %*% true_beta + t_noise
sd_hvy = c(sd(y_hvy) * sqrt(n - 1) / sqrt(n))
y_hvy = y_hvy / sd_hvy

# correlated design example
C = toeplitz(0.8^(1:p))
X_cor = mvrnorm(n, rep(0, p), C / k)
y_cor = X_cor %*% true_beta + rnorm(n, 0, 0.5)
sd_cor = c(sd(y_cor) * sqrt(n - 1) / sqrt(n))
y_cor = y_cor / sd_cor

# search grid
# tune_param = 3.16 * 10^seq(-2, -3, length.out = 25)
# tune_param = 10^seq(-1.5, -2.5, length.out = 25)
tune_param = 10^seq(-3, -1.5, length.out = 25)
# tune_param = 10^seq(-5, -2.5, length.out = 25)

# LOOCV with glmnet
CV_mis = cv.glmnet(X_mis, y_mis, lambda = tune_param, nfolds = n, grouped = F, 
                   intercept = F, standardize = F, alpha = a)

CV_hvy = cv.glmnet(X_hvy, y_hvy, lambda = tune_param, nfolds = n, grouped = F, 
                   intercept = F, standardize = F, alpha = a)

CV_cor= cv.glmnet(X_cor, y_cor, lambda = tune_param, nfolds = n, grouped = F, 
                  intercept = F, standardize = F, alpha = a)

# ALOs
alo_mis = foreach(i = 1:length(CV_mis$glmnet.fit$lambda), .combine = c) %do% {
  temp_lambda = CV_mis$glmnet.fit$lambda[i]
  temp_beta = as.vector(coef.cv.glmnet(CV_mis, s = temp_lambda)[-1])
  alo_pred = elsnetALO(temp_beta, X_mis, y_mis, temp_lambda, a)
  
  return(mean((alo_pred - y_mis)^2))
}
alo_hvy = foreach(i = 1:length(CV_hvy$glmnet.fit$lambda), .combine = c) %do% {
  temp_lambda = CV_hvy$glmnet.fit$lambda[i]
  temp_beta = as.vector(coef.cv.glmnet(CV_hvy, s = temp_lambda)[-1])
  alo_pred = elsnetALO(temp_beta, X_hvy, y_hvy, temp_lambda, a)
  
  return(mean((alo_pred - y_hvy)^2))
}
alo_cor = foreach(i = 1:length(CV_cor$glmnet.fit$lambda), .combine = c) %do% {
  temp_lambda = CV_cor$glmnet.fit$lambda[i]
  temp_beta = as.vector(coef.cv.glmnet(CV_cor, s = temp_lambda)[-1])
  alo_pred = elsnetALO(temp_beta, X_cor, y_cor, temp_lambda, a)
  
  return(mean((alo_pred - y_cor)^2))
}

# plots
par(mfrow = c(1, 3))
plot(rev(CV_mis$cvm), xlab = "lambda", ylab = paste("Elastic Net risk, a =", a), type = "l", lwd = 2, col = "darkorange")
lines(rev(alo_mis), type = "b", pch = 4, lwd = 2, col = 4)

plot(rev(CV_hvy$cvm), xlab = "lambda", ylab = paste("Elastic Net risk, a =", a), type = "l", lwd = 2, col = "darkorange")
lines(rev(alo_hvy), type = "b", pch = 4, lwd = 2, col = 4)

plot(rev(CV_cor$cvm), xlab = "lambda", ylab = paste("Elastic Net risk, a =", a), type = "l", lwd = 2, col = "darkorange")
lines(rev(alo_cor), type = "b", pch = 4, lwd = 2, col = 4)
