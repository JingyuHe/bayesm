library(coda)
library(bayesmtest)

R = 5000

set.seed(6426)
simnegbin = function(X, beta, alpha) {
      # Simulate from the Negative Binomial Regression
      lambda = exp(X%*%beta)
      y = NULL
      for (j in 1:length(lambda)) { y = c(y, rnbinom(1, mu=lambda[j], size=alpha)) }
      return(y)
}

nobs = 5000
nvar = 10 # Number of X variables
alpha = 1
Vbeta = diag(nvar)*0.01

# Construct the regdata (containing X)
simnegbindata = NULL
beta = rnorm(nvar)
X = cbind(rep(1,nobs), rnorm(nobs,mean=2,sd=0.5), rnorm(nobs,mean=2,sd=0.5), rnorm(nobs,mean=2,sd=0.5), rnorm(nobs,mean=2,sd=0.5), rnorm(nobs,mean=2,sd=0.5), rnorm(nobs,mean=2,sd=0.5), rnorm(nobs,mean=2,sd=0.5), rnorm(nobs,mean=2,sd=0.5), rnorm(nobs,mean=2,sd=0.5))
simnegbindata = list(y=simnegbin(X,beta,alpha), X=X, beta=beta)
Data1 = simnegbindata
Mcmc1 = list(R=R)




out1 = bayesm::rnegbinRw(Data=Data1, Mcmc=list(R=R))
cat("Summary of alpha/beta draw", fill=TRUE)
summary(out1$alphadraw, tvalues=alpha)
summary(out1$betadraw, tvalues=beta)




out2 = bayesmtest::rnegbinRw(Data=Data1, Mcmc = list(R=R))
cat("Summary of alpha/beta draw", fill=TRUE)
summary(out2$alphadraw, tvalues=alpha)
summary(out2$betadraw, tvalues=beta)


apply(out1$betadraw, 2, effectiveSize)
apply(out2$betadraw, 2, effectiveSize)
