library(bayesmtest)
library(coda)

set.seed(64324)

R = 5000
# Simulate from the Negative Binomial Regression
simnegbin = function(X, beta, alpha) {
      lambda = exp(X%*%beta)
      y = NULL
      for (j in 1:length(lambda)) {y = c(y, rnbinom(1, mu=lambda[j], size=alpha)) }
      return(y)
      }

nreg = 100
T=5

nobs = nreg*T
nvar = 2
nz=2
# Number of cross sectional units
# Number of observations per unit
# Number of X variables
# Number of Z variables
## Construct the Z matrix
Z = cbind(rep(1,nreg), rnorm(nreg,mean=1,sd=0.125))
Delta = cbind(c(4,2), c(0.1,-1))
alpha = 5
Vbeta = rbind(c(2,1), c(1,2))
## Construct the regdata (containing X)
simnegbindata = NULL
for (i in 1:nreg) {
    betai = as.vector(Z[i,]%*%Delta) + chol(Vbeta)%*%rnorm(nvar)
    X = cbind(rep(1,T),matrix(rnorm(T*(nvar-1),mean=2,sd=0.25),ncol = (nvar - 1)))
    simnegbindata[[i]] = list(y=simnegbin(X,betai,alpha), X=X, beta=betai)
}
Beta = NULL
for (i in 1:nreg) {Beta = rbind(Beta,matrix(simnegbindata[[i]]$beta,nrow=1))}
Data1 = list(regdata=simnegbindata, Z=Z)
Mcmc1 = list(R=R)




out1 = bayesm::rhierNegbinRw(Data=Data1, Mcmc=Mcmc1)
cat("Summary of Delta draws", fill=TRUE)
summary(out1$Deltadraw, tvalues=as.vector(Delta))
cat("Summary of Vbeta draws", fill=TRUE)
summary(out1$Vbetadraw, tvalues=as.vector(Vbeta[upper.tri(Vbeta,diag=TRUE)]))
cat("Summary of alpha draws", fill=TRUE)
summary(out1$alpha, tvalues=alpha)




out2 = bayesmtest::rhierNegbinRw(Data=Data1, Mcmc=Mcmc1)
cat("Summary of Delta draws", fill=TRUE)
summary(out2$Deltadraw, tvalues=as.vector(Delta))
cat("Summary of Vbeta draws", fill=TRUE)
summary(out2$Vbetadraw, tvalues=as.vector(Vbeta[upper.tri(Vbeta,diag=TRUE)]))
cat("Summary of alpha draws", fill=TRUE)
summary(out2$alpha, tvalues=alpha)


t1 = apply(out1$Betadraw, c(1,2), effectiveSize)
t2 = apply(out2$Betadraw, c(1,2), effectiveSize)

summary(t1)
summary(t2)
