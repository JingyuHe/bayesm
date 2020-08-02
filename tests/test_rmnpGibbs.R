# library(bayesm)
library(bayesmtest)
library(coda)


R = 1000 # number of posterior draws
burn = 200 # number of burn-in
set.seed(255242)
thinning = 5
n = 30000


Posdef <- function (n, ev = runif(n, 0, 10)) 
{     # generate random positive definite matrix as covariance matrix 
      Z <- matrix(ncol=n, rnorm(n^2)) 
      decomp <- qr(Z) 
      Q <- qr.Q(decomp) 
      R <- qr.R(decomp) 
      d <- diag(R) 
      ph <- d / abs(d) 
      O <- Q %*% diag(ph) 
      Z <- t(O) %*% diag(ev) %*% O 
      return(Z) 
} 



simmnp = function(X, p, n, beta, sigma) {
      indmax = function(x) {which(max(x)==x)}
      Xbeta = X%*%beta
      w = as.vector(crossprod(chol(sigma),matrix(rnorm((p-1)*n),ncol=n))) + Xbeta
      w = matrix(w, ncol=(p-1), byrow=TRUE)
      maxw = apply(w, 1, max)
      y = apply(w, 1, indmax)
      y = ifelse(maxw < 0, p, y)
      return(list(y=y, X=X, beta=beta, sigma=sigma))
}

p = 10
beta = rnorm(p + 1)
#Sigma = matrix(c(1, 0.5, 0.5, 1), ncol=2)
Sigma = diag(p - 1)
# p = 10
# n = 1000
# beta = rnorm(11)
# Sigma = Posdef(n = p - 1)
k = length(beta)
X1 = matrix(runif(n*p,min=0,max=2),ncol=p)
X2 = matrix(runif(n*p,min=0,max=2),ncol=p)
X = createX(p, na=2, nd=NULL, Xa=cbind(X1,X2), Xd=NULL, DIFF=TRUE, base=p)



###########################################################################
# regular Gibbs sampler
st = proc.time()[3]

simout = simmnp(X,p,n,beta,Sigma)
Data1 = list(p=p, y=simout$y, X=simout$X)
Mcmc1 = list(R=R, keep=thinning)
out = bayesm::rmnpGibbs(Data=Data1, Mcmc=Mcmc1)
cat(" Summary of Betadraws ", fill=TRUE)
betatilde = out$betadraw / sqrt(out$sigmadraw[,1])
attributes(betatilde)$class = "bayesm.mat"
summary(betatilde, tvalues=beta, burnin = burn)
cat(" Summary of Sigmadraws ", fill=TRUE)
sigmadraw = out$sigmadraw / out$sigmadraw[,1]
attributes(sigmadraw)$class = "bayesm.var"
summary(sigmadraw, tvalues=as.vector(Sigma[upper.tri(Sigma,diag=TRUE)]), burnin = burn)
## plotting examples
if(0){plot(betatilde,tvalues=beta)}

apply(out$betadraw, 2, effectiveSize)

et = proc.time()[3]
et-st

###########################################################################
# elliptical slice sampler

st = proc.time()[3]
simout = simmnp(X,p,n,beta,Sigma)
Data1 = list(p=p, y=simout$y, X=simout$X)
Mcmc1 = list(R=R, keep=thinning)
out2 = bayesmtest::rmnpGibbs(Data=Data1, Mcmc=Mcmc1)
cat(" Summary of Betadraws ", fill=TRUE)
betatilde2 = out2$betadraw / sqrt(out2$sigmadraw[,1])
attributes(betatilde2)$class = "bayesm.mat"
summary(betatilde2, tvalues=beta, burnin = burn)
cat(" Summary of Sigmadraws ", fill=TRUE)
sigmadraw2 = out2$sigmadraw / out2$sigmadraw[,1]
attributes(sigmadraw2)$class = "bayesm.var"
summary(sigmadraw2, tvalues=as.vector(Sigma[upper.tri(Sigma,diag=TRUE)]), burnin = burn)
## plotting examples
if(0){plot(betatilde2,tvalues=beta)}


apply(out2$betadraw, 2, effectiveSize)

et = proc.time()[3]
et-st

