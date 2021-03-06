library(bayesmtest)
library(coda)

set.seed(66)
R = 5000

p = 3
ncoef = 3
nlgt = 300
nz = 2
Z = matrix(runif(nz*nlgt),ncol=nz)
Z = t(t(Z)-apply(Z,2,mean))
ncomp = 3
Delta=matrix(c(1,0,1,0,1,2),ncol=2)
# num of choice alterns
# num of cross sectional units
# demean Z
# no of mixture components
comps = NULL
comps[[1]] = list(mu=c(0,-1,-2),   rooti=diag(rep(2,3)))

comps[[2]] = list(mu=c(0,-1,-2)*2, rooti=diag(rep(2,3)))
comps[[3]] = list(mu=c(0,-1,-2)*4, rooti=diag(rep(2,3)))
pvec=c(0.4, 0.2, 0.4)
##  simulate from MNL model conditional on X matrix
simmnlwX = function(n,X,beta) {
  k = length(beta)
  Xbeta = X%*%beta
  j = nrow(Xbeta) / n
  Xbeta = matrix(Xbeta, byrow=TRUE, ncol=j)
  Prob = exp(Xbeta)
  iota = c(rep(1,j))
  denom = Prob%*%iota
  Prob = Prob / as.vector(denom)
  y = vector("double", n)
  ind = 1:j
  for (i in 1:n) {
  yvec = rmultinom(1, 1, Prob[i,])
  y[i] = ind%*%yvec}
  return(list(y=y, X=X, beta=beta, prob=Prob))
}



## simulate data with a mixture of 3 normals
simlgtdata = NULL
ni = rep(50,300)
for (i in 1:nlgt) {
  betai = Delta%*%Z[i,] + as.vector(rmixture(1,pvec,comps)$x)
   Xa = matrix(runif(ni[i]*p,min=-1.5,max=0), ncol=p)
   X = createX(p, na=1, nd=NULL, Xa=Xa, Xd=NULL, base=1)
   outa = simmnlwX(ni[i], X, betai)
   simlgtdata[[i]] = list(y=outa$y, X=X, beta=betai)
}
## plot betas
if(0){
  bmat = matrix(0, nlgt, ncoef)
  for(i in 1:nlgt) { bmat[i,] = simlgtdata[[i]]$beta }
  par(mfrow = c(ncoef,1))
  for(i in 1:ncoef) { hist(bmat[,i], breaks=30, col="magenta") }
}


## set Data and Mcmc lists
keep = 5
Mcmc1 = list(R=R, keep=keep)
Data1 = list(p=p, lgtdata=simlgtdata, Z=Z)



out1 = bayesm::rhierMnlDP(Data=Data1, Mcmc=Mcmc1)
cat("Summary of Delta draws", fill=TRUE)
summary(out1$Deltadraw, tvalues=as.vector(Delta))



out2 = bayesmtest::rhierMnlDP(Data=Data1, Mcmc=Mcmc1)
cat("Summary of Delta draws", fill=TRUE)
summary(out2$Deltadraw, tvalues=as.vector(Delta))




eff1 = apply(out1$betadraw, c(1,2), effectiveSize)
eff2 = apply(out2$betadraw, c(1,2), effectiveSize)
summary(eff1)
summary(eff2)



