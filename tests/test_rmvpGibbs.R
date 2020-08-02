library(bayesm)
library(coda)

R = 3000

seed(43423)

simmvp = function(X, p, n, beta, sigma) {
      w = as.vector(crossprod(chol(sigma),matrix(rnorm(p*n),ncol=n))) + X%*%beta
      y = ifelse(w<0, 0, 1)    
      return(list(y=y, X=X, beta=beta, sigma=sigma))
}


p = 20
n = 500
beta = sample(c(-2,-1,0,1,2), p, replace = TRUE)
#Sigma = matrix(c(1, 0.5, 0.5, 0.5, 1, 0.5, 0.5, 0.5, 1), ncol=3) 
Sigma = diag(p)*.5
k = length(beta)
I2 = diag(rep(1,p))
xadd = rbind(I2)
for(i in 2:n){ 
    xadd=rbind(xadd,I2) 
}


###################################################################
# bayesm package
# original gibbs sampler


X = xadd
simout = simmvp(X,p,500,beta,Sigma)
Data1 = list(p=p, y=simout$y, X=simout$X)
Mcmc1 = list(R=R, keep=1)
st = proc.time()[3]
out = bayesm::rmvpGibbs(Data=Data1, Mcmc=Mcmc1)
ind = seq(from=0, by=p, length=k)
inda = 1:p
ind = ind + inda
cat(" Betadraws ", fill=TRUE)
betatilde = out$betadraw / sqrt(out$sigmadraw[,ind])
attributes(betatilde)$class = "bayesm.mat"
summary(betatilde, tvalues=beta/sqrt(diag(Sigma)))
rdraw = matrix(double((R)*p*p), ncol=p*p)
rdraw = t(apply(out$sigmadraw, 1, nmat))
attributes(rdraw)$class = "bayesm.var"
tvalue = nmat(as.vector(Sigma))
dim(tvalue) = c(p,p)
tvalue = as.vector(tvalue[upper.tri(tvalue,diag=TRUE)])
cat(" Draws of Correlation Matrix ", fill=TRUE)
summary(rdraw, tvalues=tvalue)
## plotting examples
if(0){plot(betatilde, tvalues=beta/sqrt(diag(Sigma)))}
et = proc.time()[3]

time1 = et - st



###################################################################
# bayesmtest package
# elliptical slice sampler
X = xadd
simout = simmvp(X,p,500,beta,Sigma)
Data1 = list(p=p, y=simout$y, X=simout$X)
Mcmc1 = list(R=R, keep=1)
st = proc.time()[3]
out2 = bayesm::rmvpGibbs_slice(Data=Data1, Mcmc=Mcmc1)
ind = seq(from=0, by=p, length=k)
inda = 1:p
ind = ind + inda
cat(" Betadraws ", fill=TRUE)
betatilde2 = out2$betadraw / sqrt(out2$sigmadraw[,ind])
attributes(betatilde2)$class = "bayesm.mat"
summary(betatilde2, tvalues=beta/sqrt(diag(Sigma)))
rdraw2 = matrix(double((R)*p*p), ncol=p*p)
rdraw2 = t(apply(out2$sigmadraw, 1, nmat))
attributes(rdraw2)$class = "bayesm.var"
tvalue2 = nmat(as.vector(Sigma))
dim(tvalue2) = c(p,p)
tvalue2 = as.vector(tvalue2[upper.tri(tvalue2,diag=TRUE)])
cat(" Draws of Correlation Matrix ", fill=TRUE)
summary(rdraw2, tvalues=tvalue2)
## plotting examples
if(0){plot(betatilde2, tvalues=beta/sqrt(diag(Sigma)))}
et = proc.time()[3]

time2 = et -st


cat("compare effective sample size \n")
apply(out$betadraw, 2, effectiveSize)
apply(out2$betadraw, 2, effectiveSize)
time1
time2
