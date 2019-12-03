library(coda)

R = 2000
nreg = 100

nobs = 100
nvar = 3
Vbeta = matrix(c(1, 0.5, 0.2, 0.5, 2, 0.7, 0.2, 0.7, 1), ncol=3)
Z = cbind(c(rep(1,nreg)), 3*runif(nreg))
Z[,2] = Z[,2] - mean(Z[,2])
nz = ncol(Z)
Delta = matrix(c(1,-1,2,0,1,0), ncol=2)
Delta = t(Delta) # first row of Delta is means of betas
Beta = matrix(rnorm(nreg*nvar),nrow=nreg)%*%chol(Vbeta) + Z%*%Delta
tau = 0.1
iota = c(rep(1,nobs))
regdata = NULL
for (reg in 1:nreg) {
  X = cbind(iota, matrix(runif(nobs*(nvar-1)),ncol=(nvar-1)))
y = X%*%Beta[reg,] + sqrt(tau)*rnorm(nobs)
regdata[[reg]] = list(y=y, X=X)
}



Data1 = list(regdata=regdata, Z=Z)
Mcmc1 = list(R=R, keep=1)


# gESS slice sampler
t1 = proc.time()
out1 = bayesm::rhierLinearModel_gESS(Data=Data1, Mcmc=Mcmc1)
t1 = proc.time() - t1

## plotting examples
if(0){
  plot(out1$betadraw)
  plot(out1$Deltadraw)
}


# Regular Gibbs sampler
t2 = proc.time()
out2 = bayesm::rhierLinearModel(Data=Data1, Mcmc=Mcmc1)
t2 = proc.time() - t2

t3 = proc.time()
out3 = bayesm::rhierLinearModel_slice(Data=Data1, Mcmc=Mcmc1)
t3 = proc.time() - t3

cat("Summary of beta draws, ESS", fill=TRUE)
summary(out1$betadraw, tvalues=as.vector(Beta))
cat("Summary of beta draws", fill=TRUE)
summary(out2$betadraw, tvalues=as.vector(Beta))





draw = function(i, j){
    par(mfrow=c(3,3))
    plot(out1$betadraw[i,j, 200:2000])
    plot(out2$betadraw[i,j, 200:2000])
    plot(out3$betadraw[i,j, 200:2000])
    cat("gESS ", mean(out1$betadraw[i,j, 200:2000]), " ",  sd(out1$betadraw[i,j, 200:2000]), "\n")
    cat("MH ", mean(out2$betadraw[i,j, 200:2000]), " ",  sd(out2$betadraw[i,j, 200:2000]), "\n")
    cat("ESS ", mean(out3$betadraw[i,j, 200:2000]), " ",  sd(out2$betadraw[i,j, 200:2000]), "\n")
    acf(out1$betadraw[i,j, 200:2000])
    acf(out2$betadraw[i,j, 200:2000])
    acf(out3$betadraw[i,j, 200:2000])
    plot(density(out1$betadraw[i,j, 200:2000]))
    lines(density(out2$betadraw[i,j, 200:2000]), col = "red")
    lines(density(out3$betadraw[i,j, 200:2000]), col = "blue")
}
draw(1,1)








cat("Summary of Delta draws, ESS", fill=TRUE)
summary(out1$Deltadraw, tvalues=as.vector(Delta))
cat("Summary of Delta draws", fill=TRUE)
summary(out2$Deltadraw, tvalues=as.vector(Delta))


cat("Summary of Vbeta draws, ESS", fill=TRUE)
summary(out1$Vbetadraw, tvalues=as.vector(Vbeta[upper.tri(Vbeta,diag=TRUE)]))
cat("Summary of Vbeta draws", fill=TRUE)
summary(out2$Vbetadraw, tvalues=as.vector(Vbeta[upper.tri(Vbeta,diag=TRUE)]))


## plotting examples
if(0){
  plot(out2$betadraw)
  plot(out2$Deltadraw)
}

mean(apply(out1$betadraw, c(1,2), effectiveSize))
mean(apply(out2$betadraw, c(1,2), effectiveSize))
mean(apply(out3$betadraw, c(1,2), effectiveSize))

t1
t2
t3





# check posterior mean and sd
par(mfrow = c(2,2))
a = as.vector(apply(out1$betadraw, c(1,2), mean))
b = as.vector(apply(out2$betadraw, c(1,2), mean))
d = as.vector(apply(out3$betadraw, c(1,2), mean))
plot(b,a, ylab="GESS", xlab="Gibbs", main = "Posterior mean")
abline(0,1,col="red",lwd = 2)
plot(d,a, ylab="ESS", xlab="Gibbs", main = "Posterior mean")
abline(0,1,col="red",lwd = 2)

a = as.vector(apply(out1$betadraw, c(1,2), sd))
b = as.vector(apply(out2$betadraw, c(1,2), sd))
d = as.vector(apply(out3$betadraw, c(1,2), sd))
plot(b,a, ylab="GESS", xlab="Gibbs", main = "Posterior std")
abline(0,1,col="red",lwd = 2)
plot(d,a, ylab="ESS", xlab="Gibbs", main = "Posterior std")
abline(0,1,col="red",lwd = 2)
