library(bayesm)
library(coda)
library(clusterGeneration)

R = 2000
# set.seed(66)
nreg = 40
nobs = 5000
nvar = 3
nz = 2
Z = matrix(runif(nreg*nz), ncol=nz)
Z = t(t(Z) - apply(Z,2,mean))
Delta = matrix(c(1,-1,2,0,1,0), ncol=nz)
tau0 = 0.1
iota = c(rep(1,nobs))
## create arguments for rmixture
tcomps = NULL
a = matrix(c(1,0,0,0.5773503,1.1547005,0,-0.4082483,0.4082483,1.2247449), ncol=3)
a1 = a2 = a3 = a

# generate random correlation matrix
# a1 = t(chol(solve(rcorrmatrix(3))))
# a2 = t(chol(solve(rcorrmatrix(3))))
# a3 = t(chol(solve(rcorrmatrix(3))))
tcomps[[1]] = list(mu=c(0,-1,-2),   rooti=a1)
tcomps[[2]] = list(mu=c(0,-1,-2)*2, rooti=a2)
tcomps[[3]] = list(mu=c(0,-1,-2)*4, rooti=a3)
tpvec = c(0.4, 0.2, 0.4)


## simulated data with Z
regdata = NULL
betas = matrix(double(nreg*nvar), ncol=nvar)
tind = double(nreg)
for (reg in 1:nreg) {
tempout = rmixture(1,tpvec,tcomps)
betas[reg,] = Delta%*%Z[reg,] + as.vector(tempout$x)
tind[reg] = tempout$z
X = cbind(iota, matrix(runif(nobs*(nvar-1)),ncol=(nvar-1)))
tau = tau0*runif(1,min=0.5,max=1)
y = X%*%betas[reg,] + sqrt(tau)*rnorm(nobs)
regdata[[reg]] = list(y=y, X=X, beta=betas[reg,], tau=tau)
}
## run rhierLinearMixture
Data1 = list(regdata=regdata, Z=Z)
Prior1 = list(ncomp=3)
Mcmc1 = list(R=R, keep=1)

t1 = proc.time()
out1 = rhierLinearMixture_gESS(Data=Data1, Prior=Prior1, Mcmc=Mcmc1)
t1 = proc.time() - t1

t2 = proc.time()
# out1 = rhierLinearMixture(Data=Data1, Prior=Prior1, Mcmc=Mcmc1)
out2 = rhierLinearMixture(Data=Data1, Prior=Prior1, Mcmc=Mcmc1)
# out3 = rhierLinearMixture(Data=Data1, Prior=Prior1, Mcmc=Mcmc1)
t2 = proc.time() - t2

t3 = proc.time()
out3 = rhierLinearMixture_slice(Data=Data1, Prior=Prior1, Mcmc=Mcmc1)
t3 = proc.time() - t3

cat("Summary of Delta draws", fill=TRUE)
summary(out1$Deltadraw, tvalues=as.vector(Delta))
cat("Summary of Normal Mixture Distribution", fill=TRUE)
summary(out1$nmix)
## plotting examples
if(0){
  plot(out1$betadraw)
  plot(out1$nmix)
  plot(out1$Deltadraw)
}




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



# K-S test, whether two empirical distribution same or not
p_values = matrix(0, dim(out1$betadraw)[1], dim(out1$betadraw)[2])

for(i in 1:(dim(out1$betadraw)[1])){
  for(j in 1:(dim(out1$betadraw)[2])){
    p_values[i,j] = ks.test(out1$betadraw[i,j, ], out2$betadraw[i,j, ])$p.value
  }
}


mean(apply(out1$betadraw, c(1,2), effectiveSize))
mean(apply(out2$betadraw, c(1,2), effectiveSize))
mean(apply(out3$betadraw, c(1,2), effectiveSize))

t1
t2
t3




pdf(file="linear_acf_gESS.pdf", height = 6, width = 6)

a1 = acf(out1$betadraw[j,2,],plot=FALSE)
plot(a1$acf~a1$lag,type='h',col="#00000005",lwd=10, xlab = "Lag", ylab = "Autocorrelation")

for(j in 1:50) {
a1 = acf(out1$betadraw[j,2,],plot=FALSE)
lines(a1$acf~a1$lag,type='h',col="#00000005",lwd=10)
}

dev.off()



pdf(file="linear_acf_Gibbs.pdf", height = 6, width = 6)

a1 = acf(out2$betadraw[j,2,],plot=FALSE)
plot(a1$acf~a1$lag,type='h',col="#00000005",lwd=10, xlab = "Lag", ylab = "Autocorrelation")

for(j in 1:50) {
a1 = acf(out2$betadraw[j,2,],plot=FALSE)
lines(a1$acf~a1$lag,type='h',col="#00000005",lwd=10)
}

dev.off()



pdf(file="linear_acf_ESS.pdf", height = 6, width = 6)

a1 = acf(out3$betadraw[j,2,],plot=FALSE)
plot(a1$acf~a1$lag,type='h',col="#00000005",lwd=10, xlab = "Lag", ylab = "Autocorrelation")

for(j in 1:50) {
a1 = acf(out3$betadraw[j,2,],plot=FALSE)
lines(a1$acf~a1$lag,type='h',col="#00000005",lwd=10)
}

dev.off()






# check posterior mean and sd
par(mfrow = c(2,2))
a = as.vector(apply(out1$betadraw, c(1,2), mean))
b = as.vector(apply(out2$betadraw, c(1,2), mean))
d = as.vector(apply(out3$betadraw, c(1,2), mean))
plot(b,a, ylab="GESS", xlab="Gibbs", main = "Posterior mean")
abline(0,1,col="red",lwd = 2)
plot(b,d, ylab="ESS", xlab="Gibbs", main = "Posterior mean")
abline(0,1,col="red",lwd = 2)

a = as.vector(apply(out1$betadraw, c(1,2), sd))
b = as.vector(apply(out2$betadraw, c(1,2), sd))
d = as.vector(apply(out3$betadraw, c(1,2), sd))
plot(b,a, ylab="GESS", xlab="Gibbs", main = "Posterior std")
abline(0,1,col="red",lwd = 2)
plot(b,d, ylab="ESS", xlab="Gibbs", main = "Posterior std")
abline(0,1,col="red",lwd = 2)
