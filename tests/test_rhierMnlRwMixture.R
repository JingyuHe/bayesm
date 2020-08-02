rm(list = ls())
library(bayesm)
library(doParallel)
library(coda)
library(clusterGeneration)

N_simu = 10 # number of simulations

presition_beta = 1
ncomp = 3


p = 5
R = 40000

burnin = 10000


nlgt = 100
nobs = 20
nz = 5


##  simulate from MNL model conditional on X matrix
simmnlwX = function(n, X, beta) {
  k = length(beta)
  Xbeta = X %*% beta
  j = nrow(Xbeta) / n
  Xbeta = matrix(Xbeta, byrow = TRUE, ncol = j)
  Prob = exp(Xbeta)
  iota = c(rep(1, j))
  denom = Prob %*% iota
  Prob = Prob / as.vector(denom)
  y = vector("double", n)
  ind = 1:j
  for (i in 1:n) {
    yvec = rmultinom(1, 1, Prob[i,])
    y[i] = ind %*% yvec
  }
  return(list(y = y, X = X, beta = beta, prob = Prob))
}



Z = matrix(runif(nz * nlgt), ncol = nz)
Z = t(t(Z) - apply(Z, 2, mean))
Delta = matrix(sample(c(-2, -1, 0, 1, 2), p * nz, TRUE), ncol = nz)
comps = NULL
mu = sample(c(-2, -1, 0, 1, 2), p, TRUE)
# comps[[1]] = list(mu=mu,   rooti=diag(rep(1,p)))
# comps[[2]] = list(mu=mu*2, rooti=diag(rep(1,p)))
# comps[[3]] = list(mu=mu*4, rooti=diag(rep(1,p)))
# pvec = c(0.4, 0.2, 0.4)
# pvec = c(1)
pvec = rdirichlet(rep(ncomp, ncomp))
comps = list()

# a = matrix(c(1,0,0.5773503,1.1547005), ncol=2)
# a = t(chol(solve(rcorrmatrix(p))))
a = diag(rep(presition_beta, p))

for (i in 1:ncomp) {
  # comps[[i]] = list(mu = mu * i, rooti = diag(rep(presition_beta, p)))
  comps[[i]] = list(mu = mu * i, rooti = a)
}




## simulate data
simlgtdata = NULL
betatrue = matrix(0, nlgt, p)
ni = rep(nobs, nlgt)
for (i in 1:nlgt) {
  betai = Delta %*% Z[i,] + as.vector(rmixture(1, pvec, comps)$x)
  betatrue[i,] = betai
  Xa = matrix(runif(ni[i] * p, min = -1.5, max = 0), ncol = p)
  X = createX(p, na = 1, nd = NULL, Xa = Xa, Xd = NULL, base = 1)
  outa = simmnlwX(ni[i], X, betai)
  simlgtdata[[i]] = list(y = outa$y, X = X, beta = betai)
}

## set parms for priors and Z
Prior1 = list(ncomp = ncomp)
keep = 1
Mcmc1 = list(R = R, keep = keep)
Data1 = list(p = p, lgtdata = simlgtdata, Z = Z)


# generalized elliptical slice sampler, without MH burnin, no fix p burnin
time1 = proc.time()
out1 = bayesm::rhierMnlRwMixture_gESS(Data = Data1, Prior = Prior1, Mcmc = Mcmc1, TRUE, FALSE)
time1 = proc.time() - time1

# Gibbs sampler
time2 = proc.time()
# out1 = bayesm::rhierMnlRwMixture(Data = Data1, Prior = Prior1, Mcmc = Mcmc1)
out2 = bayesm::rhierMnlRwMixture(Data = Data1, Prior = Prior1, Mcmc = Mcmc1)
# out3 = bayesm::rhierMnlRwMixture(Data = Data1, Prior = Prior1, Mcmc = Mcmc1)
time2 = proc.time() - time2


# elliptical slice sampler, without MH burnin, no fix p burnin
time3 = proc.time()
out3 = bayesm::rhierMnlRwMixture_slice(Data = Data1, Prior = Prior1, Mcmc = Mcmc1, TRUE, FALSE)
time3 = proc.time() - time3

out4 = bayesm::rhierMnlRwMixture(Data = Data1, Prior = Prior1, Mcmc = Mcmc1)


# check posterior mean and sd
# pdf("logit_beta.pdf", height = 8, width = 8)
par(mfrow = c(3,4))
a = as.vector(apply(out1$betadraw[,,burnin:R], c(1,2), mean))
b = as.vector(apply(out2$betadraw[,,burnin:R], c(1,2), mean))
d = as.vector(apply(out3$betadraw[,,burnin:R], c(1,2), mean))
e = as.vector(apply(out4$betadraw[,,burnin:R], c(1,2), mean))
plot(b,a, ylab="GESS", xlab="MH", main = "Posterior mean")
abline(0,1,col="red",lwd = 2)
plot(b,d, ylab="ESS", xlab="MH", main = "Posterior mean")
abline(0,1,col="red",lwd = 2)
plot(a,d, ylab="ESS", xlab="GESS", main = "Posterior mean")
abline(0,1,col="red",lwd = 2)
plot(b,e, ylab="MH2", xlab="MH", main = "Posterior mean")
abline(0,1,col="red",lwd = 2)

a = as.vector(apply(out1$betadraw[,,burnin:R], c(1,2), sd))
b = as.vector(apply(out2$betadraw[,,burnin:R], c(1,2), sd))
d = as.vector(apply(out3$betadraw[,,burnin:R], c(1,2), sd))
e = as.vector(apply(out4$betadraw[,,burnin:R], c(1,2), sd))
plot(b,a, ylab="GESS", xlab="MH", main = "Posterior std")
abline(0,1,col="red",lwd = 2)
plot(b,d, ylab="ESS", xlab="MH", main = "Posterior std")
abline(0,1,col="red",lwd = 2)
plot(a,d, ylab="ESS", xlab="GESS", main = "Posterior std")
abline(0,1,col="red",lwd = 2)
plot(b,e, ylab="MH2", xlab="MH", main = "Posterior std")
abline(0,1,col="red",lwd = 2)



plot(out1$betadraw[2,2,], main = "GESS")
plot(out2$betadraw[2,2,], main = "MH")
plot(out4$betadraw[2,2,], main = "MH2")
plot(out3$betadraw[2,2,], main = "ESS")

# dev.off()

calc_ESS = function(){
  ESS_GESS = mean(apply(out1$betadraw[,,burnin:R], c(1,2), effectiveSize))
  ESS_MH = mean(apply(out2$betadraw[,,burnin:R], c(1,2), effectiveSize))
  ESS_ESS = mean(apply(out3$betadraw[,,burnin:R], c(1,2), effectiveSize))

  cat("effective size, GESS, ", ESS_GESS, ", ratio to MH, ", ESS_GESS / ESS_MH, "\n")
  cat("effective size, MH, ", ESS_MH, "\n")
  cat("effective size, ESS, ", ESS_ESS, ", ratio to MH, ", ESS_ESS / ESS_MH, "\n")
  cat("time ratio, GESS", time1[3] / time2[3], "\n")
  cat("time ratio, ESS", time3[3] / time2[3], "\n")
}



calc_ESS()


