library(bayesm)
library(doParallel)
library(coda)

N_simu = 10 # number of simulations

set.seed(109)

presition_beta = 1
ncomp = 10


p = 2
R = 10000

burnin = 3000



ncoef = 3
nlgt = 20
nobs = 100
nz = 2


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
for (i in 1:ncomp) {
  comps[[i]] = list(mu = mu * i, rooti = diag(rep(presition_beta, p)))
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

# Gibbs sampler
time1 = proc.time()
out1 = bayesm::rhierMnlRwMixture(Data = Data1, Prior = Prior1, Mcmc = Mcmc1)
time1 = proc.time() - time1
summary1 = summary_posterior(out1, betatrue, Delta)
summary1$time = time1

# elliptical slice sampler, without MH burnin, no fix p burnin
time2 = proc.time()
out2 = bayesm::rhierMnlRwMixture_slice(Data = Data1, Prior = Prior1, Mcmc = Mcmc1, FALSE, FALSE)
time2 = proc.time() - time2
summary2 = summary_posterior(out2, betatrue, Delta)
summary2$time = time2

# elliptical slice sampler, with MH burnin, no fix p burnin
time3 = proc.time()
out3 = bayesm::rhierMnlRwMixture_slice(Data = Data1, Prior = Prior1, Mcmc = Mcmc1, TRUE, FALSE)
# out3 = bayesm::rhierMnlRwMixture_slice(Data = Data1, Prior = Prior1, Mcmc = Mcmc1, FALSE, FALSE)
time3 = proc.time() - time3
summary3 = summary_posterior(out3, betatrue, Delta)
summary3$time = time3

# elliptical slice sampler, without MH burnin, with fix p burnin
time4 = proc.time()
out4 = bayesm::rhierMnlRwMixture_slice(Data = Data1, Prior = Prior1, Mcmc = Mcmc1, FALSE, FALSE)
time4 = proc.time() - time4
summary4 = summary_posterior(out4, betatrue, Delta)
summary4$time = time4

# elliptical slice sampler, with MH burnin, with fix p burnin
time5 = proc.time()
out5 = bayesm::rhierMnlRwMixture_slice(Data = Data1, Prior = Prior1, Mcmc = Mcmc1, TRUE, FALSE)
time5 = proc.time() - time5
summary5 = summary_posterior(out5, betatrue, Delta)
summary5$time = time5

# generalized elliptical slice sampler, without MH burnin, no fix p burnin
time6 = proc.time()
out6 = bayesm::rhierMnlRwMixture_gESS(Data = Data1, Prior = Prior1, Mcmc = Mcmc1, FALSE, FALSE)
time6 = proc.time() - time6
summary6 = summary_posterior(out6, betatrue, Delta)
summary6$time = time6

# generalized elliptical slice sampler, without MH burnin, no fix p burnin
time7 = proc.time()
out7 = bayesm::rhierMnlRwMixture_gESS(Data = Data1, Prior = Prior1, Mcmc = Mcmc1, TRUE, TRUE)
time7 = proc.time() - time7
summary7 = summary_posterior(out7, betatrue, Delta)
summary7$time = time7

par(mfrow = c(3, 3))
for (i in 1:7) {
  objname = paste("beta", i, sep = "")
  resname = paste("out", i, sep = "")
  temp = apply(get(resname)$betadraw[,, (0.2 * R):R], c(1, 2), mean)
  assign(objname, temp)
}

plot(as.vector(betatrue), as.vector(beta1), main = "MH")
abline(0, 1)
plot(as.vector(betatrue), as.vector(beta2), main = "ESS - MH - p")
abline(0, 1)
plot(as.vector(betatrue), as.vector(beta3), main = "ESS + MH - p")
abline(0, 1)
plot(as.vector(betatrue), as.vector(beta4), main = "ESS - MH + p")
abline(0, 1)
plot(as.vector(betatrue), as.vector(beta5), main = "ESS + MH + p")
abline(0, 1)
plot(as.vector(betatrue), as.vector(beta6), main = "gESS - MH - p")
abline(0, 1)
plot(as.vector(betatrue), as.vector(beta7), main = "gESS + MH + p")
abline(0, 1)
plot(as.vector(beta3), as.vector(beta5), main = "ESS + MH - p vs ESS + MH + p")
abline(0, 1)
plot(as.vector(beta2), as.vector(beta4), main = "ESS - MH - p vs ESS - MH + p")
abline(0, 1)





