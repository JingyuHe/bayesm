library(bayesm)
library(doParallel)
library(coda)

filename = "1.rda"

N_simu = 10 # number of simulations

# set.seed(100)

presition_beta = 1
ncomp = 10


p = 3
# posterior samples
R = 10000
# burnin
burnin = 3000



ncoef = 3
nlgt = 100
nobs = 100
nz = 3


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



## plot betas
if (0) {
  bmat = matrix(0, nlgt, ncoef)
  for (i in 1:nlgt) { bmat[i,] = simlgtdata[[i]]$beta }
  par(mfrow = c(ncoef, 1))
  for (i in 1:ncoef) { hist(bmat[, i], breaks = 30, col = "magenta") }
}


## plotting examples
if (0) {
  plot(out1$betadraw)
  plot(out1$nmix)
}




get_cov = function(out) {
  # subtract posterior samples of covariance matrix (cholesky root) from the output object
  ncomp = length(out$nmix$compdraw[[1]])
  nsample = length(out$nmix$compdraw)
  p = dim(out$nmix$compdraw[[1]][[1]]$rooti)[1]
  output = list()
  for (i in 1:ncomp) {
    mat = matrix(0, nsample, (p * (p - 1) / 2 + p))
    for (j in 1:nsample) {
      cov_mat = out$nmix$compdraw[[j]][[i]]$rooti
      cov_mat = cov_mat[upper.tri(out$nmix$compdraw[[1]][[1]]$rooti, TRUE)]
      mat[j,] = cov_mat
    }
    output[[i]] = mat
  }
  return(output)
}

get_betabar = function(out) {
  # subtract posterior samples of betabar from the output object
  ncomp = length(out$nmix$compdraw[[1]])
  nsample = length(out$nmix$compdraw)
  p = dim(out$nmix$compdraw[[1]][[1]]$rooti)[1]
  output = list()
  for (i in 1:ncomp) {
    mat = matrix(0, nsample, p)
    for (j in 1:nsample) {
      mat[j,] = out$nmix$compdraw[[j]][[i]]$mu
    }
    output[[i]] = mat
  }
  return(output)
}


summary_posterior = function(out, betatrue, Delta) {

  # this function summary posterior

  summary1 = summary(out$Deltadraw, burnin = burnin, tvalues = as.vector(Delta))

  MSE_delta = mean((summary1[, 2] - as.vector(Delta)) ^ 2)

  EFF_delta = apply(out$Deltadraw, 2, effectiveSize)

  beta = apply(out$betadraw[,, burnin:R], c(1, 2), mean)

  MSE_beta = mean((beta - betatrue) ^ 2)

  EFF_beta = apply(out$betadraw, c(1, 2), effectiveSize)

  cov1 = get_cov(out)

  betabar1 = get_betabar(out)


  ncomp = length(cov1)
  EFF_cov = matrix(0, dim(cov1[[1]])[2], ncomp)
  for (i in 1:ncomp) {
    eff1 = apply(cov1[[i]], 2, effectiveSize)
    EFF_cov[, i] = eff1
  }


  EFF_betabar = matrix(0, dim(betabar1[[1]])[2], ncomp)
  for (i in 1:ncomp) {
    eff1 = apply(betabar1[[i]], 2, effectiveSize)
    EFF_betabar[, i] = eff1
  }

  EFF_prob = apply(out$nmix$probdraw, 2, effectiveSize)

  output = list(MSE_delta = MSE_delta, MSE_beta = MSE_beta, EFF_delta = EFF_delta, EFF_beta = EFF_beta, EFF_betabar = EFF_betabar, EFF_prob = EFF_prob, EFF_cov = EFF_cov, beta = beta, delta = summary1[, 2])

  return(output)
}





simulation = function() {
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


  output = list(summary1, summary2, summary3, summary4, summary5, summary6, summary7)
}



i = 7
j = 2
par(mfrow = c(3, 3))

plot(out1$betadraw[i, j,])
abline(h = betatrue[i, j], col = "red", lwd = 2)
plot(out2$betadraw[i, j,])
abline(h = betatrue[i, j], col = "red", lwd = 2)
plot(out3$betadraw[i, j,])
abline(h = betatrue[i, j], col = "red", lwd = 2)
plot(out4$betadraw[i, j,])
abline(h = betatrue[i, j], col = "red", lwd = 2)
plot(out5$betadraw[i, j,])
abline(h = betatrue[i, j], col = "red", lwd = 2)
plot(out6$betadraw[i, j,])
abline(h = betatrue[i, j], col = "red", lwd = 2)
plot(out7$betadraw[i, j,])
abline(h = betatrue[i, j], col = "red", lwd = 2)


cat("ESS of case 1 ", effectiveSize(out1$betadraw[i, j,]), "\n")
cat("ESS of case 2 ", effectiveSize(out2$betadraw[i, j,]), "\n")
cat("ESS of case 3 ", effectiveSize(out3$betadraw[i, j,]), "\n")
cat("ESS of case 4 ", effectiveSize(out4$betadraw[i, j,]), "\n")
cat("ESS of case 5 ", effectiveSize(out5$betadraw[i, j,]), "\n")
cat("ESS of case 6 ", effectiveSize(out6$betadraw[i, j,]), "\n")
cat("ESS of case 7 ", effectiveSize(out7$betadraw[i, j,]), "\n")




# for(i in 1:7){
#   name = paste("out", i, sep ="")
#   ss = get(name)
#   plot(ss$betadraw[i,j,])
#   abline(h = betatrue[i, j], col = "red", lwd = 2)
# }



par(mfrow = c(3, 3))
# for (i in 1:7) {
#   name = paste("summary", i, sep = "")
#   ss = get(name)
#   plot(as.vector(ss$beta), as.vector(betatrue))
#   abline(0, 1)
# }

plot(as.vector(summary1$beta), as.vector(betatrue))
abline(0, 1)
plot(as.vector(summary2$beta), as.vector(betatrue))
abline(0, 1)
plot(as.vector(summary3$beta), as.vector(betatrue))
abline(0, 1)
plot(as.vector(summary4$beta), as.vector(betatrue))
abline(0, 1)
plot(as.vector(summary5$beta), as.vector(betatrue))
abline(0, 1)
plot(as.vector(summary6$beta), as.vector(betatrue))
abline(0, 1)
plot(as.vector(summary7$beta), as.vector(betatrue))
abline(0, 1)




for (i in 1:7) {
  name = paste("summary", i, sep = "")
  ss = get(name)
  MSE = mean((as.vector(ss$beta) - as.vector(betatrue)) ^ 2)
  cat("MSE of case ", i, " is ", MSE, "\n", sep = "")
}

mean((as.vector(summary7$beta) - as.vector(betatrue)) ^ 2)

# # main loop
# output = list()
# for (i in 1:N_simu) {
#   cat("progress ", i, "\n")
#   output[i] = simulation()
# }


# parallel version
cl = makeCluster(5)
registerDoParallel(cl)

output = foreach(i = 1:N_simu, .packages = c("bayesm", "coda")) %dopar% simulation()







plot_density = function(i, j) {
  plot(density(out1$betadraw[i, j,]))
  lines(density(out2$betadraw[i, j,]), col = "red")
  lines(density(out3$betadraw[i, j,]), col = "green")
  lines(density(out4$betadraw[i, j,]), col = "yellow")
  lines(density(out5$betadraw[i, j,]), col = "blue")
  lines(density(out6$betadraw[i, j,]), col = "black", lty = 2)
  lines(density(out7$betadraw[i, j,]), col = "blue", lty = 2)
  legend("topleft", legend = c("MH", "ESS-MH-p", "ESS+MH-p", "ESS-MH+p", "ESS+MH+p", "gESS-MH-p", "gESS+MH+p"), col = c("black", "red", "green", "yellow", "black", "blue"), lwd = 2)
}














name = "MSE_beta"


MSE_beta_total = c()
MSE_delta_total = c()
EFF_delta_total = c()
EFF_beta_total = c()
EFF_betabar_total = c()
EFF_prob_total = c()
EFF_cov_total = c()
time_total = c()

for (j in 1:length(output)) {

  MSE_beta = c()
  MSE_delta = c()
  EFF_delta = c()
  EFF_beta = c()
  EFF_betabar = c()
  EFF_prob = c()
  EFF_cov = c()
  time = c()
  for (i in 1:7) {
    MSE_beta = cbind(MSE_beta, output[[j]][[i]][["MSE_beta"]])
    MSE_delta = cbind(MSE_delta, output[[j]][[i]][["MSE_delta"]])
    EFF_delta = cbind(EFF_delta, mean(as.vector(output[[j]][[i]][["EFF_delta"]])))
    EFF_beta = cbind(EFF_beta, mean(output[[j]][[i]][["EFF_beta"]]))
    EFF_betabar = cbind(EFF_betabar, mean(output[[j]][[i]][["EFF_betabar"]]))
    EFF_prob = cbind(EFF_prob, mean(output[[j]][[i]][["EFF_prob"]]))
    EFF_cov = cbind(EFF_cov, mean(output[[j]][[i]][["EFF_cov"]]))
    time = cbind(time, output[[j]][[i]][["time"]][3])
  }

  MSE_beta_total = rbind(MSE_beta_total, as.vector(MSE_beta))
  MSE_delta_total = rbind(MSE_delta_total, as.vector(MSE_delta))
  EFF_delta_total = rbind(EFF_delta_total, as.vector(EFF_delta))
  EFF_beta_total = rbind(EFF_beta_total, as.vector(EFF_beta))
  EFF_betabar_total = rbind(EFF_betabar_total, as.vector(EFF_betabar))
  EFF_prob_total = rbind(EFF_prob_total, as.vector(EFF_prob))
  EFF_cov_total = rbind(EFF_cov_total, as.vector(EFF_cov))
  time_total = rbind(time_total, as.vector(time))
}

colnames(MSE_beta_total) = c("MH", "ESS-MH-p", "ESS+MH-p", "ESS-MH+p", "ESS+MH+p", "gESS-MH-p", "gESS+MH+p")
colnames(MSE_delta_total) = c("MH", "ESS-MH-p", "ESS+MH-p", "ESS-MH+p", "ESS+MH+p", "gESS-MH-p", "gESS+MH+p")
colnames(EFF_delta_total) = c("MH", "ESS-MH-p", "ESS+MH-p", "ESS-MH+p", "ESS+MH+p", "gESS-MH-p", "gESS+MH+p")
colnames(EFF_beta_total) = c("MH", "ESS-MH-p", "ESS+MH-p", "ESS-MH+p", "ESS+MH+p", "gESS-MH-p", "gESS+MH+p")
colnames(EFF_betabar_total) = c("MH", "ESS-MH-p", "ESS+MH-p", "ESS-MH+p", "ESS+MH+p", "gESS-MH-p", "gESS+MH+p")
colnames(EFF_prob_total) = c("MH", "ESS-MH-p", "ESS+MH-p", "ESS-MH+p", "ESS+MH+p", "gESS-MH-p", "gESS+MH+p")
colnames(EFF_cov_total) = c("MH", "ESS-MH-p", "ESS+MH-p", "ESS-MH+p", "ESS+MH+p", "gESS-MH-p", "gESS+MH+p")
colnames(time_total) = c("MH", "ESS-MH-p", "ESS+MH-p", "ESS-MH+p", "ESS+MH+p", "gESS-MH-p", "gESS+MH+p")



result = rbind(
colMeans(MSE_beta_total),
colMeans(MSE_delta_total),
colMeans(EFF_delta_total),
colMeans(EFF_beta_total),
colMeans(EFF_betabar_total),
colMeans(EFF_prob_total),
colMeans(EFF_cov_total),
colMeans(time_total)
)


rownames(result) = c("MSE of beta", "MSE of delta", "ESS of delta", "ESS of beta", "ESS of betabar", "ESS of prob", "ESS of cov", "running time")







