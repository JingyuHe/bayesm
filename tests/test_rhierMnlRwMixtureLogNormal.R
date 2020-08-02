library(bayesm)
library(coda)

filename = "1.rda"

N_simu = 100 # number of simulations

# set.seed(1000)


precision_beta = 2


p = 2
R = 5000
ncoef = 3
nlgt = 20
nobs = 50
nz = 2
Z = matrix(runif(nz*nlgt),ncol=nz)
Z = t(t(Z) - apply(Z,2,mean))
ncomp = 1 
Delta = matrix(sample(c(-2,-1,0,1,2), p * nz, TRUE),ncol=nz)
comps=NULL
mu = sample(c(-2,-1,0,1,2), p, TRUE)
# comps[[1]] = list(mu=mu,   rooti=diag(rep(1,p)))
# comps[[2]] = list(mu=mu*2, rooti=diag(rep(1,p)))
# comps[[3]] = list(mu=mu*4, rooti=diag(rep(1,p)))
# pvec = c(0.4, 0.2, 0.4)
# pvec = c(1)
pvec = rdirichlet(rep(ncomp, ncomp))
comps = list()
for(i in 1:ncomp){
    comps[[i]] = list(mu=mu * i,   rooti=diag(rep(precision_beta,p)))
}

##  simulate from MNL model conditional on X matrix
simmnlwX= function(n,X,beta) {
    k = length(beta)
    Xbeta = X%*%beta  
    j = nrow(Xbeta) / n
    Xbeta = matrix(Xbeta, byrow=TRUE, ncol=j)
    Prob = exp(Xbeta)
    iota = c(rep(1,j))
    denom = Prob%*%iota
    Prob = Prob/as.vector(denom)
    y = vector("double",n)
    ind = 1:j
    for (i in 1:n) {
        yvec = rmultinom(1, 1, Prob[i,])
        y[i] = ind%*%yvec
  }
  return(list(y=y, X=X, beta=beta, prob=Prob))
}



## plot betas
if(0){
    bmat = matrix(0, nlgt, ncoef)
    for(i in 1:nlgt) {bmat[i,] = simlgtdata[[i]]$beta}
    par(mfrow = c(ncoef,1))
    for(i in 1:ncoef) { hist(bmat[,i], breaks=30, col="magenta") }
}


## plotting examples
if(0) {
    plot(out1$betadraw)
    plot(out1$nmix)
}




get_cov = function(out){
    # subtract posterior samples of covariance matrix (cholesky root) from the output object
    ncomp = length(out$nmix$compdraw[[1]])
    nsample = length(out$nmix$compdraw)
    p = dim(out$nmix$compdraw[[1]][[1]]$rooti)[1]
    output = list()
    for(i in 1:ncomp){
        mat = matrix(0, nsample, (p*(p-1)/2 + p))
        for(j in 1:nsample){
            cov_mat = out$nmix$compdraw[[j]][[i]]$rooti
            cov_mat = cov_mat[upper.tri(out$nmix$compdraw[[1]][[1]]$rooti, TRUE)]
            mat[j,] = cov_mat
        }
        output[[i]] = mat
    }
    return(output)
}

get_betabar = function(out){
    # subtract posterior samples of betabar from the output object
    ncomp = length(out$nmix$compdraw[[1]])
    nsample = length(out$nmix$compdraw)
    p = dim(out$nmix$compdraw[[1]][[1]]$rooti)[1]
    output = list()
    for(i in 1:ncomp){
        mat = matrix(0, nsample, p)
        for(j in 1:nsample){
            mat[j,] = out$nmix$compdraw[[j]][[i]]$mu
        }
        output[[i]] = mat
    }
    return(output)
}

summary_posterior = function(out1, out2, betatrue){
    # this function summary posteriors of two models
    # the first one out1 is output of gibbs sampler
    # out2 is output of elliptical slice sampler
    summary1 = summary(out1$Deltadraw, tvalues=as.vector(Delta))
    summary2 = summary(out2$Deltadraw, tvalues=as.vector(Delta))
    cat("MSE of Gibbs sampler, delta, ", mean((summary1[,2] - as.vector(Delta))^2), "\n")
    cat("MSE of slice sampler, delta, ", mean((summary2[,2] - as.vector(Delta))^2), "\n")


    beta1 = apply(out1$betadraw[,,(0.2*R):R], c(1,2), mean)
    beta2 = apply(out2$betadraw[,,(0.2*R):R], c(1,2), mean)
    cat("MSE of Gibbs sampler, beta, ", mean((beta1 - betatrue)^2), "\n")
    cat("MSE of slice sampler, beta, ", mean((beta2 - betatrue)^2), "\n")


    eff1 = apply(out1$betadraw, c(1,2), effectiveSize)
    eff2 = apply(out2$betadraw, c(1,2), effectiveSize)
    summary(eff1)
    summary(eff2)
    cat("mean of ratio of effective size for beta ", mean(eff2/eff1), "\n")

    cov1 = get_cov(out1)
    cov2 = get_cov(out2)
    betabar_1 = get_betabar(out1)
    betabar_2 = get_betabar(out2)



    ncomp = length(cov1)
    EFF1 = matrix(0, dim(cov1[[1]])[2], ncomp)
    EFF2 = matrix(0, dim(cov1[[1]])[2], ncomp)
    for(i in 1:ncomp){
        eff1 = apply(cov1[[i]], 2, effectiveSize)
        eff2 = apply(cov2[[i]], 2, effectiveSize)
        EFF1[,i] = eff1
        EFF2[,i] = eff2
        # cat(eff1,"\n")
        # cat(eff2,"\n")
        cat("ratio ", mean(eff2 / eff1), "\n")
        cat("-----", "\n")
    }
    cat("ratio of effective size of Sigma", mean(EFF2) / mean(EFF1), "\n")


    EFFbeta1 = matrix(0, dim(betabar_1[[1]])[2], ncomp)
    EFFbeta2 = matrix(0, dim(betabar_1[[1]])[2], ncomp)
    for(i in 1:ncomp){
        eff1 = apply(betabar_1[[i]], 2, effectiveSize)
        eff2 = apply(betabar_2[[i]], 2, effectiveSize)
        # cat(eff1,"\n")
        # cat(eff2,"\n")
        EFFbeta1[,i] = eff1
        EFFbeta2[,i] = eff2
        cat("ratio ", mean(eff2 / eff1), "\n")
        cat("-----", "\n")
    }
    cat("ratio of effective size of betabar", mean(EFFbeta2) / mean(EFFbeta1), "\n")


    
    apply(out2$nmix$probdraw, 2, effectiveSize)


    
    apply(out2$Deltadraw, 2, effectiveSize)


    output = list(EFF1 = EFF1, EFF2 = EFF2, eff_sigma_ratio = mean(EFF2) / mean(EFF1), EFFbeta1 = EFFbeta1, EFFbeta2 = EFFbeta2, eff_betabar_ratio = mean(EFFbeta2) / mean(EFFbeta1), eff_p_ratio = apply(out2$nmix$probdraw, 2, effectiveSize)/ apply(out1$nmix$probdraw, 2, effectiveSize), eff_delta_ratio = apply(out2$Deltadraw, 2, effectiveSize) / apply(out1$Deltadraw, 2, effectiveSize), MSE_delta = c(mean((summary1[,2] - as.vector(Delta))^2), mean((summary2[,2] - as.vector(Delta))^2)), MSE_beta = c(mean((beta1 - betatrue)^2), mean((beta2 - betatrue)^2)))

    return(output)
}




# simulation = function(){
#     ## generate data

    ## simulate data
    simlgtdata = NULL
    betatrue = matrix(0, nlgt, p)
    ni = rep(nobs, nlgt)
    for (i in 1:nlgt) {
        betai = Delta%*%Z[i,] + as.vector(rmixture(1,pvec,comps)$x)
        

        ##  
        # Draw betai from lognormal!
        ##

        betai = exp(betai)


        betatrue[i,] = betai
        Xa = matrix(runif(ni[i]*p,min=-1.5,max=0), ncol=p)
        X = createX(p, na=1, nd=NULL, Xa=Xa, Xd=NULL, base=1)
        outa = simmnlwX(ni[i], X, betai)
        simlgtdata[[i]] = list(y=outa$y, X=X, beta=betai)
    }

    ## set parms for priors and Z
    Prior1 = list(ncomp=ncomp)
    keep = 1
    Mcmc1 = list(R=R, keep=keep)
    Data1 = list(p=p, lgtdata=simlgtdata, Z=Z)

    # Gibbs sampler
    time1 = proc.time()
    out1 = bayesm::rhierMnlRwMixtureLogNormal(Data=Data1, Prior=Prior1, Mcmc=Mcmc1)
    time1 = proc.time() - time1
    cat("Summary of Delta draws", fill=TRUE)
    summary1 = summary(out1$Deltadraw, tvalues=as.vector(Delta))
    cat("Summary of Normal Mixture Distribution", fill=TRUE)
    summary(out1$nmix)



    # elliptical slice sampler
    time2 = proc.time()
    out2 = bayesm::rhierMnlRwMixtureLogNormal_slice(Data=Data1, Prior=Prior1, Mcmc=Mcmc1)
    # out2 = out1
    time2 = proc.time() - time2
    # cat("Summary of Delta draws", fill=TRUE)
    summary2 = summary(out2$Deltadraw, tvalues=as.vector(Delta))
    cat("Summary of Normal Mixture Distribution", fill=TRUE)
    summary(out2$nmix)

    output = summary_posterior(out1, out2, betatrue)
    output$time = list(time1 = time1, time2 = time2, ratio = time2 / time1)

#     return(output)
# }


par(mfrow = c(2,2))
beta1 = apply(out1$betadraw[,,(0.2*R):R], c(1,2), mean)
beta2 = apply(out2$betadraw[,,(0.2*R):R], c(1,2), mean)

plot(as.vector(betatrue), as.vector(beta1))
abline(0, 1)
plot(as.vector(betatrue), as.vector(beta2))
abline(0, 1)
plot(as.vector(beta1), as.vector(beta2))
abline(0, 1)

