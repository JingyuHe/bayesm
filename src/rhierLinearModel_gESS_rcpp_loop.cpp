#include "bayesm.h"

double log_likelihood_reg_gESS(vec const &y, mat const &X, vec const &beta, double sigma2)
{
  // log likelihood of univariate regression y ~ Xbeta + epsilon
  size_t n = X.n_rows;
  double output = 0.0;
  double mu = 0.0;
  mat Xbeta = X * beta;
  for (size_t i = 0; i < n; i++)
  {
    // loop over all data
    // output = output - 0.5 * log(2 * M_PI) - log(sigma2) - 0.5 * pow(y(i) - Xbeta(i, 0), 2) / sigma2;

    // ignore constant
    output = output - 0.5 * pow(y(i) - Xbeta(i, 0), 2) / sigma2;
  }
  return output;
}

unireg gESS_draw_hierLinearModel(vec const &y, mat const &X, vec const &beta_ini, vec const &beta_hat, double tau, mat const &incroot, mat const &incroot_inv, vec const &mu_ellipse, mat const& rootpi)
{

  /*
    sample via generalized elliptical slice sampler
    Input : beta_ini, vector of initial value
            beta_hat, mean of beta (likelihood function)
            
            the ellipse is N(mu_ellipse, Omgea)
            where Omega = incroot * trans(incroot)
            Omega^{-1} = incroot_inv * trans(incroot_inv)

            the target distribution here is N(y; beta X, sigma^2) N(B; ZDelta, V_beta)
            and V_beta = rootpi * trans(rootpi)
    */
  unireg out_struct;
  // subtract mean from the initial value, sample the deviation from mean
  vec beta = beta_ini - mu_ellipse;
  // draw the auxillary vector
  vec eps = arma::randn<vec>(incroot.n_cols);
  vec nu = incroot * eps;

  // compute the prior threshold
  double u = as_scalar(randu<vec>(1));

  double priorcomp = log_likelihood_reg_gESS(y, X, beta_ini, tau) + lndMvn(beta_ini, beta_hat, rootpi) - lndMvst(beta_ini, 2.0, mu_ellipse, incroot_inv, false);

  double ly = priorcomp + log(u); // here is log likelihood

  // elliptical slice sampling
  double thetaprop = as_scalar(randu<vec>(1)) * 2.0 * M_PI;
  vec betaprop = beta * cos(thetaprop) + nu * sin(thetaprop);
  double thetamin = thetaprop - 2.0 * M_PI;
  double thetamax = thetaprop;

  double compll;

  compll = log_likelihood_reg_gESS(y, X, betaprop + mu_ellipse, tau) + lndMvn(betaprop + mu_ellipse, beta_hat, rootpi) - lndMvst(betaprop + mu_ellipse, 2.0, mu_ellipse, incroot_inv, false);

  while (compll < ly)
  {
    // count ++ ;

    if (thetaprop < 0)
    {
      thetamin = thetaprop;
    }
    else
    {
      thetamax = thetaprop;
    }

    // runif(thetamin, thetamax)
    thetaprop = as_scalar(randu<vec>(1)) * (thetamax - thetamin) + thetamin;

    betaprop = beta * cos(thetaprop) + nu * sin(thetaprop);

    compll = log_likelihood_reg_gESS(y, X, betaprop + mu_ellipse, tau) + lndMvn(betaprop + mu_ellipse, beta_hat, rootpi) - lndMvst(betaprop + mu_ellipse, 2.0, mu_ellipse, incroot_inv, false);
  }

  // accept the proposal
  beta = betaprop;

  out_struct.beta = beta + mu_ellipse;
  return out_struct;
}

// [[Rcpp::export]]
List rhierLinearModel_gESS_rcpp_loop(List const &regdata, mat const &Z, mat const &Deltabar, mat const &A, double nu,
                                     mat const &V, double nu_e, vec const &ssq, vec tau, mat Delta, mat Vbeta, int R, int keep, int nprint)
{

  // Keunwoo Kim 09/16/2014

  // Purpose: run hiearchical regression model

  // Arguments:
  //   Data list of regdata,Z
  //     regdata is a list of lists each list with members y, X
  //        e.g. regdata[[i]]=list(y=y,X=X)
  //     X has nvar columns
  //     Z is nreg=length(regdata) x nz

  //   Prior list of prior hyperparameters
  //     Deltabar,A, nu.e,ssq,nu,V
  //          note: ssq is a nreg x 1 vector!

  //   Mcmc
  //     list of Mcmc parameters
  //     R is number of draws
  //     keep is thining parameter -- keep every keepth draw
  //     nprint - print estimated time remaining on every nprint'th draw

  // Output:
  //   list of
  //   betadraw -- nreg x nvar x R/keep array of individual regression betas
  //   taudraw -- R/keep x nreg  array of error variances for each regression
  //   Deltadraw -- R/keep x nz x nvar array of Delta draws
  //   Vbetadraw -- R/keep x nvar*nvar array of Vbeta draws

  // Model:
  // nreg regression equations
  //        y_i = X_ibeta_i + epsilon_i
  //        epsilon_i ~ N(0,tau_i)
  //             nvar X vars in each equation

  // Prior:
  //        tau_i ~ nu.e*ssq_i/chisq(nu.e)  tau_i is the variance of epsilon_i
  //        beta_i ~ N(ZDelta[i,],V_beta)
  //               Note:  ZDelta is the matrix Z * Delta; [i,] refers to ith row of this product!

  //          vec(Delta) | V_beta ~ N(vec(Deltabar),Vbeta (x) A^-1)
  //          V_beta ~ IW(nu,V)  or V_beta^-1 ~ W(nu,V^-1)
  //              Delta, Deltabar are nz x nvar
  //              A is nz x nz
  //              Vbeta is nvar x nvar

  //          NOTE: if you don't have any z vars, set Z=iota (nreg x 1)

  // Update Note:
  //        (Keunwoo Kim 04/07/2015)
  //        Changed "rmultireg" to return List object, which is the original function.
  //        Efficiency is almost same as when the output is a struct object.
  //        Nothing different from "rmultireg1" in the previous R version.

  int reg, mkeep;
  mat Abeta, betabar, ucholinv, Abetabar;
  List regdatai, rmregout;
  unireg regout_struct;

  mat incroot_ellipse;
  mat ucholinv_ellipse;

  int nreg = regdata.size();
  int nvar = V.n_cols;
  int nz = Z.n_cols;

  // convert List to std::vector of struct
  std::vector<moments> regdata_vector;
  moments regdatai_struct;

  // store vector with struct
  for (reg = 0; reg < nreg; reg++)
  {
    regdatai = regdata[reg];

    regdatai_struct.y = as<vec>(regdatai["y"]);
    regdatai_struct.X = as<mat>(regdatai["X"]);
    regdatai_struct.XpX = as<mat>(regdatai["XpX"]);
    regdatai_struct.Xpy = as<vec>(regdatai["Xpy"]);
    regdata_vector.push_back(regdatai_struct);
  }

  mat betas(nreg, nvar);
  mat Vbetadraw(R / keep, nvar * nvar);
  mat Deltadraw(R / keep, nz * nvar);
  mat taudraw(R / keep, nreg);
  cube betadraw(nreg, nvar, R / keep);

  if (nprint > 0)
    startMcmcTimer();

  mat L;
  mat oldbetas(nreg, nvar);
  vec oldtau = ones<vec>(tau.n_elem);
  vec oldbeta_temp;
  vec betabar_temp;
  double s;
  mat incroot;
  mat incroot_inv;
  double lambda;
  double ss1, ss2;
  double ss = 2;
  vec mu_ellipse;
  vec temp;

  //start main iteration loop
  for (int rep = 0; rep < R; rep++)
  {

    // compute the inverse of Vbeta
    // incroot * trans*incroot = Vbeta
    incroot = chol(Vbeta, "lower");

    ucholinv = solve(trimatu(trans(incroot)), eye(nvar, nvar)); //trimatu interprets the matrix as upper triangular and makes solve more efficient

    betabar = Z * Delta;


    // Abeta = ucholinv * trans(ucholinv);
    // Abetabar = Abeta * trans(betabar);
    
    // loop over all regressions
    // can be replaced by elliptical slice sampler
    // the ellipce is defined as N(Zdelta_i, lambda * Vbeta)
    for (reg = 0; reg < nreg; reg++)
    {

        // sampling residual term of elliptical slice sampler
        mu_ellipse = trans(betabar(reg, span::all));

        // temp = rootpi' * X
        // trans(temp) * temp = X' * rootpi * rootpi' * X = X' * Sigma^{-1} * X
        temp = vectorise(trans(ucholinv) * (vectorise(oldbetas(reg, span::all)) - mu_ellipse));
        ss1 = (ss + mu_ellipse.n_elem) / 2.0;
        ss2 = (0.5 * (ss + (trans(temp) * temp)))[0];
        lambda = 1.0 / randg<vec>(1, distr_param(ss1, 1.0 / ss2))[0];

        incroot_ellipse = incroot * sqrt(lambda);
        ucholinv_ellipse = ucholinv / sqrt(lambda);

        // sampling beta
        regout_struct = gESS_draw_hierLinearModel(regdata_vector[reg].y, regdata_vector[reg].X, trans(oldbetas(reg, span::all)), trans(betabar(reg, span::all)), oldtau[reg], incroot_ellipse, ucholinv_ellipse, mu_ellipse, ucholinv);

        betas(reg, span::all) = trans(regout_struct.beta);
        // sampling tau
        s = sum(square(regdata_vector[reg].y - regdata_vector[reg].X * regout_struct.beta));
        tau[reg] = (s + nu_e * ssq[reg]) / rchisq(1, nu_e + regdata_vector[reg].y.n_elem)[0]; //rchisq returns a vectorized object, so using [0] allows for the conversion to double
    }

    //draw Vbeta, Delta | {beta_i}
    rmregout = rmultireg(betas, Z, Deltabar, A, nu, V);
    Vbeta = as<mat>(rmregout["Sigma"]); //conversion from Rcpp to Armadillo requires explict declaration of variable type using as<>
    Delta = as<mat>(rmregout["B"]);

    //print time to completion and draw # every nprint'th draw
    if (nprint > 0)
      if ((rep + 1) % nprint == 0)
        infoMcmcTimer(rep, R);

    if ((rep + 1) % keep == 0)
    {
      mkeep = (rep + 1) / keep;
      Vbetadraw(mkeep - 1, span::all) = trans(vectorise(Vbeta));
      Deltadraw(mkeep - 1, span::all) = trans(vectorise(Delta));
      taudraw(mkeep - 1, span::all) = trans(tau);
      betadraw.slice(mkeep - 1) = betas;

      oldtau = tau;
      oldbetas = betas;
    }
  }

  if (nprint > 0)
    endMcmcTimer();

  return List::create(
      Named("Vbetadraw") = Vbetadraw,
      Named("Deltadraw") = Deltadraw,
      Named("betadraw") = betadraw,
      Named("taudraw") = taudraw);
}
