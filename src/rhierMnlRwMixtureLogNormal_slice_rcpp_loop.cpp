#include "bayesm.h"

mnlMetropOnceOut ESS_draw_hierLogitMixtureLogNormal(vec const &y, mat const &X, vec const &beta_ini, vec const &beta_hat, mat const &rootpi, double oldll, mat const &incroot, mat const &incroot_inv, vec const &mu_ellipse, vec const &SignRes = NumericVector::create(2))
{
    // since this is log normal distributions
    // E[X] = exp(beta_hat + diag(Sigma) / 2)

    /*
    sample via elliptical slice sampler
    Input : beta_ini, vector of initial value
            beta_hat, mean of beta (likelihood function)
            L, Cholesky factor (lower triangular LL' = Sigma) of covariance matrix of normal part
  */
    mnlMetropOnceOut out_struct;

    // subtract mean from the initial value, sample the deviation from mean
    vec beta = beta_ini - mu_ellipse;

    // draw the auxillary vector
    vec eps = arma::randn<vec>(incroot.n_cols);
    // vec nu = L * eps;

    vec nu = incroot * eps;

    // compute the prior threshold
    double u = as_scalar(randu<vec>(1));


    double ly = llmnl_con(beta_ini, y, X, SignRes) + lndLogMvn(beta_ini, beta_hat, rootpi) - lndMvn(beta_ini, mu_ellipse, incroot_inv);

    // slice sampling
    ly = ly + log(u);

    // elliptical slice sampling
    double thetaprop = as_scalar(randu<vec>(1)) * 2.0 * M_PI;
    vec betaprop = beta * cos(thetaprop) + nu * sin(thetaprop);
    double thetamin = thetaprop - 2.0 * M_PI;
    double thetamax = thetaprop;

    double compll;

    // now the "likelihood" to evaluate is MNL likelihood * lognormal density
    compll = llmnl_con(betaprop + mu_ellipse, y, X, SignRes) + lndLogMvn(betaprop + mu_ellipse, beta_hat, rootpi) - lndMvn(betaprop + mu_ellipse, mu_ellipse, incroot_inv);

    // cout << "--------------------" << endl;

    while (compll < ly)
    {
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

        compll = llmnl_con(betaprop + mu_ellipse, y, X, SignRes) + lndLogMvn(betaprop + mu_ellipse, beta_hat, rootpi) - lndMvn(betaprop + mu_ellipse, mu_ellipse, incroot_inv);

    }

    // accept the proposal
    beta = betaprop;

    oldll = compll;
    // cout << "saved value " << oldll << endl;
    // cout << "-----" << endl;
    // add the mean back
    out_struct.betadraw = beta + mu_ellipse;
    out_struct.oldll = oldll;
    return out_struct;
}

//MAIN FUNCTION-------------------------------------------------------------------------------------

//[[Rcpp::export]]
List rhierMnlRwMixtureLogNormal_slice_rcpp_loop(List const &lgtdata, mat const &Z,
                                                vec const &deltabar, mat const &Ad, mat const &mubar, mat const &Amu,
                                                double nu, mat const &V, double s,
                                                int R, int keep, int nprint, bool drawdelta,
                                                mat olddelta, vec const &a, vec oldprob, mat oldbetas, vec ind, vec const &SignRes, double p_MH)
{
    cout << "--------------------" << endl;
    cout << "mixture of MH and slice sampler" << endl;
    cout << "--------------------" << endl;

    // Wayne Taylor 10/01/2014

    int nlgt = lgtdata.size();
    int nvar = V.n_cols;
    int nz = Z.n_cols;

    mat rootpi, betabar, ucholinv, L;
    int mkeep;
    mnlMetropOnceOut metropout_struct;
    List lgtdatai, nmix;

    // convert List to std::vector of struct
    std::vector<moments> lgtdata_vector;
    moments lgtdatai_struct;
    for (int lgt = 0; lgt < nlgt; lgt++)
    {
        lgtdatai = lgtdata[lgt];

        lgtdatai_struct.y = as<vec>(lgtdatai["y"]);
        lgtdatai_struct.X = as<mat>(lgtdatai["X"]);
        lgtdatai_struct.hess = as<mat>(lgtdatai["hess"]);
        lgtdata_vector.push_back(lgtdatai_struct);
    }

    // allocate space for draws
    vec oldll = zeros<vec>(nlgt);
    cube betadraw(nlgt, nvar, R / keep);
    mat probdraw(R / keep, oldprob.size());
    vec loglike(R / keep);
    mat Deltadraw(1, 1);
    if (drawdelta)
        Deltadraw.zeros(R / keep, nz * nvar); //enlarge Deltadraw only if the space is required
    List compdraw(R / keep);

    // initialize beta at a positive value
    betadraw.fill(1);
    oldbetas.fill(1);

    mat Sigma;
    vec mu_ellipse;
    mat cov_ellipse;

    mat incroot;
    mat incroot_inv;


    if (nprint > 0)
        startMcmcTimer();

    for (int rep = 0; rep < R; rep++)
    {

        //first draw comps,ind,p | {beta_i}, delta
        // ind,p need initialization comps is drawn first in sub-Gibbs
        List mgout;
        if (drawdelta)
        {
            olddelta.reshape(nvar, nz);
            mgout = rmixGibbs(log(oldbetas) - Z * trans(olddelta), mubar, Amu, nu, V, a, oldprob, ind);
        }
        else
        {
            mgout = rmixGibbs(log(oldbetas), mubar, Amu, nu, V, a, oldprob, ind);
        }

        List oldcomp = mgout["comps"];
        oldprob = as<vec>(mgout["p"]); //conversion from Rcpp to Armadillo requires explict declaration of variable type using as<>
        ind = as<vec>(mgout["z"]);

        //now draw delta | {beta_i}, ind, comps
        if (drawdelta)
            olddelta = drawDelta(Z, log(oldbetas), ind, oldcomp, deltabar, Ad);

        //loop over all LGT equations drawing beta_i | ind[i],z[i,],mu[ind[i]],rooti[ind[i]]
        for (int lgt = 0; lgt < nlgt; lgt++)
        {
            List oldcomplgt = oldcomp[ind[lgt] - 1];

            // rooti * trans(rooti) = sigma^{-1}  !!! cholesky root of sigma INVERSE
            rootpi = as<mat>(oldcomplgt[1]);

            //note: beta_i = Delta*z_i + u_i  Delta is nvar x nz
            if (drawdelta)
            {
                olddelta.reshape(nvar, nz);
                betabar = as<vec>(oldcomplgt[0]) + olddelta * vectorise(Z(lgt, span::all));
            }
            else
            {
                betabar = as<vec>(oldcomplgt[0]);
            }

            if (rep == 0)
                oldll[lgt] = llmnl_con(vectorise(oldbetas(lgt, span::all)), lgtdata_vector[lgt].y, lgtdata_vector[lgt].X, SignRes) + lndLogMvn(vectorise(oldbetas(lgt, span::all)), betabar, rootpi);
            ;


            // rootpi * trans(rootpi) = inv(Sigma)
            Sigma = inv(rootpi * trans(rootpi));


            // expectation and covariance of the proposal ellipse 
            // mu_ellipse = exp(betabar + Sigma.diag() / 2.0);

            mu_ellipse = betabar;


            // % is elementwise multiplication
            // cov_ellipse = (mu_ellipse * trans(mu_ellipse)) % (exp(Sigma) - ones(Sigma.n_rows, Sigma.n_cols));

            cov_ellipse = Sigma;

            // incroot * trans(incroot) = cov_ellipse
            incroot = chol(cov_ellipse, "lower");

            // incroot_inv * trans(incroot_inv) = inv(cov_ellipse), used for calculating likelihood
            incroot_inv = chol(inv(cov_ellipse), "lower");


            // betabar is mean AFTER taking log

            metropout_struct = ESS_draw_hierLogitMixtureLogNormal(lgtdata_vector[lgt].y, lgtdata_vector[lgt].X, vectorise(oldbetas(lgt, span::all)), betabar, rootpi, oldll[lgt], incroot, incroot_inv, mu_ellipse, SignRes);

            oldbetas(lgt, span::all) = trans(metropout_struct.betadraw);
            oldll[lgt] = metropout_struct.oldll;
        }

        //print time to completion and draw # every nprint'th draw
        if (nprint > 0)
            if ((rep + 1) % nprint == 0)
                infoMcmcTimer(rep, R);

        if ((rep + 1) % keep == 0)
        {
            mkeep = (rep + 1) / keep;
            betadraw.slice(mkeep - 1) = oldbetas;
            probdraw(mkeep - 1, span::all) = trans(oldprob);
            loglike[mkeep - 1] = sum(oldll);
            if (drawdelta)
                Deltadraw(mkeep - 1, span::all) = trans(vectorise(olddelta));
            compdraw[mkeep - 1] = oldcomp;
        }
    }

    if (nprint > 0)
        endMcmcTimer();

    nmix = List::create(Named("probdraw") = probdraw,
                        Named("zdraw") = R_NilValue, //sets the value to NULL in R
                        Named("compdraw") = compdraw);

    //ADDED FOR CONSTRAINTS
    //If there are sign constraints, return f(betadraws) as "betadraws"
    //conStatus will be set to true if SignRes has any non-zero elements
    bool conStatus = any(SignRes);

    if (conStatus)
    {
        int SignResSize = SignRes.size();

        //loop through each sign constraint
        for (int i = 0; i < SignResSize; i++)
        {

            //if there is a constraint loop through each slice of betadraw
            if (SignRes[i] != 0)
            {
                for (int s = 0; s < R / keep; s++)
                {
                    betadraw(span(), span(i), span(s)) = SignRes[i] * exp(betadraw(span(), span(i), span(s)));
                }
            }

        } //end loop through SignRes
    }

    if (drawdelta)
    {
        return (List::create(
            Named("Deltadraw") = Deltadraw,
            Named("betadraw") = betadraw,
            Named("nmix") = nmix,
            Named("loglike") = loglike,
            Named("SignRes") = SignRes));
    }
    else
    {
        return (List::create(
            Named("betadraw") = betadraw,
            Named("nmix") = nmix,
            Named("loglike") = loglike,
            Named("SignRes") = SignRes));
    }
}
