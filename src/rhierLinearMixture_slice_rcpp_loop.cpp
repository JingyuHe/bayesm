#include "bayesm.h"

unireg ESS_draw_rhierLinearMixture(vec const &y, mat const &X, vec const &beta_ini, vec const &beta_hat, mat const &L, double tau)
{

    /*
    sample via elliptical slice sampler
    Input : beta_ini, vector of initial value
            beta_hat, mean of beta (likelihood function)
            L, Cholesky factor (lower triangular LL' = Sigma) of covariance matrix of normal part
    */
    unireg out_struct;
    // subtract mean from the initial value, sample the deviation from mean
    vec beta = beta_ini - beta_hat;
    // draw the auxillary vector
    vec eps = arma::randn<vec>(L.n_cols);
    vec nu = L * eps;

    // compute the prior threshold
    double u = as_scalar(randu<vec>(1));

    double priorcomp = log_likelihood_reg(y, X, beta_ini, tau);

    double ly = priorcomp + log(u); // here is log likelihood

    // elliptical slice sampling
    double thetaprop = as_scalar(randu<vec>(1)) * 2.0 * M_PI;
    vec betaprop = beta * cos(thetaprop) + nu * sin(thetaprop);
    double thetamin = thetaprop - 2.0 * M_PI;
    double thetamax = thetaprop;

    double compll;

    compll = log_likelihood_reg(y, X, betaprop + beta_hat, tau);

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

        compll = log_likelihood_reg(y, X, betaprop + beta_hat, tau);
    }

    // accept the proposal
    beta = betaprop;

    out_struct.beta = beta + beta_hat;
    return out_struct;
}

//[[Rcpp::export]]
List rhierLinearMixture_slice_rcpp_loop(List const &regdata, mat const &Z,
                                        vec const &deltabar, mat const &Ad, mat const &mubar, mat const &Amu,
                                        double nu, mat const &V, double nu_e, vec const &ssq,
                                        int R, int keep, int nprint, bool drawdelta,
                                        mat olddelta, vec const &a, vec oldprob, vec ind, vec tau)
{

    // Wayne Taylor 10/02/2014

    int nreg = regdata.size();
    int nvar = V.n_cols;
    int nz = Z.n_cols;

    mat rootpi, betabar, Abeta, Abetabar;
    int mkeep;
    unireg runiregout_struct;
    List regdatai, nmix;

    // convert List to std::vector of type "moments"
    std::vector<moments> regdata_vector;
    moments regdatai_struct;

    // store vector with struct
    for (int reg = 0; reg < nreg; reg++)
    {
        regdatai = regdata[reg];

        regdatai_struct.y = as<vec>(regdatai["y"]);
        regdatai_struct.X = as<mat>(regdatai["X"]);
        regdatai_struct.XpX = as<mat>(regdatai["XpX"]);
        regdatai_struct.Xpy = as<vec>(regdatai["Xpy"]);
        regdata_vector.push_back(regdatai_struct);
    }

    // allocate space for draws
    mat oldbetas = zeros<mat>(nreg, nvar);
    mat taudraw(R / keep, nreg);
    cube betadraw(nreg, nvar, R / keep);
    mat probdraw(R / keep, oldprob.size());
    mat Deltadraw(1, 1);
    if (drawdelta)
        Deltadraw.zeros(R / keep, nz * nvar); //enlarge Deltadraw only if the space is required
    List compdraw(R / keep);

    if (nprint > 0)
        startMcmcTimer();


    mat incroot;
    double s;

    for (int rep = 0; rep < R; rep++)
    {

        //first draw comps,ind,p | {beta_i}, delta
        // ind,p need initialization comps is drawn first in sub-Gibbs
        List mgout;
        if (drawdelta)
        {
            olddelta.reshape(nvar, nz);
            mgout = rmixGibbs(oldbetas - Z * trans(olddelta), mubar, Amu, nu, V, a, oldprob, ind);
        }
        else
        {
            mgout = rmixGibbs(oldbetas, mubar, Amu, nu, V, a, oldprob, ind);
        }

        List oldcomp = mgout["comps"];
        oldprob = as<vec>(mgout["p"]); //conversion from Rcpp to Armadillo requires explict declaration of variable type using as<>
        ind = as<vec>(mgout["z"]);     //conversion from Rcpp to Armadillo requires explict declaration of variable type using as<>

        //now draw delta | {beta_i}, ind, comps
        if (drawdelta)
            olddelta = drawDelta(Z, oldbetas, ind, oldcomp, deltabar, Ad);

        //loop over all regression equations drawing beta_i | ind[i],z[i,],mu[ind[i]],rooti[ind[i]]
        for (int reg = 0; reg < nreg; reg++)
        {
            List oldcompreg = oldcomp[ind[reg] - 1];
            rootpi = as<mat>(oldcompreg[1]);

            //note: beta_i = Delta*z_i + u_i  Delta is nvar x nz
            if (drawdelta)
            {
                olddelta.reshape(nvar, nz);
                betabar = as<vec>(oldcompreg[0]) + olddelta * vectorise(Z(reg, span::all));
            }
            else
            {
                betabar = as<vec>(oldcompreg[0]);
            }

            // I think this is a bug
            // Abeta = trans(rootpi) * rootpi;

            // Abeta = rootpi * trans(rootpi);
            // Abetabar = Abeta * betabar;

            // incroot = trans(inv(rootpi));

            incroot = trans(as<mat>(oldcompreg[2]));
            // cout << "incroot" << incroot << endl;
            // cout << "IW_chol" << as<mat>(oldcompreg[2]) << endl;

            // incroot * trans(incroot) = Sigma
            // incroot = chol(Sigma, "lower");
            // cout <<  trans(oldbetas(reg, span::all)) << endl;
            // cout << trans(betabar(reg, span::all)) << endl;

            runiregout_struct = ESS_draw_rhierLinearMixture(regdata_vector[reg].y, regdata_vector[reg].X, trans(oldbetas(reg, span::all)), betabar, incroot, tau[reg]);

            oldbetas(reg, span::all) = trans(runiregout_struct.beta);

            // sampling tau
            s = sum(square(regdata_vector[reg].y - regdata_vector[reg].X * runiregout_struct.beta));
            tau[reg] = (s + nu_e * ssq[reg]) / rchisq(1, nu_e + regdata_vector[reg].y.n_elem)[0];
        }

        //print time to completion and draw # every nprint'th draw
        if (nprint > 0)
            if ((rep + 1) % nprint == 0)
                infoMcmcTimer(rep, R);

        if ((rep + 1) % keep == 0)
        {
            mkeep = (rep + 1) / keep;
            taudraw(mkeep - 1, span::all) = trans(tau);
            betadraw.slice(mkeep - 1) = oldbetas;
            probdraw(mkeep - 1, span::all) = trans(oldprob);
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

    if (drawdelta)
    {
        return (List::create(
            Named("taudraw") = taudraw,
            Named("Deltadraw") = Deltadraw,
            Named("betadraw") = betadraw,
            Named("nmix") = nmix));
    }
    else
    {
        return (List::create(
            Named("taudraw") = taudraw,
            Named("betadraw") = betadraw,
            Named("nmix") = nmix));
    }
}