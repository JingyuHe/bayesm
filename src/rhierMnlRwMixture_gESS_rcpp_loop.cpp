#include "bayesm.h"

mnlMetropOnceOut mnlMetropOnce_con_MH_gESS(vec const &y, mat const &X, vec const &oldbeta,
                                           double oldll, double s, mat const &incroot,
                                           vec const &betabar, mat const &rootpi, vec const &SignRes = NumericVector::create(2))
{
    // Wayne Taylor 10/01/2014

    // function to execute rw metropolis for the MNL
    // y is n vector with element = 1,...,j indicating which alt chosen
    // X is nj x k matrix of xvalues for each of j alt on each of n occasions
    // RW increments are N(0,s^2*t(inc.root)%*%inc.root)
    // prior on beta is N(betabar,Sigma)  Sigma^-1=rootpi*t(rootpi)
    //  inc.root, rootpi are upper triangular
    //  this means that we are using the UL decomp of Sigma^-1 for prior
    // oldbeta is the current

    mnlMetropOnceOut out_struct;

    double unif;
    vec betadraw, alphaminv;

    int stay = 0;

    // Note trans(incroot) here
    // The actual covariance is trans(incroot) * incroot = inv(H + Vb^{-1})
    vec betac = oldbeta + s * trans(incroot) * as<vec>(rnorm(X.n_cols));

    double cll = llmnl_con(betac, y, X, SignRes);
    double clpost = cll + lndMvn(betac, betabar, rootpi);
    double ldiff = clpost - oldll - lndMvn(oldbeta, betabar, rootpi);
    alphaminv << 1 << exp(ldiff);
    double alpha = min(alphaminv);

    if (alpha < 1)
    {
        unif = as_scalar(vec(runif(1)));
    }
    else
    {
        unif = 0;
    }
    if (unif <= alpha)
    {
        betadraw = betac;
        oldll = cll;
    }
    else
    {
        betadraw = oldbeta;
        stay = 1;
    }

    out_struct.betadraw = betadraw;
    out_struct.stay = stay;
    out_struct.oldll = oldll;

    return (out_struct);
}

mnlMetropOnceOut gESS_draw_hierLogitMixture(vec const &y, mat const &X, vec const &beta_ini, vec const &beta_hat, mat const &rootpi, double oldll, mat const &incroot, mat const &incroot_inv, vec const &mu_ellipse, vec const &SignRes = NumericVector::create(2))
{

    /*
    sample via elliptical slice sampler
    Input : beta_ini, vector of initial value
            beta_hat, mean of beta (likelihood function)
            L, Cholesky factor (lower triangular LL' = Sigma) of covariance matrix of normal part
  */
    mnlMetropOnceOut out_struct;
    // subtract mean from the initial value, sample the deviation from mean
    vec beta = beta_ini - beta_hat;

    // draw the auxillary vector
    vec eps = arma::randn<vec>(incroot.n_cols);
    vec nu = incroot * eps;

    // compute the prior threshold
    double u = as_scalar(randu<vec>(1));

    double priorcomp = oldll; //llmnl_con(beta_ini, y, X, SignRes);

    // cout << oldll << endl;
    // cout << llmnl(beta_ini, y, X) << endl;

    // double ly = priorcomp + log(u); // here is log likelihood

    double ly = llmnl_con(beta_ini, y, X, SignRes) + lndMvn(beta_ini, beta_hat, rootpi) - lndMvst(beta_ini, 2.0, mu_ellipse, incroot_inv, false);

    ly = ly + log(u);

    // elliptical slice sampling
    double thetaprop = as_scalar(randu<vec>(1)) * 2.0 * M_PI;
    vec betaprop = beta * cos(thetaprop) + nu * sin(thetaprop);
    double thetamin = thetaprop - 2.0 * M_PI;
    double thetamax = thetaprop;

    double compll;

    compll = llmnl_con(betaprop + beta_hat, y, X, SignRes) + lndMvn(betaprop + beta_hat, beta_hat, rootpi) - lndMvst(betaprop + mu_ellipse, 2.0, mu_ellipse, incroot_inv, false);

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

        compll = llmnl_con(betaprop + beta_hat, y, X, SignRes) + lndMvn(betaprop + beta_hat, beta_hat, rootpi) - lndMvst(betaprop + mu_ellipse, 2.0, mu_ellipse, incroot_inv, false);
    }

    // accept the proposal
    beta = betaprop;

    oldll = compll;
    // cout << "saved value " << oldll << endl;
    // cout << "-----" << endl;
    // add the mean back
    out_struct.betadraw = beta + beta_hat;
    out_struct.oldll = oldll;
    return out_struct;
}

//MAIN FUNCTION-------------------------------------------------------------------------------------

//[[Rcpp::export]]
List rhierMnlRwMixture_gESS_rcpp_loop(List const &lgtdata, mat const &Z,
                                      vec const &deltabar, mat const &Ad, mat const &mubar, mat const &Amu,
                                      double nu, mat const &V, double s,
                                      int R, int keep, int nprint, bool drawdelta,
                                      mat olddelta, vec const &a, vec oldprob, mat oldbetas, vec ind, vec const &SignRes, double p_MH, bool MH_burnin, bool fix_p_burnin)
{
    cout << "--------------------" << endl;
    cout << "mixture of MH and slice sampler" << endl;
    cout << "--------------------" << endl;

    // Wayne Taylor 10/01/2014

    int nlgt = lgtdata.size();
    int nvar = V.n_cols;
    int nz = Z.n_cols;

    mat rootpi, betabar, ucholinv, incroot, L;
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


    mat Sigma;
    vec mu_ellipse;
    mat cov_ellipse;
    mat incroot_inv;

    double ss1;
    double ss2;
    vec temp;
    double lambda;
    double ss = 2; 

    mat scale = zeros<mat>(R, nlgt);

    if (nprint > 0)
        startMcmcTimer();

    // cout << "old probs" << oldprob << endl;

    for (int rep = 0; rep < R; rep++)
    {

        //first draw comps,ind,p | {beta_i}, delta
        // ind,p need initialization comps is drawn first in sub-Gibbs
        List mgout;
        if (drawdelta)
        {
            olddelta.reshape(nvar, nz);

            if (rep < 2000 && fix_p_burnin)
            {
                mgout = rmixGibbs_fix_p(oldbetas - Z * trans(olddelta), mubar, Amu, nu, V, a, oldprob, ind);
            }
            else
            {
                mgout = rmixGibbs(oldbetas - Z * trans(olddelta), mubar, Amu, nu, V, a, oldprob, ind);
            }
        }
        else
        {
            if (rep < 2000 && fix_p_burnin)
            {
                mgout = rmixGibbs_fix_p(oldbetas, mubar, Amu, nu, V, a, oldprob, ind);
            }
            else
            {
                mgout = rmixGibbs(oldbetas, mubar, Amu, nu, V, a, oldprob, ind);
            }
        }

        List oldcomp = mgout["comps"];
        oldprob = as<vec>(mgout["p"]); //conversion from Rcpp to Armadillo requires explict declaration of variable type using as<>
        ind = as<vec>(mgout["z"]);

        //now draw delta | {beta_i}, ind, comps
        if (drawdelta)
            olddelta = drawDelta(Z, oldbetas, ind, oldcomp, deltabar, Ad);

        //loop over all LGT equations drawing beta_i | ind[i],z[i,],mu[ind[i]],rooti[ind[i]]
        for (int lgt = 0; lgt < nlgt; lgt++)
        {
            List oldcomplgt = oldcomp[ind[lgt] - 1];

            // rooti * trans(rooti) = sigma^{-1}  !!! lower cholesky root of sigma Inverse
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
                oldll[lgt] = llmnl_con(vectorise(oldbetas(lgt, span::all)), lgtdata_vector[lgt].y, lgtdata_vector[lgt].X, SignRes);

            if (rep < 2000 && MH_burnin)
            {
                // burnin period, MH
                // ucholinv * trans(ucholinv) = inv(H + Vb^{-1})
                ucholinv = solve(trimatu(chol(lgtdata_vector[lgt].hess + rootpi * trans(rootpi))), eye(nvar, nvar)); //trimatu interprets the matrix as upper triangular and makes solve more efficient

                // t(incroot) * incroot = inv(H + Vb^{-1})
                // here incroot is upper triangular, different from gESS
                incroot = chol(ucholinv * trans(ucholinv));

                metropout_struct = mnlMetropOnce_con_MH_gESS(lgtdata_vector[lgt].y, lgtdata_vector[lgt].X, vectorise(oldbetas(lgt, span::all)), oldll[lgt], s, incroot, betabar, rootpi, SignRes);
            }
            else
            {
                // generalized elliptical slice Sampler
                mu_ellipse = betabar;

                // sampling s, generalized ESS
                // temp = rootpi' * X
                // trans(temp) * temp = X' * rootpi * rootpi' * X = X' * Sigma^{-1} * X
                temp = vectorise(trans(rootpi) * (vectorise(oldbetas(lgt, span::all)) - mu_ellipse));

                ss1 = (ss + betabar.n_elem) / 2.0;
                ss2 = (0.5 * (ss + (trans(temp) * temp)))[0];

                // draw scale parameter from inverse gamma
                lambda = 1.0 / randg<vec>(1, distr_param(ss1, 1.0 / ss2))[0];

                scale(rep, lgt) = lambda;

                // cov_ellipse = inv(rootpi * trans(rootpi));

                // L = inv(rootpi * trans(rootpi));
                // L * trans(L) = Sigma
                // L = chol(L, "lower");

                L = trans(inv(rootpi));
                incroot = L * sqrt(lambda);
                incroot_inv = rootpi / sqrt(lambda);

                // incroot = chol(cov_ellipse, "lower") * sqrt(lambda);

                // incroot_inv = chol(inv(cov_ellipse), "lower") / sqrt(lambda);

                metropout_struct = gESS_draw_hierLogitMixture(lgtdata_vector[lgt].y, lgtdata_vector[lgt].X, vectorise(oldbetas(lgt, span::all)), betabar, rootpi, oldll[lgt], incroot, incroot_inv, mu_ellipse, SignRes);
            }

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
            Named("SignRes") = SignRes,
            Named("scale_of_gESS") = scale));
    }
    else
    {
        return (List::create(
            Named("betadraw") = betadraw,
            Named("nmix") = nmix,
            Named("loglike") = loglike,
            Named("SignRes") = SignRes,
            Named("scale_of_gESS") = scale));
    }
}
