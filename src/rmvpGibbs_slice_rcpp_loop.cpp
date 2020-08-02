#include "bayesm.h"



double prior_function_mvp(vec const& beta, int const& p, ivec const& y){
  /*
    "prior function" for elliptical slice sampler
    multivariate probit model
  */
  ivec y2 = y;
  int n = y.n_elem;
  for(int i = 0; i < n; i ++){
    if(y(i) == 0){
      y2(i) = -1;
    }
  }
  if(any(sign(beta) != y2)){
    return 0.0;
  }else{
    return 1.0;
  }
  // vec output;
  // output = beta % y;
  // if(output.min() > 0){
  //   return 1.0;
  // }else{
  //   return 0.0;
  // }
}


vec ESS_draw_mvp(vec const& beta_ini, vec const& beta_hat, mat const& L, int const& p, ivec const& y){

  /*
    sample via elliptical slice sampler
    Input : beta_ini, vector of initial value
            beta_hat, mean of beta (likelihood function)
            L, Cholesky factor (lower triangular LL' = Sigma) of covariance matrix of normal part
  */

  // subtract mean from the initial value, sample the deviation from mean
  vec beta = beta_ini - beta_hat;

  // draw the auxillary vector
  vec eps = arma::randn<vec>(L.n_cols);
  vec nu = L * eps;

  // compute the prior threshold
  double u = as_scalar(randu<vec>(1));

  double priorcomp = prior_function_mvp(beta + beta_hat, p, y);

  // double ly = priorcomp + log(u); // WRONG!

  double ly = priorcomp * u;

  // elliptical slice sampling
  double thetaprop = as_scalar(randu<vec>(1)) * 2.0 * M_PI;
  vec betaprop = beta * cos(thetaprop) + nu * sin(thetaprop);
  double thetamin = thetaprop - 2.0 * M_PI;
  double thetamax = thetaprop;


  while(prior_function_mvp(betaprop + beta_hat, p, y) < ly){
    if(thetaprop < 0){
      thetamin = thetaprop;
    }else{
      thetamax = thetaprop;
    }

    // runif(thetamin, thetamax)
    thetaprop = as_scalar(randu<vec>(1)) * (thetamax - thetamin) + thetamin;

    betaprop = beta * cos(thetaprop) + nu * sin(thetaprop);
  } 

  // accept the proposal
  beta = betaprop;

  // add the mean back
  vec output = beta + beta_hat;

  
  return output;
}




vec draww_ESS_mvp(vec const& w, vec const& mu, mat const& L, ivec const& y){
  // function draw w vector for all n obs
  // using elliptical slice sampler
  // y is a lenght-n vector contains choices, ranges from 1 to p ! not 0 to p-1!
  int p = L.n_cols; // Caution! p is length of w_i vector == number of alternatives - 1
  int n = w.size() / p;

  int ind = 0;
  vec outw = zeros<vec>(w.size());
  // draw wi
  for(int i = 0; i < n; i++){
    ind = p * i;
    outw.subvec(ind, ind + p - 1) = ESS_draw_mvp(w.subvec(ind, ind + p - 1), mu.subvec(ind, ind + p - 1), L, p, y.subvec(ind, ind + p - 1));
  }
  return (outw);
}



//EXTRA FUNCTIONS SPECIFIC TO THE MAIN FUNCTION--------------------------------------------
vec drawwi_mvp2(vec const& w, vec const& mu, mat const& sigmai, int p, ivec y){
  
//Wayne Taylor 9/8/2014
  
//function to draw w_i by Gibbing thru p vector

  int above;
  vec outwi = w;

	for(int i = 0; i<p; i++){	
    if (y[i]){
			above = 0;
	  } else { 
			above = 1;
	  }
  
  vec CMout = condmom(outwi,mu,sigmai,p,i+1);
  // outwi[i] = rtrun1(CMout[0],CMout[1],0.0,above);
  outwi[i] = trunNorm(CMout[0],CMout[1],0.0,above);
  }

  return (outwi);
}

vec draww_mvp2(vec const& w, vec const& mu, mat const& sigmai, ivec const& y){
  
// Wayne Taylor 9/8/2014
  
//function to gibbs down entire w vector for all n obs

  int p = sigmai.n_cols;
  int n = w.size()/p;
  int ind; 
  vec outw = zeros<vec>(w.size());
  
  for(int i = 0; i<n; i++){
    ind = p*i;
		outw.subvec(ind,ind+p-1) = drawwi_mvp2(w.subvec(ind,ind+p-1),mu.subvec(ind,ind+p-1),sigmai,p,y.subvec(ind,ind+p-1));
	}

  return (outw);
}

//MAIN FUNCTION---------------------------------------------------------------------------------------
//[[Rcpp::export]]
List rmvpGibbs_slice_rcpp_loop(int R, int keep, int nprint, int p, 
                         ivec const& y, mat const& X, vec const& beta0, mat const& sigma0, 
                         mat const& V, double nu, vec const& betabar, mat const& A) {
                           
// Wayne Taylor 9/24/2014

  int n = y.size()/p;
  int k = X.n_cols;
  
  //allocate space for draws
  mat sigmadraw = zeros<mat>(R/keep, p*p);
  mat betadraw = zeros<mat>(R/keep,k);
  vec wnew = zeros<vec>(X.n_rows);
  
  //set initial values of w,beta, sigma (or root of inv)
  vec wold = wnew;
  vec betaold = beta0;
  mat C = chol(solve(trimatu(sigma0),eye(sigma0.n_cols,sigma0.n_cols))); //C is upper triangular root of sigma^-1 (G) = C'C
                                                                         //trimatu interprets the matrix as upper triangular and makes solve more efficient
  
  mat sigmai, zmat, epsilon, S, IW, ucholinv, VSinv; 
  vec betanew;
  List W;
  
  // start main iteration loop
  int mkeep = 0;

  mat L = C;
  
  if(nprint>0) startMcmcTimer();
  
    for(int rep = 0; rep<R; rep++) {
    
      //draw w given beta(rep-1),sigma(rep-1)
      sigmai = trans(C)*C;
  
      //draw latent vector
      
      //w is n x (p-1) vector
      //   X ix n(p-1) x k  matrix
      //   y is n x (p-1) vector of binary (0,1) outcomes 
      //   beta is k x 1 vector
      //   sigmai is (p-1) x (p-1) 
          
      if(0){
        sigmai = trans(C)*C;
        wnew = draww_mvp2(wold,X*betaold,sigmai,y);
      }else{
        wnew = draww_ESS_mvp(wold,X*betaold, L, y);
      }  

      //draw beta given w(rep) and sigma(rep-1)
      //  note:  if Sigma^-1 (G) = C'C then Var(Ce)=CSigmaC' = I
      //  first, transform w_i = X_ibeta + e_i by premultiply by C
      
      zmat = join_rows(wnew,X); //similar to cbind(wnew,X)
      zmat.reshape(p,n*(k+1));
      zmat = C*zmat;
      zmat.reshape(X.n_rows,k+1);
      
      betanew = breg(zmat(span::all,0),zmat(span::all,span(1,k)),betabar,A);
      
      //draw sigmai given w and beta
      epsilon = wnew-X*betanew;
      epsilon.reshape(p,n);  
      S = epsilon*trans(epsilon);
      
      //same as chol2inv(chol(V+S))
      ucholinv = solve(trimatu(chol(V+S)), eye(p,p));
      VSinv = ucholinv*trans(ucholinv);
      
      W = rwishart(nu+n,VSinv);
      C = as<mat>(W["C"]); //conversion from Rcpp to Armadillo requires explict declaration of variable type using as<>
      L = as<mat>(W["CI"]);
      //print time to completion
      if (nprint>0) if ((rep+1)%nprint==0) infoMcmcTimer(rep, R);
      
      //save every keepth draw
        if((rep+1)%keep==0){
          mkeep = (rep+1)/keep;
          betadraw(mkeep-1,span::all) = trans(betanew);
          IW  = as<mat>(W["IW"]);
          sigmadraw(mkeep-1,span::all) = trans(vectorise(IW));
         }
        
      wold = wnew;
      betaold = betanew;
    }
  
  if(nprint>0) endMcmcTimer();
      
  return List::create(
    Named("betadraw") = betadraw, 
    Named("sigmadraw") = sigmadraw);
}
