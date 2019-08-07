#include "bayesm.h"
 
// [[Rcpp::export]]
double lndLogMvn(vec const& x, vec const& mu, mat const& rooti){

//Wayne Taylor 9/7/2014

// function to evaluate log of MV LOG Normal density with  mean mu, var Sigma
// Sigma=t(root)%*%root   (root is upper tri cholesky root)
// Sigma^-1=rooti%*%t(rooti)   
// rooti is in the inverse of upper triangular chol root of sigma
//          note: this is the UL decomp of sigmai not LU!
//                Sigma=root'root   root=inv(rooti)

  vec z = vectorise(trans(rooti)*(log(x)-mu));
  
  return((-(x.size()/2.0)*log(2*M_PI) -.5*(trans(z)*z) - sum(log(x)) + sum(log(diagvec(rooti))))[0]);
}



// density of log normal is just  1/x * phi(log(x))
// -sum(log(x)) is the Jacobian term, in log