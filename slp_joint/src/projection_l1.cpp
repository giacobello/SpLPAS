/*  
  Copyright 2015 Tobias L. Jensen <tlj@es.aau.dk>
  Department of Electronic Systems, Aalborg University, Denmark

  This file is part of slp_joint.
    
  slp_joint is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  
  slp_joint is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License
  along with slp_joint.  If not, see <http://www.gnu.org/licenses/>.
*/


#include "projection_l1.hpp"


bool descending (FTYPE i, FTYPE j) { return (i>j); }

void div_range(int size, FTYPE * x){
  int i;

   for( i = 1 ; i < size ; ++i){
     x[i] /= (i+1);
  }  
}

/*

 Returns the solution to the problem
 
  minimize    ||x - b||_2
  subject to  ||x||_1 \leq tau

 Useful for calculating the projection 

 xp = P_Q(b)

 where Q = {x | ||x||_1 \leq tau}

 This version implements a complete sort for
 solving this optimization problem making it an 
 O(n log n) algorithm.

 Tobias L. Jensen, tlj@es.aau.dk, 2015
 Aalborg University

*/

void projection_l1(int n, FTYPE *b, FTYPE tau, FTYPE *x){

  FTYPE lambda;
  FTYPE * cs;
  FTYPE nrmb = 0;

  nrmb = CvmdAsum(n, b);
  if( tau >= nrmb ){
    CvmdCopy(n, b, x); return;
  }

  cs = vector(n);

  /* cs_i = |b_i| for all i */
  CvmdAbs(n, b, cs);
  
  /* Sort in descending order of magnitude */
  std::sort(cs, cs+n, descending);

  /* lambda = max((cs-tau)./(1:n)'); */
  CvmdCumsum(n, cs);
  CvmdAddConstant(n, -tau, cs, cs);
  div_range(n, cs);
  lambda = *std::max_element(cs, cs+n);
  
  S(n, b, lambda, x);

  del_vector(cs);
}




/* Softhreshold of the vector x, length n using parameter t
   output is in y */
void S(int n, FTYPE *x, FTYPE t, FTYPE * y){

  for(int k = 0 ; k < n ; ++k ){
    if(x[k] > t)
      y[k] = x[k] - t;
    else if(x[k] < -t)
      y[k] = x[k] + t;
    else
      y[k] = 0.0;
  }
}

