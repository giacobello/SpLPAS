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


#include "slp_joint.hpp"

/* To use this, subclass class spec, implement the neccessary setup
   and call the function. Initialized using z 

   Special version for problems on the form 

   minimize f(x)
   subject x in Q

*/
void ofm(spec_ofm * s, FTYPE * xk){
  int k;
  FTYPE alphak = 1-1e-6, alphakp1, beta;
  FTYPE rel_res = 0, nrm_res;


  //unpacking - makes it more readable
  FTYPE * t1 = s->t1;
  int kmax = s->kmax;
  int n = s->n;
  FTYPE eps_rel = s->eps_rel;
  int verbose = s->verbose;
  FTYPE L = s->L;
  FTYPE * yk = s->yk;
  FTYPE * xkm1 = s->xkm1;

  CvmdCopy(n, xk, yk);
  CvmdCopy(n, xk, xkm1);

  if( verbose ){
    printf("---------- OFM -------------- \n");
    printf("n = %d\n", n);
  }

  for( k = 0 ; k < kmax ; ++k ){
    //xk = PQ(yk - (1/L)*g(yk));
    s->g(yk, t1);
    CvmdScal(n, -(1/L), t1);
    CvmdAxpy(n, 1.0, yk, t1);
    s->PQ(t1); 
    CvmdCopy(n, t1, xk);

    //Calculate optimality residual
    CvmdAxpy(n, -1.0, yk, t1);
    nrm_res = CvmdNrm2(n, t1);
    rel_res = (L/2)*nrm_res*nrm_res/n;


    if(verbose)
      printf("k=%5d, rel_res=%.4e f(x_k) = %1.4e\n", k, rel_res, s->f(xk));

    if( rel_res <= eps_rel )
      break;
 
    //Momentum term
    alphakp1 = (-(alphak*alphak) + 
                sqrt((pow(alphak, 4) + 4*alphak*alphak)))*0.5;
    beta = alphak*(1-alphak)/(alphak*alphak+alphakp1);

    //  yk = xk + beta*(xk-xkm1);
    CvmdCopy(n, xk, yk);
    CvmdAxpy(n, -1.0, xk, xkm1);
    CvmdAxpy(n, -beta, xkm1, yk);

    //xkm1 = xk;
    //alphak = alphakp1;
    CvmdCopy(n, xk, xkm1);
    alphak= alphakp1;

  }
  
  s->k = k;

}



slp_joint_iir::slp_joint_iir(int Nr, FTYPE * xr, 
                             int kmaxr, FTYPE eps_relr, 
                             FTYPE deltar, FTYPE gammar, 
                             bool fftfilteringr){
  
  /* Set the intern variables */
  N = Nr;

  delta = deltar;
  gamma = gammar;

  kmax = kmaxr;
  eps_rel = eps_relr;
  verbose = false;
  fftfiltering = fftfilteringr;

  n = 2*N - 1;

  M = 2*N - 2;
  Mp = M/2+1;

  /* allocate */
  yf = vector(2*N);
  t1 = vector(n);
  yk = vector(n);
  xkm1 = vector(n);


  xp = (FTYPE*) FFTW_MALLOC(sizeof(FTYPE) * M);
  X = (FFTW_COMPLEX*) FFTW_MALLOC(sizeof(FFTW_COMPLEX) * Mp);
  p1 = FFTW_PLAN_DFT_R2C_1D(M, xp, X, FFTW_MEASURE);

  set_signal(xr);

  if( fftfiltering ){
    h = new fftfilter(N-1, x, 2*N);
    ha = new fftadjointfilter(N-1, x, 2*N);
  }
}

void slp_joint_iir::update_signal(FTYPE * xr){

  set_signal(xr);

  if( fftfiltering ){//Update the coefficient in the fftfilter
    h->set_signal(xr);
    ha->set_signal(xr);
  }
}

void slp_joint_iir::set_signal(FTYPE * xr){

  x = xr;
  xq = x + N;

  /* First and last element of the x vector is not a part of Xq 
     so it is sufficient to consider M = 2*N-2 elements 
     L = max(abs(fft(x(2:end-1))))^2 + 1;
  */

  CvmdCopy(M, x+1, xp);
  FFTW_EXECUTE(p1);
  CvmdAbs2z(Mp, (FTYPE*) X, t1);
  L = t1[CvmdArgmax(Mp, t1)] + 1;
  
}

void slp_joint_iir::set_signal(FTYPE * xr, FTYPE deltar, FTYPE gammar){

  delta = deltar;
  gamma = gammar;

  set_signal(xr);
}


slp_joint_iir::~slp_joint_iir(){
  
  del_vector(yf);
  del_vector(t1);
  del_vector(yk);
  del_vector(xkm1);

  FFTW_FREE(xp);
  FFTW_FREE(X);
  FFTW_DESTROY_PLAN(p1);

  if( fftfiltering ){
    delete h;
    delete ha;
  }

}


// Calculates 0.5*||xq + Xq * a - r||_2^2
FTYPE slp_joint_iir::f(FTYPE * z){ 
    //z = [a;r]
    FTYPE * a = z;
    FTYPE * r = z + N - 1;

    if( fftfiltering )
      h->filter(a, yf);
    else
      filter(a, N-1, x, 2*N, yf);
    
    CvmdAxpy(N, 1.0, xq, yf+N-1);
    CvmdAxpy(N, -1.0, r, yf+N-1);
    FTYPE nrm = CvmdNrm2(N, yf+N-1);

    return 0.5*nrm*nrm;
}


// Calculates the gradient
void slp_joint_iir::g(FTYPE * z, FTYPE * y){
  //z = [a;r]
  FTYPE * a = z;
  FTYPE * r = z + N - 1;

  if( fftfiltering )
    h->filter(a, yf);
  else
    filter(a, N-1, x, 2*N, yf);

  CvmdAxpy(N, 1.0, xq, yf+N-1);
  CvmdAxpy(N, -1.0, r, yf+N-1);

  CvmdCopy(N, yf+N-1, y+N-1);
  CvmdScal(N, -1, y+N-1);

  CvmdInit(N-1, 0.0, yf); yf[2*N-1] = 0.0;

  if( fftfiltering )
    ha->filter(yf, y);
  else
    adjointfilter(x, yf, 2*N, y, N-1);
}


/*
  Projection onto the feasible set

   {[a, r] | ||a||_1 <= \delta,  ||r||_1 <= \gamma}

*/
void slp_joint_iir::PQ(FTYPE * z){
  FTYPE * a = z;
  FTYPE * r = z + N - 1;
  
  // The projection is seperable
  projection_l1(N-1, a, delta, a);
  projection_l1(N, r, gamma, r);

}


