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

#include "filter.hpp"


/* Adjoint FIR filter with all-zero initialization
   If filter is y <- Xa
   then the adjoint filter implements a <- X^T y

  x has length k
  y has lenght k
  a has length l 
*/
void adjointfilter(FTYPE * x, FTYPE * y, int k, FTYPE *a, int l){
  
  int i, j;

  for( i = 0 ; i < l ; i++ ){
    a[i] = 0.0;
    
    for( j = i ; j < k ; j++ ){
      a[i] += y[j]*x[j-i];
    }
  }
}


/* FIR filter with all-zero initialization
  Identical to Matlab y = filter(a, 1, x) 
  a has length n
  x has length k
  y has lenght k
*/
void filter(FTYPE * a, int n, FTYPE * x, int k, FTYPE *y){
  int i, j;
  int m;

  for( i = 0 ; i < k ; i++ ){
    m = MIN(i, n-1);

    y[i] = 0.0;
    for( j = 0 ; j <= m ; j++ ){
      y[i] += a[j]*x[i-j];
    }
    
  }
}

/* watch out. Signal needs to be zero padded for this to work */
void fftfilterfunc(FTYPE * a, int n, FTYPE * x, int k, FTYPE *y){
  
  FFTW_PLAN p1, p2, p3;
  FFTW_COMPLEX *H, *X, *Y;
  FTYPE *yp, *ap, *xp;
  int i, N, Np;
  FTYPE s;

  N = 2;
  while(true){
    if ( N >= k + n)
      break;
    N = N*2;
  }

  Np = N/2 + 1;
  s = 1.0/N;

  H = (FFTW_COMPLEX*) FFTW_MALLOC(sizeof(FFTW_COMPLEX) * Np);
  X = (FFTW_COMPLEX*) FFTW_MALLOC(sizeof(FFTW_COMPLEX) * Np);
  Y = (FFTW_COMPLEX*) FFTW_MALLOC(sizeof(FFTW_COMPLEX) * Np);
  yp = (FTYPE*) FFTW_MALLOC(sizeof(FTYPE) * N);
  ap = (FTYPE*) FFTW_MALLOC(sizeof(FTYPE) * N);
  xp = (FTYPE*) FFTW_MALLOC(sizeof(FTYPE) * N);

  CvmdCopy(n+1, a, ap);
  for( i = n+1 ; i < N ; i++ )
    ap[i] = 0.0;

  CvmdCopy(k, x, xp);
  for( i = k ; i < N ; i++ )
    xp[i] = 0.0;

  p1 = FFTW_PLAN_DFT_R2C_1D(N, ap, H, FFTW_MEASURE);
  p2 = FFTW_PLAN_DFT_R2C_1D(N, xp, X, FFTW_MEASURE);
  p3 = FFTW_PLAN_DFT_C2R_1D(N, Y, yp, FFTW_MEASURE);

  FFTW_EXECUTE(p1);
  FFTW_EXECUTE(p2);

  for( i = 0 ; i < Np ; i++ )
    Y[i] = H[i]*X[i];

  FFTW_EXECUTE(p3);

  //for( i = 0 ; i < Np ; i++)
  //  printf("%d: %5.2f +j%5.2f\n", i, creal(H[i]), cimag(H[i]));

  for( i = 0 ; i < k ; i++)
    y[i] = s*yp[i];

  FFTW_DESTROY_PLAN(p1);
  FFTW_DESTROY_PLAN(p2);
  FFTW_DESTROY_PLAN(p3);
  FFTW_FREE(H); 
  FFTW_FREE(X); 
  FFTW_FREE(Y); 
  FFTW_FREE(yp); 
  FFTW_FREE(ap); 
  FFTW_FREE(xp);  
}


/* watch out. Signal needs to be zero padded for this to work */
fftfilter::fftfilter(int nr, FTYPE * x, int kr){

  n = nr;
  k = kr;

  N = 2;
  while(true){
    if ( N >= k + n )
      break;
    N = N*2;
  }

  Np = N/2 + 1;
  s = 1.0/N;

  H = (FFTW_COMPLEX*) FFTW_MALLOC(sizeof(FFTW_COMPLEX) * Np);
  X = (FFTW_COMPLEX*) FFTW_MALLOC(sizeof(FFTW_COMPLEX) * Np);
  Y = (FFTW_COMPLEX*) FFTW_MALLOC(sizeof(FFTW_COMPLEX) * Np);
  yp = (FTYPE*) FFTW_MALLOC(sizeof(FTYPE) * N);
  ap = (FTYPE*) FFTW_MALLOC(sizeof(FTYPE) * N);
  xp = (FTYPE*) FFTW_MALLOC(sizeof(FTYPE) * N);

  p1 = FFTW_PLAN_DFT_R2C_1D(N, ap, H, FFTW_MEASURE);
  p2 = FFTW_PLAN_DFT_R2C_1D(N, xp, X, FFTW_MEASURE);
  p3 = FFTW_PLAN_DFT_C2R_1D(N, Y, yp, FFTW_MEASURE);

  set_signal(x);
}


fftfilter::~fftfilter(){
  FFTW_DESTROY_PLAN(p1);
  FFTW_DESTROY_PLAN(p2);
  FFTW_DESTROY_PLAN(p3);
  FFTW_FREE(H); 
  FFTW_FREE(X); 
  FFTW_FREE(Y); 
  FFTW_FREE(yp); 
  FFTW_FREE(ap); 
  FFTW_FREE(xp);  
}

void fftfilter::set_signal(FTYPE * x){

  CvmdCopy(k, x, xp);
  for( i = k ; i < N ; i++ )
    xp[i] = 0.0;

  FFTW_EXECUTE(p2);
}

void fftfilter::filter(FTYPE * a, FTYPE *y){

  CvmdCopy(n, a, ap);
  for( i = n ; i < N ; i++ )
    ap[i] = 0.0;

  FFTW_EXECUTE(p1);

  CvmdMulz(Np, (FTYPE*)H, (FTYPE*)X, (FTYPE*)Y);

  FFTW_EXECUTE(p3);


  for( i = 0 ; i < k ; i++)
    y[i] = s*yp[i];

}


fftadjointfilter::fftadjointfilter(int lr, FTYPE * x, int kr){

  l = lr;
  k = kr;

  N = 2;
  while(1){
    if ( N >= k + l )
      break;
    N = N*2;
  }

  Np = N/2 + 1;
  s = 1.0/N;

  H = (FFTW_COMPLEX*) FFTW_MALLOC(sizeof(FFTW_COMPLEX) * Np);
  X = (FFTW_COMPLEX*) FFTW_MALLOC(sizeof(FFTW_COMPLEX) * Np);
  Y = (FFTW_COMPLEX*) FFTW_MALLOC(sizeof(FFTW_COMPLEX) * Np);
  yp = (FTYPE*) FFTW_MALLOC(sizeof(FTYPE) * N);
  ap = (FTYPE*) FFTW_MALLOC(sizeof(FTYPE) * N);
  xp = (FTYPE*) FFTW_MALLOC(sizeof(FTYPE) * N);

  p1 = FFTW_PLAN_DFT_C2R_1D(N, H, ap, FFTW_MEASURE);
  p2 = FFTW_PLAN_DFT_R2C_1D(N, xp, X, FFTW_MEASURE);
  p3 = FFTW_PLAN_DFT_R2C_1D(N, yp, Y, FFTW_MEASURE);

  set_signal(x);
}


fftadjointfilter::~fftadjointfilter(){
  FFTW_DESTROY_PLAN(p1);
  FFTW_DESTROY_PLAN(p2);
  FFTW_DESTROY_PLAN(p3);
  FFTW_FREE(H); 
  FFTW_FREE(X); 
  FFTW_FREE(Y); 
  FFTW_FREE(yp); 
  FFTW_FREE(ap); 
  FFTW_FREE(xp);  
}

void fftadjointfilter::set_signal(FTYPE * x){
  for( i = 0 ; i < k ; i++ )
    xp[i] = x[k-1-i]; //flipped coefficients
  for( i = k ; i < N ; i++ )
    xp[i] = 0.0;

  FFTW_EXECUTE(p2);
}

void fftadjointfilter::filter(FTYPE *y, FTYPE * a){

  CvmdCopy(k, y, yp);
  for( i = k ; i < N ; i++ )
    yp[i] = 0.0;

  FFTW_EXECUTE(p3);

  CvmdMulz(Np, (FTYPE*)X, (FTYPE*)Y, (FTYPE*)H);

  //for( i = 0 ; i < N ; i++ )
  //  printf("%d: %5.2f +j%5.2f\n", i, creal(H[i]), cimag(H[i]));

  FFTW_EXECUTE(p1);

  for( i = k - 1; i < k + l - 1; i++ )
    a[i-k+1] = s*ap[i];

}
