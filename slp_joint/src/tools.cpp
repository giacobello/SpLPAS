/* 

  Various elementwise tools.

  Tobias L Jensen, 
  March 2015
*/


#include "tools.hpp"

// Copy y <= x
void CvmdCopy(int n, FTYPE *x, FTYPE *y){

  #ifdef MKL
  COPY(n, x, 1, y, 1);
  #else
  for(int i = 0 ; i < n ; ++i)
    y[i] = x[i];
  #endif

}

// Scal x <= alpha*x
void CvmdScal(int n, FTYPE alpha, FTYPE *x){
  
  #ifdef MKL
  SCAL(n, alpha, x, 1);
  #else
  for(int i = 0 ; i < n ; ++i)
    x[i] = alpha*x[i];
  #endif
}

// returns  y <= alpha x + y
void CvmdAxpy(int n, FTYPE alpha, FTYPE *x, FTYPE *y){

  #ifdef MKL
  AXPY(n, alpha, x, 1, y, 1);
  #else
  for(int i = 0 ; i < n ; ++i)
    y[i] += alpha*x[i];
  #endif
}

// returns inner dot product a^T b
FTYPE CvmdDot(int n, FTYPE *a, FTYPE *b){

  #ifdef MKL
  return DOT(n, a, 1, b, 1);
  #else
  FTYPE c = 0.0;
  for(int i = 0 ; i < n ; ++i)
    c += a[i]*b[i];
  return c;
  #endif


}

// Adds c_i <- a + b_i
void CvmdAddConstant(int n, FTYPE a, FTYPE *b, FTYPE *c){

  for(int i = 0 ; i < n ; ++i)
    c[i] = a + b[i];
}

// Adds c <- a + b
void CvmdAdd(int n, FTYPE *a, FTYPE *b, FTYPE *c){

  #ifdef MKL
  VADD(n, a, b, c);
  #else
  for(int i = 0 ; i < n ; ++i)
    c[i] = a[i] + b[i];
  #endif
}

FTYPE CvmdAsum(int n, FTYPE *x){
  
  #ifdef MKL
  return ASUM (n, x, 1);
  #else
  FTYPE sum = 0;
  for(int i = 0 ; i < n ; ++i)
    sum += ABS(x[i]);
  return sum;
  #endif
}

// Adds c <- a - b
void CvmdSub(int n, FTYPE *a, FTYPE *b, FTYPE *c){

  #ifdef MKL
  VSUB(n, a, b, c);
  #else  
  for(int i = 0 ; i < n ; ++i)
    c[i] = a[i] - b[i];
  #endif
}

// Adds a <- a + b
void CvmdAddInplace(int n, FTYPE *a, FTYPE *b){

  #ifdef MKL
  VADD(n, a, b, a);
  #else  
  for(int i = 0 ; i < n ; ++i)
    a[i] += b[i];
  #endif
}

// Adds a <- a - b
void CvmdSubInplace(int n, FTYPE *a, FTYPE *b){

  #ifdef MKL
  VSUB(n, a, b, a);
  #else    
  for(int i = 0 ; i < n ; ++i)
    a[i] -= b[i];
  #endif
}

//forms reverse vector b <- a_[n:1:-1]
void CvmdReverse(int n, FTYPE *a, FTYPE *b){ 

  for(int i=0 ; i < n ; ++i)
    b[i] = a[n-i-1];

}

//forms symmetric vector b <- [a_[1:n]; a_[n-1:1:-1]]
void CvmdSymmetric(int n, FTYPE *a, FTYPE *b){ 

  for(int i=0 ; i < n ; ++i)
    b[i] = a[i];

  for(int i=0 ; i < n-1 ; ++i)
    b[n+i] = a[n-i-2];
}

//computer c <- a_[1:n] + [0; b_[1:n-1]]
void CvmdShift(int n, FTYPE *a, FTYPE *b, FTYPE *c){ 
  c[0] = a[0];
  for(int i=1 ; i < n; ++i)
    c[i] = a[i] + b[i-1];
}

//Computes elementwise y_i <- a - x_i^2
void CvmdScalSquare(int n, FTYPE a, FTYPE * x, FTYPE * y){
  
  for(int i = 0 ; i < n ; ++i){
    y[i] = a - x[i]*x[i];
  }
}



//Initialize a vector with value a
void CvmdInit(int n, FTYPE a, FTYPE * x){
  
  for(int i = 0 ; i < n ; ++i){
    x[i] = a;
  }
}


//Computes elementwise multiplication y_i <- x_i * z_i
void CvmdMul(int n, FTYPE * x, FTYPE * z, FTYPE * y){
  
  #ifdef MKL
  VMUL(n, x, z, y);
  #else
  for(int i = 0 ; i < n ; ++i){
    y[i] = x[i]*z[i];
  }
  #endif
}



//Computes elementwise multiplication y <- x_i * z_i 
// with x, z, y complex
void CvmdMulz(int n, FTYPE * x, FTYPE * z, FTYPE *y){

  #ifdef MKL
  VMULZ(n, (MKL_COMPLEX*)x, (MKL_COMPLEX*)z, (MKL_COMPLEX*)y);
  #else
  for(int i = 0 ; i < n ; ++i){
    y[2*i] = x[2*i]*z[2*i] - x[2*i+1]*z[2*i+1];
    y[2*i+1] = x[2*i]*z[2*i+1] + x[2*i+1]*z[2*i];
  }
  #endif
}



//Computes elementwise division y_i <- x_i / z_i
void CvmdDiv(int n, FTYPE * x, FTYPE * z, FTYPE * y){
  
  #ifdef MKL
  VDIV(n, x, z, y);
  #else
  for(int i = 0 ; i < n ; ++i){
    y[i] = x[i]/z[i];
  }
  #endif
}

//Computes elementwise inverse y_i <- 1.0 / x_i
void CvmdInverse(int n, FTYPE * x, FTYPE * y){
  
  #ifdef MKL
  VINV(n, x, y);
  #else
  for(int i = 0 ; i < n ; ++i){
    y[i] = 1/x[i];
  }
  #endif
}

// packs x from step s to unit stride in y
void CvmdPack(int n, FTYPE * x, int s, FTYPE * y){
  int ii = 0;
  for( int i = 0 ; i < n ; ++i){
    y[i] = x[ii];
    ii += s;
  }
}

/* returns ||x||_2 of a vector */
FTYPE CvmdNrm2(int n, FTYPE *x){

    
  #ifdef MKL
  return NRM2 (n, x, 1);
  #else
  FTYPE nrm=0;
  for ( int i = 0 ; i < n ; ++i)
    nrm += x[i]*x[i];
  return sqrt(nrm);
  #endif
}

/* Returns cumulative sum inplace */
void CvmdCumsum(int n, FTYPE * x){
  
  for ( int i = 1 ; i < n ; ++i)
    x[i] += x[i-1];

}

/* Returns elementwise abs  y_i = |x_i| */
void CvmdAbs(int n, FTYPE * x, FTYPE * y){
  
  for ( int i = 0 ; i < n ; ++i)
    y[i] = ABS(x[i]);
}
  
/* Returns elementwize squared abs of a complex vector */
void  CvmdAbs2z(int n, FTYPE * x, FTYPE *y){
  int i;

  for(i = 0 ; i < n ; ++i )
    y[i] = x[2*i]*x[2*i] + x[2*i+1]*x[2*i+1];

}


int CvmdArgmax(int n, FTYPE * x){

  FTYPE max;
  int arg;

  max = x[0];
  arg = 0;

  for( int k = 1 ; k < n ; ++k){
    if( x[k] > max){
      max = x[k];
      arg =  k;
    }
  }

  return arg;
}
