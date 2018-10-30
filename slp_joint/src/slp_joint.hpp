/*  
  Copyright 2014-2015 Tobias L. Jensen <tlj@es.aau.dk>
  Department of Electronic Systems, Aalborg University, Denmark

  This file is part of slp_sm.
    
  slp_sm is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  
  slp_sm is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License
  along with slp_sm.  If not, see <http://www.gnu.org/licenses/>.
*/


#ifndef __SLP_JOINT_HPP__
#define __SLP_JOINT_HPP__


#include <stdlib.h>
#include <stdio.h>
#include <math.h>


#include "tools.hpp"
#include "vector.hpp"
#include "filter.hpp"
#include "projection_l1.hpp"


/* 
  Subclass this abstract class to define what the ofm solves
 */
class spec_ofm {
 public:
  int n;
  int kmax;
  int k;
  int M;
  int Mp;
  FTYPE L;
  int verbose;
  FTYPE eps_rel;
  FTYPE * t1;
  FTYPE * yk;
  FTYPE * xkm1;
  FTYPE * xp;
  FFTW_COMPLEX * X;
  FFTW_PLAN p1;

  virtual FTYPE f(FTYPE * z) =0; //objective function 
  virtual void g(FTYPE * x, FTYPE * y) =0; //gradient function
  virtual void PQ(FTYPE * z) =0;          //projection onto the set

  void set_L(FTYPE r){L = r;};
  void set_verbose(int r){verbose = r;};
  void set_kmax(int r){kmax = r;};
  void set_eps_rel(FTYPE r){eps_rel = r;};
};

/*
   This class defines functions for solving the slp_joint_iir problem


   minimize_{a, r} 0.5*||xq + Xq * a - r||_2^2
   subject to ||a||_1 <= \delta
              ||r||_1 <= \gamma

   where Xt = convmtx(x, 2*N); [xq, Xq] = Xt(N+1:2*N, 1:N);

   using an optimal first-order method.
*/


class slp_joint_iir : public spec_ofm{

  public:
    FTYPE *yf;
    FTYPE *xq;
    FTYPE *x;
    int N;
    FTYPE delta;
    FTYPE gamma;
    bool fftfiltering;
    fftfilter * h;
    fftadjointfilter * ha;


  slp_joint_iir(int N, FTYPE * x, int kmax, FTYPE eps_rel, 
                FTYPE delta, FTYPE gamma, bool fftfiltering);

  virtual ~slp_joint_iir();

  // Calculates 0.5*||xq + Xq * a - r||_2^2
  FTYPE f(FTYPE * z);

  // Calculates the gradient
  void g(FTYPE * z, FTYPE * y);

  // Projection onto the feasible set 
  void PQ(FTYPE * z);

  void update_signal(FTYPE * xr);

private:
  void set_signal(FTYPE * xr);
  void set_signal(FTYPE * xr, FTYPE deltar, FTYPE gammar);

};

void ofm(spec_ofm * s, FTYPE * xk);

#endif 
