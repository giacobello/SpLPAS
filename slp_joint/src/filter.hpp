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

#ifndef __FILTER_HPP__
#define __FILTER_HPP__

#include <complex.h>
#include <fftw3.h>

#include "tools.hpp"



void adjointfilter(FTYPE * x, FTYPE * y, int k, FTYPE *a, int l);
void filter(FTYPE * a, int n, FTYPE * x, int k, FTYPE *y);
void fftfilterfunc(FTYPE * a, int n, FTYPE * x, int k, FTYPE *y);


class fftfilter{
  FFTW_PLAN p1, p2, p3;
  FFTW_COMPLEX *H, *X, *Y;
  FTYPE *yp, *ap, *xp;
  int i, N, Np;
  int n, k;
  FTYPE s;
  
 public:
  fftfilter(int n, FTYPE * x, int k);
  ~fftfilter();
  void filter(FTYPE * a, FTYPE * y);
  void set_signal(FTYPE * x);
};

class fftadjointfilter{
  FFTW_PLAN p1, p2, p3;
  FFTW_COMPLEX *H, *X, *Y;
  FTYPE *yp, *ap, *xp;
  int i, N, Np;
  int l, k;
  FTYPE s;

public:
  fftadjointfilter(int l, FTYPE * x, int k);
  ~fftadjointfilter();
  void filter(FTYPE * y, FTYPE * a);
  void set_signal(FTYPE * x);
};

#endif
