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

#ifndef __PROJECTION_L1_HPP__
#define __PROJECTION_L1_HPP__

#include <algorithm>

#include "tools.hpp"
#include "vector.hpp"


void projection_l1(int n, FTYPE *b, FTYPE tau, FTYPE *x);
void S(int n, FTYPE *x, FTYPE t, FTYPE *y);

#endif

