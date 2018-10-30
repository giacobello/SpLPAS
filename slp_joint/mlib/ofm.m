function [xk, info]= ofm(f, g, P, x0, L, eps_rel, Nit_max, varargin)
% 
% Solves the problem
% 
%  mininimize  f(x)
%  subject to  x \in Q
%
% INPUT:
%  f  = a function that evaluates the objective with call f(x)
%  g  = a function that evaluates the gradient with call g(x)
%  P  = projection of x onto the set Q that can be evaluate as P(x)
%  x0 = initialization
%  L = Lipshitz constant of the gradient function 
%         (uses fixed step size)
%  eps_rel = relative stopping criteria 
%  Nit_max = maximum number of iterations.
%
%  Optional argument: A struct with zero or more of the fields
%       verbose = true/false (default false)
%       q       = inverse condition number (default 0)
%       record_objective = record the objective for all iterations.
%                          will be returned in struct info as info.f
%                          (default false);
%       record_rel_res = record the relative residual.
%                          will be returned in struct info as info.r
%                          (default false);
%
% OUTPUT:
%  xk = approximate solution after k iterations.
%  info = a structure with fields
%      k = number of iterations
%      f = objective if record_objective was True
%      r = residual if record_rel_res was True
%
% The function uses an optimal first-order method for smooth, convex objectives.
%
% Tobias L. Jensen, tlj@es.aau.dk
% Aalborg University
% 2015
%

xk = x0;
yk = xk;
xkm1 = xk;

res = inf;
alphak = 1-1e-6;

q = 0;
record_objective = false;
record_rel_res = false;
verbose = false;

if length(varargin) == 1
    setup = varargin{1};
    if isfield(setup, 'q')
        q = setup.k;
    end
    if isfield(setup, 'f')
        record_objective = true;
        ff = zeros(Nit_max, 1);
    end
    if isfield(setup, 'r')
        record_rel_res = true;
        rr = zeros(Nit_max, 1);
    end
    if isfield(setup, 'verbose')
        verbose = setup.verbose;
    end
    
elseif length(varargin) > 1
    error('ofm.m: Only accept length one varargin')
end


% Iterate until upper limit on number of iterations is reached, if not
% aborted before
for k = 1:Nit_max
	
  xk = P(yk - (1/L)*g(yk));
  
  if record_objective
      ff(k) = f(xk);
  end
  
  %Calculate optimality residual
  rel_res = (L/2)*norm(yk-xk, 'fro')^2/numel(xk);

  if record_rel_res
      rr(k) = rel_res;
  end

  % Control output
  if verbose
    fxk = f(xk);
    s = sprintf('k=%5d, rel_res = %1.4e, f(x) = %1.4e', k, rel_res, ...
                fxk);
    disp(s)
  end

  if rel_res <= eps_rel
    break;
  end
    
  %Momentum term
  %beta = ((k-1)/(k+2));
  alphakp1 = (-(-q + alphak^2) ...
              + sqrt((-q + alphak^2)^2 + 4*alphak^2))*0.5;

  beta = alphak*(1-alphak)/(alphak^2+alphakp1);
  yk = xk + beta*(xk-xkm1);

  xkm1 = xk;
  alphak = alphakp1;
end

if verbose
	fxk = f(xk);
	s = sprintf('k=%5d, rel_res = %1.4e, f(x) = %1.4e',k,rel_res,fxk);
	disp(s)
end

info.k = k;

if record_objective
    info.f = ff(1:k);
end
if record_rel_res
    info.r = rr(1:k);
end
