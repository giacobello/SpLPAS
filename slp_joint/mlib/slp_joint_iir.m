function [ak, rk, flag] = slp_joint_iir(x, N, delta, gamma, eps_rel,...
                                Nit_max, varargin)
%
% Solves the problem
%
%   minimize_{a, r} ||xq + Xq * a - r||_2^2
%   subject to ||a||_1 <= \delta
%              ||r||_1 <= \gamma
%
% where Xt = convmtx(x, 2*N); [xq, Xq] = Xt(N+1:2*N, 1:N);
%
% INPUT:
%  x       : real vector 
%  N       : size of frame
%  eps_rel : relative stopping criteria 
%             ineffective with negative numbers. Then just stop
%             based on Nit_max.
%  Nit_max : maximum number of iterations
%
%  eps_rel = relative stopping criteria 
%  Nit_max = maximum number of iterations.
%
%  Optional:
%   A structure that can be accepted as info struct by ofm.
%   Can optional also contain the fields
%    a0    : initialization of the a part of the optimization problem
%    r0    : initialization of the r part of the optimization problem
%      If not provided the default is zero initialization.
%
% OUTPUT:
%   ak     : Approximate a solution (real vector size (N-1 x 1))
%   rk     : Approximate r solution (real vector size (N x 1)) 
%   info   : A struct with additional info from the solver. 
%             See help ofm
%
% This solver relies on the function ofm.m which implements an optimal 
% first-order method for smooth, convex objectives. Use help ofm 
% to learn more
%
% Tobias L. Jensen, tlj@es.aau.dk
% Aalborg University
% 2015


Xt = convmtx(x, 2*N);
Xt = Xt(N+1:2*N, 1:N);

xq = Xt(:, 1);
Xq = Xt(:, 2:end);

% In standard form
A = [Xq -eye(N, N)];
b = - xq;

f = @(x) 0.5*norm(A*x-b, 2)^2;
g = @(x) A'*(A*x - b);
P = @(x) [projection_l1(x(1:N-1), delta); ...
          projection_l1(x(N:end), gamma)];

%Lh = svds(A, 1, 'L')^2;
Lh = max(abs(fft(x(2:end-1))))^2 + 1;

x0 = zeros(2*N-1, 1);

if length(varargin) == 1
    setup = varargin{1};
    if isfield(setup, 'a0');
        x0(1:N-1) = a0;
    end
    if isfield(setup, 'r0');
        x0(N:end) = r0;
    end
else
    setup = 0;
end


    
[xk, flag] = ofm(f, g, P, x0, Lh, eps_rel, Nit_max, ...
                 setup);

% Unpack solution
ak = xk(1:N-1);
rk = xk(N:end);