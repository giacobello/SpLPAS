function [x] = projection_l1(b, tau)
%
% Returns the solution to the problem
% 
%  minimize    ||x - b||_2
%  subject to  ||x||_1 \leq tau
%
% Useful for calculating the projection 
%
% xp = P_Q(b)
%
% where Q = {x | ||x||_1 \leq tau}
%
% This version implements a complete sort for
% solving this optimization problem making it an 
% O(n log n) algorithm.
%
% Tobias L. Jensen, tlj@es.aau.dk, 2015
% Aalborg University
%

if (tau >= norm(b, 1))
    x = b; 
    return; 
end

n = length(b);

% Just do a complete sort. Other algorithms may be more efficient 
s = sort(abs(b), 'descend');
cs = cumsum(s);
    
lambda = max((cs-tau)./(1:n)');

%softhreshold with parameter t
x = sign(b).*max(abs(b)-lambda, 0);
