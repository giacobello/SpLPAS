%
% This exemplifies the convergence of the used method
% for joint sparse linear prediction
% Note this is not a unittest. Use instead test.m
%
% Tobias L. Jensen
% tlj@es.aau.dk
% Aalborg University 
% 2015
%

load voiced_speech_example.mat
Xq = YY;
xq = yy;
N = length(xq)/2;

delta = 2;
gamma = 0.2;

cvx_begin
  cvx_quiet(true)
  variable aa(N-1)
  variable rr(2*N)
  minimize norm(Xq * aa + xq - rr, 2) 
  subject to
  rr(1:N) == rr_mem;
  norm(aa, 1) <= delta;
  norm(rr, 1) <= gamma;
cvx_end

f = @(aa, rr) 0.5*norm( Xq * aa + xq - rr, 2)^2;
fs = f(aa, rr);

setup.f = true;
setup.r = true;

[ak, rk, info] = slp_joint(xq, xq(1:N), delta, gamma-norm(rr_mem, 1), -1, ...
                           500, setup);

figure(1)
clf
set(gca, 'defaulttextinterpreter', 'latex');
semilogy((info.f-fs)./fs)
xlabel('k')
ylabel('(f(x^{(k)})- f^*)/f^*')

figure(2)
clf
set(gca, 'defaulttextinterpreter', 'latex');
semilogy(info.r)
xlabel('k')
ylabel('(L/2)||y^{(k)}-x^{(k)}||_2^2/numel(x)');

