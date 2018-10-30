% Simple example of how to use the slp_joint code

addpath ../mlib

delta = 2;
gamma = 0.2;
N = 20;

% Construct a synthetic residual from last frame
rr_mem = randn(N, 1);
I = find(abs(rr_mem) < 0.5);
rr_mem(I) = 0;
rr_mem = rr_mem/(20*norm(rr_mem, 1));


xt = 0.01*randn(2*N, 1);
Xt = toeplitz(xt, [xt(1) zeros(1, N-1)]);
xq = Xt(:, 1);
Xq = Xt(:, 2:end);

cvx_begin
  cvx_quiet(true)
  variable aa(N-1)
  variable rropt(2*N)
  minimize norm( Xq * aa + xq - rropt, 2) 
  subject to
  rropt(1:N) == rr_mem;
  norm(aa, 1) <= delta;
  norm(rropt, 1) <= gamma;
cvx_end

[ak, rk] = slp_joint(xt, rr_mem, delta, gamma-norm(rr_mem, 1), 1e-20, 10000);