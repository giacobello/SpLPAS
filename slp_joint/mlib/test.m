function tests = test()
%
% Unittest. Can be executed as runtests('test')
%
    rng(1234);    
    tests = functiontests(localfunctions);
end    



function test_projection_l1(testCase)

n = 20;
b = randn(20, 1);
tau = 3.2;

x = projection_l1(b, tau);

cvx_begin
cvx_quiet(true)
cvx_precision best
variable xc(n)
minimize norm(xc - b)
subject to
norm(xc, 1) <= tau
cvx_end

testCase.assertEqual(x, xc, 'absTol', 2e-7)
testCase.assertEqual(tau, norm(x, 1), 'absTol', 1e-8)

end

function test_ofm(testCase)
m = 20;
n = 10;

A = randn(m,n);
b = randn(m,1);

f = @(x) 0.5*norm(A*x-b, 2)^2;
g = @(x) A'*(A*x-b);
s = svds(A, 1, 'L');
delta = 0.8;

P = @(x) projection_l1(x, delta);

setup.f = true;
[xs, info] = ofm(f, g, P, zeros(n,1), s^2, 1e-10, 1000, setup);

cvx_begin
cvx_quiet(true)
cvx_precision high
variable xc(n)
minimize norm(A*xc-b)
subject to
norm(xc, 1) <= delta
cvx_end

testCase.assertEqual(xs, xc, 'absTol', 5e-5)
testCase.assertEqual(f(xs), f(xc), 'absTol', 5e-5)
testCase.assertEqual(norm(xs, 1), norm(xc, 1), 'absTol', 5e-5)
testCase.assertEqual(f(xs), info.f(end))

end


function test_fftfilter(testCase)

N = 20;
M = 5;

x = randn(N, 1);
h = randn(M, 1);
b = randn(N, 1);

X = toeplitz(x, [x(1) zeros(1, M-1)]);

testCase.assertEqual(filter(h, 1, x), X*h, 'absTol', 1e-14)


testCase.assertEqual(filter(h, 1, x), fftfilter(h, x)...
                     , 'absTol', 1e-14)

testCase.assertEqual(filter(h, 1, x), fftfilter(h, x, fft(x, 2*N))...
                     , 'absTol', 1e-14)


af = flipud(filter(flipud(b), 1, x));
testCase.assertEqual(af(1:M), X'*b, 'absTol', 1e-14)

af = flipud(fftfilter(flipud(b), x));
testCase.assertEqual(af(1:M), X'*b, 'absTol', 1e-14)

af = flipud(fftfilter(flipud(b), x, fft(x, 2*N)));
testCase.assertEqual(af(1:M), X'*b, 'absTol', 1e-14)

end

function test_slp_joint(testCase)

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

f = @(aa, rropt) 0.5*norm( Xq * aa + xq - rropt, 2)^2;

setup.f = true;
[ak, rk, info] = slp_joint(xt, rr_mem, delta, gamma-norm(rr_mem, 1), ...
                        -1, 10000, setup);

testCase.assertEqual(ak, aa, 'absTol', 5e-5)
testCase.assertEqual(rk, rropt(N+1:end), 'absTol', 5e-5)
testCase.assertEqual(info.f(end), f(aa, rropt), 'absTol', 1e-10);
testCase.assertEqual(info.f(end), f(ak, [rr_mem; rk]), 'absTol', 1e-15);

end


function test_slp_joint_iir(testCase)

delta = 0.3;
gamma = 0.4; 
N = 20; 

xt = randn(2*N, 1);
Xt = convmtx(xt, 2*N);
Xt = Xt(N+1:2*N, 1:N);

xq = Xt(:, 1);
Xq = Xt(:, 2:end);


cvx_begin
  cvx_quiet(true)
  variable aa(N-1)
  variable rr(N)
  minimize norm( xq + Xq * aa - rr, 2) 
  subject to
  norm(aa, 1) <= delta;
  norm(rr, 1) <= gamma;
cvx_end

f = @(aa, rr) 0.5*norm( xq + Xq * aa - rr, 2)^2;

setup.f = true;

[ak, rk, info] = slp_joint_iir(xt, N, delta, gamma, ...
                        -1, 10000, setup);

testCase.assertEqual(ak, aa, 'absTol', 5e-5)
testCase.assertEqual(rk, rr, 'absTol', 5e-5)
testCase.assertEqual(info.f(end), f(aa, rr), 'absTol', 1e-6);
testCase.assertEqual(info.f(end), f(ak, rk), 'absTol', 1e-15);

end
