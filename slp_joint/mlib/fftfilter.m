function r = fftfilter(h, x, varargin)
%
% Implements fft filtering such that
%
%
% varargin = if existing, should contain fft(x, 2*n)
%            Implemented for reuse.
%
% Tobias L. Jensen
% tlj@es.aau.dk
% Aalborg University
% 2015
%

n = max(length(x), length(h));

if length(varargin) == 1
    s = varargin{1};
else
    s = fft(x, 2*n);
end

rp = ifft(s.*fft(h, 2*n));
r = rp(1:n);
end



