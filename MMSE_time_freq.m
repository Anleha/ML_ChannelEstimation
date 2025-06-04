function H_MMSE = MMSE__time_freq(Nfft,H_LS,H,beta,snr_linear,h_length)
H = H.';
h = ifft(H,Nfft);
h = h(1:h_length);
H_LS = H_LS.';
% h_LS = ifft(H_LS,Nfft);
% Rgg = diag(h.*conj(h));
% WW = Rgg/(Rgg+(beta/snr_linear))*eye(h_length);
% h_est = WW*h_LS(1:h_length);
% H_MMSE = fft(h_est,Nfft);
nt =randn(Nfft*2,1) + 1i*randn(Nfft*2,1);
No = 10^(-log10(snr_linear));
noise = sqrt(No/2)*nt;
var_noise = var(noise);
Rhh = H*H' + sqrt(var_noise*eye(Nfft));

% Rhh = H*H' + 3*eye(Nfft); 
% Rhh = H*H' + 10;
% Rhh = H_LS*H_LS';
% Rhh = H*H';
W = Rhh/(Rhh+(beta/snr_linear)*eye(Nfft));
H_MMSE = W*H_LS;
H_MMSE = H_MMSE.';

