
%---------------------------------------------------------------------------
%  OFDM modulator
%  NFFT: FFT length
%  chnr: number of subcarrier
%  G: guard length
%---------------------------------------------------------------------------

function [y] = OFDM_Modulator(data,NFFT,G);

chnr = length(data);%64
N = NFFT;

x = [data,zeros(1,NFFT - chnr)]; %Zero padding

a = ifft(x); % fft

y = [a(NFFT-G+1:NFFT),a]; % insert the guard interval

	
