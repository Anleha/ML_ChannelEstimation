function H_est = MMSE_MIMO_time_freq(NFFT,H_LS,H,NofOFDMSymbol,beta,snr_linear,h_length)
% function H_est = MMSE_MIMO_time_freq(NFFT,H_LS,NofOFDMSymbol,snr_linear,h_CIR,D_f)

for i = 1:NofOFDMSymbol
      H_est(i,:) = MMSE_time_freq(NFFT,H_LS(i,:),H(i,:),beta,snr_linear,h_length);
end
% for i = 1:NofOFDMSymbol
%       H_est(i,:) = MMSE_time_freq_new(NFFT,H_LS(i,:),snr_linear,D_f,h_CIR(i,:));
% end
end

















