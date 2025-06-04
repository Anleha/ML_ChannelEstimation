function H_LS = LS_CE_MIMO(Y,Xp1,Xp2,Xp3,Xp4,D_t,NFFT, h_length,NofOFDMSymbol)
N_P = h_length;

%----------------------------------------
% FFT matrix
%----------------------------------------

F = [];

for k=0:NFFT-1
    W_tem = [];
    for n = 0:NFFT-1
        W_tem = [W_tem,exp(-1i*2*pi*n*k/NFFT)];
    end
    F = [F;W_tem];
end

%LS fillter coeficients
PP = [diag(Xp1)*F(:,1:N_P),diag(Xp2)*F(:,1:N_P),diag(Xp3)*F(:,1:N_P),diag(Xp4)*F(:,1:N_P)];
Q = pinv(PP'*PP);
H_LS = zeros(NFFT,NofOFDMSymbol);
h_LS = zeros(4*h_length,NofOFDMSymbol);
pilot_loc = [];
index = 1;

%LS_estimation
for i = 1:NofOFDMSymbol
    
   if ( mod(i-1,D_t) == 0)
       h_LS_temp(:,index) = Q * PP' * Y(i,:).';
       
       H_LS_temp_1(:,index) = fft(h_LS_temp(1:h_length,index),NFFT);
       H_LS_temp_2(:,index) = fft(h_LS_temp(h_length+1:h_length*2,index),NFFT);
       H_LS_temp_3(:,index) = fft(h_LS_temp(h_length*2+1:h_length*3,index),NFFT);
       H_LS_temp_4(:,index) = fft(h_LS_temp(h_length*3+1:h_length*4,index),NFFT);
       
       H_LS_temp(:,index) = [H_LS_temp_1(:,index); H_LS_temp_2(:,index);H_LS_temp_3(:,index);H_LS_temp_4(:,index)];
       index = index + 1;
       
       pilot_loc = [pilot_loc i];   %Pilot location TD
       
   end
  
end
H_temp = [];
%interpolation
if pilot_loc(end) < NofOFDMSymbol
    for i = 1:NFFT*4
        slope = (H_LS_temp(i,end)-H_LS_temp(i,end-1))/(pilot_loc(end)-pilot_loc(end-1));
        H_temp = [H_temp; (H_LS_temp(i,end)+slope*(NofOFDMSymbol-pilot_loc(end)))];
    end
    pilot_loc = [pilot_loc NofOFDMSymbol];
end
H_LS_temp = [H_LS_temp, H_temp];
for i=1:NFFT*4
    H_LS(i,:) = interp1(pilot_loc,H_LS_temp(i,:),1:NofOFDMSymbol,'spline');
end
       
       





















