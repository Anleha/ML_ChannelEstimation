%clc;
clear

clearvars;
%Gen CNN or LSTM???
%choose mode: 
train_mode = 2; %1-CNN/ 2-LSTM/ 3-DNN




NFFT = 256;               % FFT length
% G = NFFT/8;               % Guard interval length
G = 18;
M_ary = 16;                % Multilevel of M-ary symbol

P_A = sqrt(2);           % Amplitude of pilot symbol

D_f = 2;                 % Pilot distance in frequency domain
D_t = 2;                 % pilot distance in time domain


NofZeros = D_f-1;
M = NFFT / (D_f);        % Number of pilot symbol  per OFDM symbol

%SNR in dB
snr = -5:5:25;

%SNR linear
snr_linear = 10.^(snr*0.1); 
beta = 17/9;


%-------------------------------------------------
% Parameters channel
%-------------------------------------------------

f_dmax = 200;   % Maximum Doppler frequency
channel_profile = 'TDL-C';  %PDP profile
numTx = 4;  %num of transmitted antenna
numRx = 4;  %num of received antenna
f_s = 3.84e6;
T_a = 1/(f_s); %sampling cycle

MSE_LS = zeros(1,length(snr));
err_LS = zeros(1,length(snr));
err_MMSEeq_LS = err_LS;

MSE_MMSE = zeros(1,length(snr));
err_MMSE = zeros(1,length(snr));

MSE_CNN = zeros(1,length(snr));

MSE_LSTM = zeros(1,length(snr));

NofOFDMSymbol = 20;        %Number of  total (data and pilot OFDM symbol)

No_Of_OFDM_Data_Symbol = NofOFDMSymbol-ceil(NofOFDMSymbol/D_t);
                            %Number of datay symbols
Tsym = NFFT*T_a;                                                      
length_data = numTx*(No_Of_OFDM_Data_Symbol) * NFFT;  
                            % The total data length
 
if train_mode == 1 
    Number_Repetation = 20;
elseif train_mode == 2
    Number_Repetation = 100;
else
    Number_Repetation = 10;
end

training_index = 0;

%data for DL models
training_in = [];
training_out = [];

XTrain_CNN = [];
YTrain_CNN = [];

XTrain_LSTM = [];
YTrain_LSTM = [];

Noisy_channel = [];
Channel = [];
%channel generation

channel = channel_gen(channel_profile, f_dmax, f_s, NofOFDMSymbol, NFFT, numTx, numRx);
h11_symbol = channel(:,:,1,1);
h12_symbol = channel(:,:,1,2);
h13_symbol = channel(:,:,1,3);
h14_symbol = channel(:,:,1,4);
h21_symbol = channel(:,:,2,1);
h22_symbol = channel(:,:,2,2);
h23_symbol = channel(:,:,2,3);
h24_symbol = channel(:,:,2,4);
h31_symbol = channel(:,:,3,1);
h32_symbol = channel(:,:,3,2);
h33_symbol = channel(:,:,3,3);
h34_symbol = channel(:,:,3,4);
h41_symbol = channel(:,:,4,1);
h42_symbol = channel(:,:,4,2);
h43_symbol = channel(:,:,4,3);
h44_symbol = channel(:,:,4,4);
h_length  = length(h22_symbol(1,:));  % Channel impulse response length

h11_initial = h11_symbol(1,:);

for NumofRep = 1:Number_Repetation 
     
    %Create data
    source_data = randi([0 1],log2(M_ary),length_data);

    
    %16-qam mod
    Demod_symbol = qammod(source_data,M_ary,'inputtype','bit').';
    Demod_symbol = Demod_symbol/sqrt(10);
    % Layer mapping
    
    [x0,x1,x2,x3] = layer_map(Demod_symbol, length_data, numTx);
    
    %Data matrix gen before IFFT
    
    Data_matrix_1 = reshape(x0,[No_Of_OFDM_Data_Symbol NFFT]);
    
    Data_matrix_2 = reshape(x1,[No_Of_OFDM_Data_Symbol NFFT]);
    
    Data_matrix_3 = reshape(x2,[No_Of_OFDM_Data_Symbol NFFT]);
    
    Data_matrix_4 = reshape(x3,[No_Of_OFDM_Data_Symbol NFFT]);

%Preparing Pilot Seq  
    %Anten 1
    PL_A1 = []; 
    ref_sig_1 = psudogen(M,0);
    for m = 0:M-1 
        PL_A1 = [PL_A1,ref_sig_1(m+1)];
        PL_A1=[PL_A1,zeros(1,NofZeros)]; 
    end
    %Anten 2
    PL_A2 = [];
    ref_sig_2 = psudogen(M,1);
    for m = 0:M-1
        PL_A2 = [PL_A2,ref_sig_2(m+1)];
        PL_A2=[PL_A2,zeros(1,NofZeros)]; 
    end
    %Anten 3
    PL_A3 = [];
    ref_sig_3 = psudogen(M,2);
    for m = 0:M-1
        PL_A3 = [PL_A3,ref_sig_3(m+1)];
        PL_A3=[PL_A3,zeros(1,NofZeros)]; 
    end
    %Anten 4
    PL_A4 = [];
    ref_sig_4 = psudogen(M,3);
    for m = 0:M-1
        PL_A4 = [PL_A4,ref_sig_4(m+1)];
        PL_A4=[PL_A4,zeros(1,NofZeros)]; 
    end
% Transmitted Signal Antenna 1 Time Domain(Insert Pilot)
    TS1_TD = Insert_PilotSymbol(PL_A1,Data_matrix_1,D_t,NofOFDMSymbol);
% Transmitted Signal Antenna 2 Time Domain(Insert Pilot)
    TS2_TD = Insert_PilotSymbol(PL_A2,Data_matrix_2,D_t,NofOFDMSymbol);
% Transmitted Signal Antenna 2 Time Domain(Insert Pilot)
    TS3_TD = Insert_PilotSymbol(PL_A3,Data_matrix_3,D_t,NofOFDMSymbol);
% Transmitted Signal Antenna 2 Time Domain(Insert Pilot)
    TS4_TD = Insert_PilotSymbol(PL_A4,Data_matrix_4,D_t,NofOFDMSymbol);
    
% FFT matrix 
    F = [];  
    for k=0:NFFT-1
        W_tem = [];
        for n = 0:NFFT-1
            W_tem = [W_tem,exp(-1i*2*pi*n*k/NFFT)];
        end
        F = [F;W_tem];
    end      
% Data modulation and transmittion
    for snr_index = 1:length(snr)
        rs11_frame = [];
        rs12_frame = [];
        rs13_frame = [];
        rs14_frame = [];
        rs21_frame = [];
        rs22_frame = [];
        rs23_frame = [];
        rs24_frame = [];
        rs31_frame = [];
        rs32_frame = [];
        rs33_frame = [];
        rs34_frame = [];
        rs41_frame = [];
        rs42_frame = [];
        rs43_frame = [];
        rs44_frame = [];
        initial_time=0;   

        for i = 0:NofOFDMSymbol-1
        
        %channel matrix
            H11_symbol(i+1,:) = fft(h11_symbol(i+1,:),NFFT);
            H12_symbol(i+1,:) = fft(h12_symbol(i+1,:),NFFT);
            H13_symbol(i+1,:) = fft(h13_symbol(i+1,:),NFFT);
            H14_symbol(i+1,:) = fft(h14_symbol(i+1,:),NFFT);
            H21_symbol(i+1,:) = fft(h21_symbol(i+1,:),NFFT);
            H22_symbol(i+1,:) = fft(h22_symbol(i+1,:),NFFT);
            H23_symbol(i+1,:) = fft(h23_symbol(i+1,:),NFFT);
            H24_symbol(i+1,:) = fft(h24_symbol(i+1,:),NFFT);
            H31_symbol(i+1,:) = fft(h31_symbol(i+1,:),NFFT);
            H32_symbol(i+1,:) = fft(h32_symbol(i+1,:),NFFT);
            H33_symbol(i+1,:) = fft(h33_symbol(i+1,:),NFFT);
            H34_symbol(i+1,:) = fft(h34_symbol(i+1,:),NFFT);
            H41_symbol(i+1,:) = fft(h41_symbol(i+1,:),NFFT);
            H42_symbol(i+1,:) = fft(h42_symbol(i+1,:),NFFT);
            H43_symbol(i+1,:) = fft(h43_symbol(i+1,:),NFFT);
            H44_symbol(i+1,:) = fft(h44_symbol(i+1,:),NFFT);
            
         %OFDM Modulation  
            %Data from anten 1
            OFDM_signal_tem = OFDM_Modulator(TS1_TD(i+1,:),NFFT,G);
            
            %data Tranmittion
            
            rs_11 = conv(OFDM_signal_tem, h11_symbol(i+1,:));
            
            rs_11 = awgn(rs_11,snr(snr_index),'measured','db');
            
            rs_12 = conv(OFDM_signal_tem, h12_symbol(i+1,:));
            rs_12 = awgn(rs_12,snr(snr_index),'measured','db');
            
            rs_13 = conv(OFDM_signal_tem, h13_symbol(i+1,:));
            rs_13 = awgn(rs_13,snr(snr_index),'measured','db');
            
            rs_14 = conv(OFDM_signal_tem, h14_symbol(i+1,:));
            rs_14 = awgn(rs_14,snr(snr_index),'measured','db');
           
            rs11_frame = [rs11_frame; rs_11];
            rs12_frame = [rs12_frame; rs_12];
            rs13_frame = [rs13_frame; rs_13];
            rs14_frame = [rs14_frame; rs_14];

            clear OFDM_signal_tem;
            
            %Data from anten 2
            OFDM_signal_tem = OFDM_Modulator(TS2_TD(i+1,:),NFFT,G);
            %data Tranmittion
            
            rs_21 = conv(OFDM_signal_tem, h21_symbol(i+1,:));
            rs_21 = awgn(rs_21,snr(snr_index),'measured','db');
            
            rs_22 = conv(OFDM_signal_tem, h22_symbol(i+1,:));
            rs_22 = awgn(rs_22,snr(snr_index),'measured','db');
            
            rs_23 = conv(OFDM_signal_tem, h23_symbol(i+1,:));
            rs_23 = awgn(rs_23,snr(snr_index),'measured','db');
            
            rs_24 = conv(OFDM_signal_tem, h24_symbol(i+1,:));
            rs_24 = awgn(rs_24,snr(snr_index),'measured','db');
            
            rs21_frame = [rs21_frame; rs_21];
            rs22_frame = [rs22_frame; rs_22];
            rs23_frame = [rs23_frame; rs_23];
            rs24_frame = [rs24_frame; rs_24];
    
            clear OFDM_signal_tem;
            
            %Data from anten 3
            OFDM_signal_tem = OFDM_Modulator(TS3_TD(i+1,:),NFFT,G);
            %data Tranmittion
            
            rs_31 = conv(OFDM_signal_tem, h31_symbol(i+1,:));
            rs_31 = awgn(rs_31,snr(snr_index),'measured','db');
            
            rs_32 = conv(OFDM_signal_tem, h32_symbol(i+1,:));
            rs_32 = awgn(rs_32,snr(snr_index),'measured','db');
            
            rs_33 = conv(OFDM_signal_tem, h33_symbol(i+1,:));
            rs_33 = awgn(rs_33,snr(snr_index),'measured','db');
            
            rs_34 = conv(OFDM_signal_tem, h34_symbol(i+1,:));
            rs_34 = awgn(rs_34,snr(snr_index),'measured','db');
            
            rs31_frame = [rs31_frame; rs_31];
            rs32_frame = [rs32_frame; rs_32];
            rs33_frame = [rs33_frame; rs_33];
            rs34_frame = [rs34_frame; rs_34];
    
            clear OFDM_signal_tem;
            
            %Data from anten 4
            OFDM_signal_tem = OFDM_Modulator(TS4_TD(i+1,:),NFFT,G);
            %data Tranmittion
            
            rs_41 = conv(OFDM_signal_tem, h41_symbol(i+1,:));
            rs_41 = awgn(rs_41,snr(snr_index),'measured','db');
            
            rs_42 = conv(OFDM_signal_tem, h42_symbol(i+1,:));
            rs_42 = awgn(rs_42,snr(snr_index),'measured','db');
            
            rs_43 = conv(OFDM_signal_tem, h43_symbol(i+1,:));
            rs_43 = awgn(rs_43,snr(snr_index),'measured','db');
            
            rs_44 = conv(OFDM_signal_tem, h44_symbol(i+1,:));
            rs_44 = awgn(rs_44,snr(snr_index),'measured','db');
            
            rs41_frame = [rs41_frame; rs_41];
            rs42_frame = [rs42_frame; rs_42];
            rs43_frame = [rs43_frame; rs_43];
            rs44_frame = [rs44_frame; rs_44];
    
            clear OFDM_signal_tem;
             
        end
        
       %OFDM Demod, CE and channel equalizer

        for i = 1:NofOFDMSymbol
            
            rs1_i = rs11_frame(i,:) + rs21_frame(i,:) + rs31_frame(i,:) + rs41_frame(i,:);
            rs2_i = rs12_frame(i,:) + rs22_frame(i,:) + rs32_frame(i,:) + rs42_frame(i,:);
            rs3_i = rs13_frame(i,:) + rs23_frame(i,:) + rs33_frame(i,:) + rs43_frame(i,:);
            rs4_i = rs14_frame(i,:) + rs24_frame(i,:) + rs34_frame(i,:) + rs44_frame(i,:);
            
            % OFDM Demod
            Demoded_data_A1(i,:) = OFDM_Demodulator(rs1_i,NFFT,NFFT,G); %Rx 1
            Demoded_data_A2(i,:) = OFDM_Demodulator(rs2_i,NFFT,NFFT,G); %Rx 2
            Demoded_data_A3(i,:) = OFDM_Demodulator(rs3_i,NFFT,NFFT,G); %Rx 3
            Demoded_data_A4(i,:) = OFDM_Demodulator(rs4_i,NFFT,NFFT,G); %Rx 4
            h_11_21(i,:) = [h11_symbol(i,:) h21_symbol(i,:)];
            h_12_22(i,:) = [h12_symbol(i,:) h22_symbol(i,:)];
        end
        
  
    %channel estimation
        %LS estimation
        %Anten 1
        H_est_LS_1 = LS_CE_MIMO(Demoded_data_A1,PL_A1,PL_A2,PL_A3,PL_A4,D_t,NFFT,h_length,NofOFDMSymbol);
        %Anten 2
        H_est_LS_2 = LS_CE_MIMO(Demoded_data_A2,PL_A1,PL_A2,PL_A3,PL_A4,D_t,NFFT,h_length,NofOFDMSymbol);
        %Anten 3
        H_est_LS_3 = LS_CE_MIMO(Demoded_data_A3,PL_A1,PL_A2,PL_A3,PL_A4,D_t,NFFT,h_length,NofOFDMSymbol);
        %Anten 4
        H_est_LS_4 = LS_CE_MIMO(Demoded_data_A4,PL_A1,PL_A2,PL_A3,PL_A4,D_t,NFFT,h_length,NofOFDMSymbol);
        
        H_est_LS_11 = H_est_LS_1(1:NFFT,:).';
        H_est_LS_21 = H_est_LS_1(NFFT+1:2*NFFT,:).';
        H_est_LS_31 = H_est_LS_1(2*NFFT+1:3*NFFT,:).';
        H_est_LS_41 = H_est_LS_1(3*NFFT+1:4*NFFT,:).';
        
        H_est_LS_12 = H_est_LS_2(1:NFFT,:).';
        H_est_LS_22 = H_est_LS_2(NFFT+1:2*NFFT,:).';  
        H_est_LS_32 = H_est_LS_2(2*NFFT+1:3*NFFT,:).';
        H_est_LS_42 = H_est_LS_2(3*NFFT+1:4*NFFT,:).';
        
        H_est_LS_13 = H_est_LS_3(1:NFFT,:).';
        H_est_LS_23 = H_est_LS_3(NFFT+1:2*NFFT,:).';  
        H_est_LS_33 = H_est_LS_3(2*NFFT+1:3*NFFT,:).';
        H_est_LS_43 = H_est_LS_3(3*NFFT+1:4*NFFT,:).';
        
        H_est_LS_14 = H_est_LS_4(1:NFFT,:).';
        H_est_LS_24 = H_est_LS_4(NFFT+1:2*NFFT,:).';  
        H_est_LS_34 = H_est_LS_4(2*NFFT+1:3*NFFT,:).';
        H_est_LS_44 = H_est_LS_4(3*NFFT+1:4*NFFT,:).';
        
    %training input for DL models      
     
     % Get training data for CNN  
        if train_mode == 1
            for train_symbol = 1:NofOFDMSymbol
                training_index = training_index + 1;     
                input_temp = [H_est_LS_11(train_symbol,:); H_est_LS_12(train_symbol,:); ...
                    H_est_LS_13(train_symbol,:); H_est_LS_14(train_symbol,:);...
                    H_est_LS_21(train_symbol,:); H_est_LS_22(train_symbol,:);...
                    H_est_LS_23(train_symbol,:); H_est_LS_24(train_symbol,:);...
                    H_est_LS_31(train_symbol,:); H_est_LS_32(train_symbol,:);...
                    H_est_LS_33(train_symbol,:); H_est_LS_34(train_symbol,:);...
                    H_est_LS_41(train_symbol,:); H_est_LS_42(train_symbol,:);...
                    H_est_LS_43(train_symbol,:); H_est_LS_44(train_symbol,:)];
                input_train = cat(3,real(input_temp),imag(input_temp));
                XTrain_CNN(:,:,:,training_index) =  input_train;
                output_temp = [H11_symbol(train_symbol,:); H12_symbol(train_symbol,:);...
                    H13_symbol(train_symbol,:); H14_symbol(train_symbol,:);...
                    H21_symbol(train_symbol,:); H22_symbol(train_symbol,:);...
                    H23_symbol(train_symbol,:); H24_symbol(train_symbol,:);...
                    H31_symbol(train_symbol,:); H32_symbol(train_symbol,:);...
                    H33_symbol(train_symbol,:); H34_symbol(train_symbol,:);...
                    H41_symbol(train_symbol,:); H42_symbol(train_symbol,:);...
                    H43_symbol(train_symbol,:); H44_symbol(train_symbol,:)];
                output_train = cat(3,real(output_temp),imag(output_temp));
                YTrain_CNN(:,:,:,training_index) = output_train;
            end
        elseif train_mode == 2
    % Get training data for LSTM
            for train_symbol = 1:NofOFDMSymbol
                temp = [H_est_LS_11(train_symbol,:); H_est_LS_12(train_symbol,:);...
                    H_est_LS_13(train_symbol,:); H_est_LS_14(train_symbol,:);...
                    H_est_LS_21(train_symbol,:); H_est_LS_22(train_symbol,:);...
                    H_est_LS_23(train_symbol,:); H_est_LS_24(train_symbol,:);...
                    H_est_LS_31(train_symbol,:); H_est_LS_32(train_symbol,:);...
                    H_est_LS_33(train_symbol,:); H_est_LS_34(train_symbol,:);...
                    H_est_LS_41(train_symbol,:); H_est_LS_42(train_symbol,:);...
                    H_est_LS_43(train_symbol,:); H_est_LS_44(train_symbol,:)];
                temp_out = [H11_symbol(train_symbol,:); H12_symbol(train_symbol,:);...
                    H13_symbol(train_symbol,:); H14_symbol(train_symbol,:);...
                    H21_symbol(train_symbol,:); H22_symbol(train_symbol,:);...
                    H23_symbol(train_symbol,:); H24_symbol(train_symbol,:);...
                    H31_symbol(train_symbol,:); H32_symbol(train_symbol,:);...
                    H33_symbol(train_symbol,:); H34_symbol(train_symbol,:);...
                    H41_symbol(train_symbol,:); H42_symbol(train_symbol,:);...
                    H43_symbol(train_symbol,:); H44_symbol(train_symbol,:)];
                
                for i = 1:16
                    input(2*i-1,:) = real(temp(i,:));
                    input(2*i,:) = imag(temp(i,:));
                    output(2*i-1,:) = real(temp_out(i,:));
                    output(2*i,:) = imag(temp_out(i,:));
                end
                training_index = training_index + 1;
                Noisy_channel(:,:,training_index) = temp;
                Channel(:,:,training_index) = temp_out;
                XTrain_LSTM = [XTrain_LSTM; num2cell(input,[1 2])];
                YTrain_LSTM = [YTrain_LSTM; num2cell(output,[1 2])];
            end
        else
        % Training data for DNN
            training_in = setUpTrain(H_est_LS_11,H_est_LS_12,H_est_LS_13,H_est_LS_14,...
                H_est_LS_21,H_est_LS_22,H_est_LS_23,H_est_LS_24,...
                H_est_LS_31,H_est_LS_32,H_est_LS_33,H_est_LS_34,...
                H_est_LS_41,H_est_LS_42,H_est_LS_43,H_est_LS_44,training_in);
            training_out = setUpTrain(H11_symbol,H12_symbol,H13_symbol,H14_symbol,...
                H21_symbol,H22_symbol,H23_symbol,H24_symbol,...
                H31_symbol,H32_symbol,H33_symbol,H34_symbol,...
                H41_symbol,H42_symbol,H43_symbol,H44_symbol,training_out);
        end
    end    
end
%save training data
switch f_dmax
    case 0
        if train_mode == 1
            save('CNN_training_data_MIMO', 'XTrain_CNN','YTrain_CNN')
        elseif train_mode == 2
            save('data_LSTM_MIMO','XTrain_LSTM','YTrain_LSTM')
            
        else
            save('data_DNN_MIMO','training_in','training_out')
        end
    case 100
        if train_mode == 1
            save('CNN_training_data_MIMO_100Hz', 'XTrain_CNN','YTrain_CNN')
        elseif train_mode == 2
            save('data_LSTM_MIMO_100Hz','XTrain_LSTM','YTrain_LSTM')
        else
            save('data_DNN_MIMO_100Hz','training_in','training_out')
        end
    case 200
        if train_mode == 1
            save('CNN_training_data_MIMO_200Hz', 'XTrain_CNN','YTrain_CNN')
           
        elseif train_mode == 2
            save('data_LSTM_MIMO_200Hz','XTrain_LSTM','YTrain_LSTM')
        else
            save('data_DNN_MIMO_200Hz','training_in','training_out')
        end
end
switch f_dmax
    case 0
        if train_mode == 1
            save('CNN_training_data_MIMO_ver2', 'XTrain_CNN','YTrain_CNN')
        elseif train_mode == 2
            save('data_LSTM_MIMO_ver2','XTrain_LSTM','YTrain_LSTM')
        else
            save('data_DNN_MIMO_ver2','training_in','training_out')
        end
    case 100
        if train_mode == 1
            save('CNN_training_data_MIMO_100Hz_ver2', 'XTrain_CNN','YTrain_CNN')
        elseif train_mode == 2
            save('data_LSTM_MIMO_100Hz_ver2','XTrain_LSTM','YTrain_LSTM')
        else
            save('data_DNN_MIMO_100Hz_ver2','training_in','training_out')
        end
    case 200
        if train_mode == 1
            save('CNN_training_data_MIMO_200Hz_ver2', 'XTrain_CNN','YTrain_CNN')
           
        elseif train_mode == 2
            save('data_LSTM_MIMO_200Hz_ver2','XTrain_LSTM','YTrain_LSTM')
        else
            save('data_DNN_MIMO_200Hz_ver2','training_in','training_out')
        end
    case 36
        if train_mode == 1
            save('CNN_training_data_MIMO_36Hz_ver2', 'XTrain_CNN','YTrain_CNN')
           
        elseif train_mode == 2
            save('data_LSTM_MIMO_36Hz_ver2','XTrain_LSTM','YTrain_LSTM')
        else
            save('data_DNN_MIMO_36Hz_ver2','training_in','training_out')
        end
end


% save('data_LSTM_MIMO_test','XTrain_LSTM','YTrain_LSTM')








